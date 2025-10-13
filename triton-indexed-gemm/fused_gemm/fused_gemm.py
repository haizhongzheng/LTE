import torch
import tqdm
import triton
import triton.language as tl
import time

M, N, K = 16, 10240, 4096
# CUDA_VISIBLE_DEVICES=3 VERBOSE=1 numactl -i 1 python3 fused_gemm/fused_gemm.py 
# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
    ],
    key=['M', 'N', 'L', 'K'],
)

@triton.jit
def indexed_matmul_fused_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
        l_ptr,
        # Matrix dimensions
        # ideal a (M, K), b (K, N), c (M, L), d (L, K), index (L)
        # actual a (M, K), b (N, K), c (M, L), d (N, K), index (L)
        M, N, K,
        L,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  # M x K
        stride_bn, stride_bk,  # N x K
        stride_cm, stride_cn,  # M x L
        stride_dn, stride_dk,  # N x K
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M) # the number of programs we need for M dim
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) # the number of programs we need for N dim
    # num_pid_l = tl.cdiv(L, BLOCK_SIZE_K) # the number of programs we need for L dim
    # num_pid_k = tl.cdiv(K, BLOCK_SIZE_N) # the number of programs we need for K dim
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % L
    offs_bl = tl.load(l_ptr + offs_bn)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bl[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < L)
    tl.store(c_ptrs, c, mask=c_mask)
    
    # -----------------------------------------------------------
    # Create pointers for the first blocks of C and D.
    # We will advance this pointer as we move in the L direction
    # and accumulate
    # 'c_ptrs' is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # 'd_ptrs' is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # K is still the shared dim, but the length is L
    offs_cm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_dk = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % K
    c1_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_k[None, :]
    d_ptrs = d_ptr + stride_dk * offs_dk[None, :]
    
    # -----------------------------------------------------------
    # Iterate to compute a block of the E matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for l in range(0, tl.cdiv(L, BLOCK_SIZE_K)):
        offs_dn = tl.load(l_ptr + l * BLOCK_SIZE_K + offs_k, mask=offs_k < L - l * BLOCK_SIZE_K, other=N-1)
        # Load the next block of C and D, generate a mask by checking the L dimension.
        # If it is out of bounds, set it to 0.
        c1 = tl.load(c1_ptrs, mask=offs_k[None, :] < L - l * BLOCK_SIZE_K, other=0.0)
        d = tl.load(d_ptrs + offs_dn[:, None] * stride_dn, mask=offs_k[:, None] < L - l * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the L dimension.
        accumulator = tl.dot(c1, d, accumulator)
        # Advance the ptrs to the next L block.
        c1_ptrs += BLOCK_SIZE_K * stride_cn
    
    e = accumulator.to(tl.float16)
    
    # -----------------------------------------------------------
    # Write back the block of the output matrix E with masks.
    # use A to store E as they have the same shape
    offs_em = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ek = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    e_ptrs = e_ptr + stride_am * offs_em[:, None] + stride_ak * offs_ek[None, :]
    e_mask = (offs_em[:, None] < M) & (offs_ek[None, :] < K)
    tl.store(e_ptrs, e, mask=e_mask)

def indexed_matmul_fused(a, b, c, d, e, index, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    N, K = b.shape
    L = index.shape[0]

    # print(M,K,N,L)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    indexed_matmul_fused_kernel[grid](
        a, b, c, d, e,#
        index,
        M, N, K,  #
        L,
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        d.stride(0), d.stride(1),  #
        ACTIVATION=activation  #
    )
    return e


index = torch.arange(0, N, 16).to('cuda')
L = index.shape[0]
print(f"Index shape: {L}/{N}")
# index = torch.tensor([0, 1, 2, 3, 512, 513, 514, 1000, 1002]).to('cuda')
a = torch.randn((M, K), device='cuda', dtype=torch.float16)
b = torch.randn((N, K), device='cuda', dtype=torch.float16) # b is transposed
c = torch.empty((M, L), device=a.device, dtype=a.dtype)
d = torch.randn((N, K), device=a.device, dtype=torch.float16)     # d is also transposed
e = torch.empty((M, K), device=a.device, dtype=a.dtype)

# computation
# a @ b.T[index] = c, then c.T @ d = e
triton_output = indexed_matmul_fused(a, b, c, d, e, index, "leaky_relu")
torch.cuda.synchronize()

tc = a @ b[index].T
tc = torch.nn.functional.leaky_relu(tc)
te = tc @ d[index]
torch.cuda.synchronize()

print(f"C triton output: {c.shape}\n", c)

print(f"C pytorch output: {tc.shape}\n", tc)

print(f"E triton output: {triton_output.shape}\n", triton_output)

print(f"E pytorch output: {te.shape}\n", te)

start = time.perf_counter()
for i in range(100):
    triton_output = indexed_matmul_fused(a, b, c, d, e, index, "leaky_relu")
    torch.cuda.synchronize()
end = time.perf_counter()
print(f"Triton matmul time {(end - start)/100}s")
# print(triton_output)

start = time.perf_counter()
for i in range(100):
    c = a @ b[index].T
    c = torch.nn.functional.leaky_relu(c)
    e = c @ d[index]
    torch.cuda.synchronize()
end = time.perf_counter()
print(f"Pytorch matmul time {(end - start)/100}s")
