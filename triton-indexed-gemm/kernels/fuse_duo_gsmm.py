import triton
import triton.language as tl
import torch

from utils.utils import get_cuda_autotune_config, leaky_relu, gelu, silu

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_cuda_autotune_config(),
    # key=['N', 'K', 'L'],
    key=['M', 'N', 'K', 'L'],
)
@triton.jit
def indexed_matmul_duo_kernel(
        # Pointers to matrices
        a_ptr, b0_ptr, b1_ptr, d_ptr, e_ptr,
        l_ptr,
        # Matrix dimensions
        # ideal a (M, K), b (K, L), c (M, L), d (L, K), index (L)
        # actual a (M, K), b (N, K), c (M, L), d (N, K), index (L)
        M, N, K,
        L,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  # M x K
        stride_bn, stride_bk,  # N x K
        stride_dn, stride_dk,  # N x K
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(L, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # if pid_m >= num_pid_m or pid_n >= num_pid_n:
    #     return

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % L
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_bl = tl.load(l_ptr + offs_bn)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b0_ptrs = b0_ptr + (offs_bl[None, :] * stride_bn + offs_k[:, None] * stride_bk)
    b1_ptrs = b1_ptr + (offs_bl[None, :] * stride_bn + offs_k[:, None] * stride_bk)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # TODO: try (k + pid_n) % tl.cdiv(K, BLOCK_SIZE_K) to avoid congestion add
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b0 = tl.load(b0_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator0 = tl.dot(a, b0, accumulator0)

        b1 = tl.load(b1_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator1 = tl.dot(a, b1, accumulator1)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b0_ptrs += BLOCK_SIZE_K * stride_bk
        b1_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator0 = silu(accumulator0)
    accumulator = accumulator0 * accumulator1
    c = accumulator.to(a_ptr.dtype.element_ty)

    # -----------------------------------------------------------
    # Iterate to compute a block of subE
    # We put E block into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # Require atomic add when writing to E results.
    d_ptrs = d_ptr + (offs_bl[:, None] * stride_dn + offs_k[None, :] * stride_dk)
    e_ptrs = e_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    offs_em = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_dn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        d_mask = (offs_dn[:, None] < L) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        d = tl.load(d_ptrs, mask=d_mask, other=0.0)
        e = tl.dot(c, d)

        e_mask = (offs_em[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        tl.atomic_add(e_ptrs, e, mask=e_mask)
        d_ptrs += BLOCK_SIZE_K * stride_dk
        e_ptrs += BLOCK_SIZE_K * stride_ak

def indexed_matmul_fused(a, b0, b1, d, index, activation="silu"):
    # Check constraints. Comment this if it is slow
    assert a.shape[1] == b0.shape[1], "Incompatible dimensions"
    assert a.shape[1] == b1.shape[1], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b0.is_contiguous(), "Matrix B must be contiguous"
    assert b1.is_contiguous(), "Matrix B must be contiguous"
    assert d.is_contiguous(), "Matrix D must be contiguous"
    assert a.dtype == b0.dtype == b1.dtype == d.dtype, "Matrix A B D must be in the same type"
    assert activation == "silu", "Only support gelu activation for GPT2"
    M, K = a.shape
    N, K = b0.shape
    L = index.shape[0]

    e = torch.zeros((M, K), device=a.device, dtype=a.dtype)
    # e = torch.zeros((M, K), device=a.device, dtype=torch.float32)
    # print(M,K,N,L)
    # 1D launch kernel where each block gets its own program.
    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(L, META['BLOCK_SIZE_N']), )
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(L, META['BLOCK_SIZE_N']), )
    indexed_matmul_duo_kernel[grid](
        a, b0, b1, d, e,#
        index,
        M, N, K,  #
        L,
        a.stride(0), a.stride(1),  #
        b0.stride(0), b0.stride(1),  # B matrices have the same dim
        d.stride(0), d.stride(1),  #
    )
    return e
