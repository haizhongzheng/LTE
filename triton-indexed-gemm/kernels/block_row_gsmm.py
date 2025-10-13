import triton
import triton.language as tl
import torch

from utils.utils import get_cuda_autotune_config

GROUP_SIZE_L = 32
# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K', 'L'],
)
@triton.jit
def indexed_row_matmul_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        l_ptr,   # l is the index of L group of rows, not each row
        # A (M, L), B (K, N), select B rows by index, K > L
        # Matrix dimensions
        M, N, K, # select L rows out of K
        L, # L is the target dimension, K > L
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        L_div: tl.constexpr,  # length of indices of k_groups, L = L_div * GROUP_SIZE_L
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        GROUP_SIZE_L: tl.constexpr,  # row group size, normally <= BLOCK_SIZE_K, BLOCK_SIZE_K dividable by GROUP_SIZE_L 
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Static check
    tl.static_assert(L_div * GROUP_SIZE_L >= BLOCK_SIZE_K, "L_div * GROUP_SIZE_L must be >= BLOCK_SIZE_K")
    tl.static_assert(GROUP_SIZE_L <= BLOCK_SIZE_K, "GROUP_SIZE_L must be <= BLOCK_SIZE_K")
    tl.static_assert(BLOCK_SIZE_K % GROUP_SIZE_L == 0, "BLOCK_SIZE_K must be divisible by GROUP_SIZE_L")
    
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
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
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # 512 * GROUP_SIZE_L // BLOCK_SIZE_K must be power of 2
    offs_l = tl.reshape(tl.arange(0, 512), (512 * GROUP_SIZE_L // BLOCK_SIZE_K, BLOCK_SIZE_K // GROUP_SIZE_L))
    # NOTE: the following way is more robust but triton doesn't support step arange
    # offs_l = tl.arange(0, L_div, BLOCK_SIZE_K // GROUP_SIZE_L)[:, None] + tl.arange(0, BLOCK_SIZE_K // GROUP_SIZE_L)[None, :]
    offs_div = tl.load(l_ptr + offs_l, mask=offs_l < L_div, other=0) # 2d array of indices depends on how many L groups per K-Block
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + offs_bn[None, :] * stride_bn

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    offs_gl = tl.arange(0, GROUP_SIZE_L)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for l in range(0, tl.cdiv(L, BLOCK_SIZE_K)):
        offs_bb = tl.sum(offs_div * (offs_l//(BLOCK_SIZE_K // GROUP_SIZE_L) == l), axis=0) 
        offs_bl = (offs_bb[:, None] + offs_gl[None, :])
        offs_bl = tl.ravel(offs_bl)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < L - l * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs + offs_bl[:, None] * stride_bk, mask=offs_k[:, None] < L - l * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def indexed_matmul(a, b, index, activation=""):
    # Check constraints.
    # assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, L = a.shape
    K, N = b.shape
    L_div = index.shape[0] 
    assert L != 0 and K != 0, "Index length and K must be non-zero"
    assert L % GROUP_SIZE_L == 0, "Indexed dimension must be multiple of GROUP_SIZE_L"
    assert L == L_div * GROUP_SIZE_L, "L must be L_div * GROUP_SIZE_L"
    # 1D launch kernel where each block gets its own program.
    
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    indexed_row_matmul_kernel[grid](
        a, b, c,  #
        index,
        M, N, K,  #
        L,
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        L_div=L_div,  # index length is now a const parameter
        GROUP_SIZE_L=GROUP_SIZE_L,  #
    )
    return c
