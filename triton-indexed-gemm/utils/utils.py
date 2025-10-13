import torch

import triton
import triton.language as tl

cuda_configs = []

def is_cuda():
    return True


def is_hip_mi200():
    return False

def init_cuda_autotune_config(BLOCK_SIZE_Ms, BLOCK_SIZE_Ns, BLOCK_SIZE_Ks, GROUP_SIZE_Ms, num_stagess, num_warpss):
    global cuda_configs
    for bs_m in BLOCK_SIZE_Ms:
        for bs_n in BLOCK_SIZE_Ns:
            for bs_k in BLOCK_SIZE_Ks:
                for gs_m in GROUP_SIZE_Ms:
                    for num_stages in num_stagess:
                        for num_warps in num_warpss:
                            cuda_configs.append(
                                triton.Config({'BLOCK_SIZE_M': bs_m, 'BLOCK_SIZE_N': bs_n, 'BLOCK_SIZE_K': bs_k, 'GROUP_SIZE_M': gs_m},
                                              num_stages=num_stages, num_warps=num_warps))

    print(f"Initialized {len(cuda_configs)} configs")


def get_cuda_autotune_config():
    global cuda_configs
    if len(cuda_configs) == 0:
        # init_cuda_autotune_config([16], [16, 32, 64, 128, 256], [32, 64, 128, 256, 512], [1, 2], [2, 3, 4, 5], [8, 16])
        init_cuda_autotune_config([16], [16, 32, 64], [16, 32, 64, 128, 256, 512], [1], [3], [8, 16])
        # init_cuda_autotune_config([16], [64], [64], [1], [3], [16])
    return cuda_configs

# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    # x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)

@triton.jit
def gelu(x):
    # using tl.math.erf as a custom operation
    return 0.5 * x * (1.0 + tl.math.erf(x / 1.41421))

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)