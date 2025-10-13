import torch

import triton
import triton.language as tl

from utils.utils import is_cuda
from kernels.basic_gemm import matmul
from kernels.col_gsmm import indexed_matmul as col_gsmm
from kernels.row_gsmm import indexed_matmul as row_gsmm
from kernels.fuse_bias_gsmm import indexed_matmul_fused as fuse_gsmm
from kernels.fuse_bias_gsmm import indexed_matmul_fused_kernel as fuse_gsmm_kernel

ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

DATA_TYPE = torch.float16

configs = []

configs.append(
    triton.testing.Benchmark(
        x_names=["L"],  # Argument names to use as an x-axis for the plot
        x_vals=[32 * i for i in range(8, 128, 8)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=[ref_lib.lower(), "triton_indexed"],  # Label name for the lines
        line_names=[ref_lib, "Triton_indexed"],  # Line styles
        styles=[("green", "-"), ("red", "-")],  # Line colors and styles
        # ylabel="TFLOPS",  # Label name for the y-axis
        ylabel="Time (us)",  # Label name for the y-axis
        # plot_name="fidx-matmul-performance-" + "tflops-" +
        plot_name="fidx-matmul-performance-" + "time-" +
        ("fp16"),  # Name for the plot, used also as a file name for saving the plot.
        # args={"M": 16, "K": 4096, "N": 10240, "indices": torch.arange(10240)[torch.randperm(10240)]},  # Constant arguments to pass to `benchmark`
        args={"M": 16, "K": 1024, "N": 4096, "indices": torch.arange(4096)[torch.randperm(4096)]},  # Constant arguments to pass to `benchmark`
    ))

def fnTorch(a, b, d, bias1, bias2):
    c = a @ b + bias1[None, :]
    c = torch.nn.functional.gelu(c)
    e = c @ d + bias2[None, :]
    return e

@triton.testing.perf_report(configs)
def benchmark_fuse(L, provider, M, K, N, indices):
    index = indices[:L].sort()[0].cuda()
    a = torch.randn((M, K), device='cuda', dtype=DATA_TYPE)
    bias2 = torch.randn((K), device='cuda', dtype=DATA_TYPE)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        b = torch.randn((K, L), device='cuda', dtype=DATA_TYPE)
        d = torch.randn((L, K), device='cuda', dtype=DATA_TYPE)
        bias1 = torch.randn((L), device='cuda', dtype=DATA_TYPE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fnTorch(a, b, d, bias1, bias2), quantiles=quantiles)
    if provider == 'triton_indexed':
        # print(f"Running Triton Indexed with L={L}, M={M}, K={K}, N={N}")
        b = torch.randn((N, K), device='cuda', dtype=DATA_TYPE)
        d = torch.randn((N, K), device='cuda', dtype=DATA_TYPE)
        bias1 = torch.randn((N), device='cuda', dtype=DATA_TYPE)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fuse_gsmm(a, b, d, bias1, bias2, index), quantiles=quantiles)
        print(f"Best config L={L}", fuse_gsmm_kernel.best_config)
    # perf = lambda ms: 2 * M * L * K * 1e-12 / (ms * 1e-3)
    perf = lambda ms: ms * 1e3
    return perf(ms), perf(max_ms), perf(min_ms)

if "__main__" in __name__:
    benchmark_fuse.run(print_data=True, show_plots=False, save_path="performance")

