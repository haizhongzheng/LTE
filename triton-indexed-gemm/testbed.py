import torch

import triton
import triton.language as tl

from utils.utils import is_cuda
from kernels.basic_gemm import matmul
from kernels.col_gsmm import indexed_matmul as col_gsmm
from kernels.row_gsmm import indexed_matmul as row_gsmm

ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

col_configs = []

col_configs.append(
    triton.testing.Benchmark(
        x_names=["L"],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(16, 81, 2)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=[ref_lib.lower(), "triton", "triton_indexed"],  # Label name for the lines
        line_names=[ref_lib, "Triton", "Triton_indexed"],  # Line styles
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],  # Line colors and styles
        # ylabel="TFLOPS",  # Label name for the y-axis
        ylabel="Time (us)",  # Label name for the y-axis
        # plot_name="matmul-performance-" + "tflops-" + 
        plot_name="colidx-matmul-performance-" + "time-" + 
        ("fp16"),  # Name for the plot, used also as a file name for saving the plot.
        args={"M": 16, "K": 4096, "N": 10240, "indices": torch.arange(10240)[torch.randperm(10240)]},  # Constant arguments to pass to `benchmark`
    ))

@triton.testing.perf_report(col_configs)
def benchmark_col(L, provider, M, K, N, indices):
    index = indices[:L].sort()[0].cuda()
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        b = torch.randn((K, L), device='cuda', dtype=torch.float16)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        b = torch.randn((K, L), device='cuda', dtype=torch.float16)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    if provider == 'triton_indexed':
        # print(f"Running Triton Indexed with L={L}, M={M}, K={K}, N={N}")
        b = torch.randn((N, K), device='cuda', dtype=torch.float16)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: col_gsmm(a, b, index), quantiles=quantiles)
    # perf = lambda ms: 2 * M * L * K * 1e-12 / (ms * 1e-3)
    perf = lambda ms: ms * 1e3
    return perf(ms), perf(max_ms), perf(min_ms)

row_configs = []

row_configs.append(
    triton.testing.Benchmark(
        x_names=["L"],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(16, 81, 2)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=[ref_lib.lower(), "triton", "triton_indexed"],  # Label name for the lines
        line_names=[ref_lib, "Triton", "Triton_indexed"],  # Line styles
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],  # Line colors and styles
        # ylabel="TFLOPS",  # Label name for the y-axis
        ylabel="Time (us)",  # Label name for the y-axis
        # plot_name="matmul-performance-" + "tflops-" + 
        plot_name="rowidx-matmul-performance-" + "time-" + 
        ("fp16"),  # Name for the plot, used also as a file name for saving the plot.
        args={"M": 16, "K": 10240, "N": 4096, "indices": torch.arange(10240)[torch.randperm(10240)]},  # Constant arguments to pass to `benchmark`
    ))

@triton.testing.perf_report(row_configs)
def benchmark_row(L, provider, M, K, N, indices):
    index = indices[:L].sort()[0].cuda()
    a = torch.randn((M, L), device='cuda', dtype=torch.float16)   
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        b = torch.randn((L, N), device='cuda', dtype=torch.float16)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        b = torch.randn((L, N), device='cuda', dtype=torch.float16)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    if provider == 'triton_indexed':
        # print(f"Running Triton Indexed with L={L}, M={M}, K={K}, N={N}")
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: row_gsmm(a, b, index), quantiles=quantiles)
    # perf = lambda ms: 2 * M * L * K * 1e-12 / (ms * 1e-3)
    perf = lambda ms: ms * 1e3
    return perf(ms), perf(max_ms), perf(min_ms)

if "__main__" in __name__:
    benchmark_col.run(print_data=True, show_plots=False, save_path="performance")
    # benchmark_row.run(print_data=True, show_plots=False, save_path="performance")

