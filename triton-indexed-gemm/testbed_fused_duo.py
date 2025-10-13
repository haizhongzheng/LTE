import torch

import triton
import triton.language as tl

from utils.utils import is_cuda
from kernels.fuse_duo_gsmm import indexed_matmul_fused as duo_gsmm
from kernels.fuse_duo_gsmm import indexed_matmul_duo_kernel as duo_kernel

ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

DATA_TYPE = torch.float16

configs = []

configs.append(
    triton.testing.Benchmark(
        x_names=["L"],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(8, 81, 4)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
        line_vals=[ref_lib.lower(), "triton_indexed"],  # Label name for the lines
        line_names=[ref_lib, "Triton_indexed"],  # Line styles
        styles=[("green", "-"), ("red", "-")],  # Line colors and styles
        # ylabel="TFLOPS",  # Label name for the y-axis
        ylabel="Time (us)",  # Label name for the y-axis
        # plot_name="fidx-matmul-performance-" + "tflops-" +
        plot_name="fdidx-matmul-performance-" + "time-" +
        ("fp16"),  # Name for the plot, used also as a file name for saving the plot.
        args={"M": 1, "K": 4096, "N": 11008, "indices": torch.arange(11008)[torch.randperm(11008)]},  # Constant arguments to pass to `benchmark`
    ))

def fnTorch(a, b0, b1, d):
    c0 = a @ b0
    c1 = a @ b1
    c0 = torch.nn.functional.silu(c0)
    c = c0 * c1
    e = c @ d
    return e

@triton.testing.perf_report(configs)
def benchmark_fuse(L, provider, M, K, N, indices):
    index = indices[:L].sort()[0].cuda()
    a = torch.randn((M, K), device='cuda', dtype=DATA_TYPE)/20
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        b0 = torch.randn((K, L), device='cuda', dtype=DATA_TYPE)/20
        b1 = torch.randn((K, L), device='cuda', dtype=DATA_TYPE)/20
        d = torch.randn((L, K), device='cuda', dtype=DATA_TYPE)/100
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fnTorch(a, b0, b1, d), quantiles=quantiles)
    if provider == 'triton_indexed':
        # print(f"Running Triton Indexed with L={L}, M={M}, K={K}, N={N}")
        b0 = torch.randn((N, K), device='cuda', dtype=DATA_TYPE)/20
        b1 = torch.randn((N, K), device='cuda', dtype=DATA_TYPE)/20
        d = torch.randn((N, K), device='cuda', dtype=DATA_TYPE)/100
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: duo_gsmm(a, b0, b1, d, index), quantiles=quantiles)
        print(f"Best config L={L}", duo_kernel.best_config)
    # perf = lambda ms: 2 * M * L * K * 1e-12 / (ms * 1e-3)
    perf = lambda ms: ms * 1e3
    return perf(ms), perf(max_ms), perf(min_ms)

if "__main__" in __name__:
    perf_df = benchmark_fuse.run(print_data=True, show_plots=False, save_path="performance", return_df=True)
    

