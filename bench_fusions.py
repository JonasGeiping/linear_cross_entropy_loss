"""This file is a trashfire. But it benchmarks fusions."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math

import triton

import torch._dynamo.config

torch._dynamo.config.cache_size_limit = 4096  # just in case, we do want to recompile a bunch of settings


device = torch.device("cuda:0")
torch.cuda.device_count()


f = 1  # make larger to let test go fast. f=1 is target size


def cosim(x, y):
    return (
        (x.reshape(-1).double() * y.reshape(-1).double()).sum()
        / x.reshape(-1).double().norm()
        / y.reshape(-1).double().norm()
    ).float()


@torch._dynamo.disable
def baseline_torch(x, y, A):
    V = A.shape[0]
    return F.cross_entropy(F.linear(x, A).view(-1, V).float(), y.view(-1))


@torch.compile  # need to define this twice, otherwise there's weird shadowing happening in the notebook
def compiled_baseline(x, y, A):
    V = A.shape[0]
    return F.cross_entropy(F.linear(x, A).view(-1, V).float(), y.view(-1))


def torch_mixed_precision(x, y, A):
    V = A.shape[0]
    with torch.autocast(device_type="cuda"):
        return F.cross_entropy(F.linear(x, A).view(-1, V).float(), y.view(-1))


compiled_amp = torch.compile(torch_mixed_precision)


def _inner_function(x_block, y_block, A, num_blocks):
    return F.cross_entropy(F.linear(x_block, A), y_block) / num_blocks


@torch.compile(dynamic=True)
def torch_compiled_checkpoint(x, y, A, default_chunk_size=4096):
    loss = 0.0
    _, H = A.shape
    N = x.view(-1, H).shape[0]
    chunk_size = min(default_chunk_size, N)
    if chunk_size % N != 0:
        chunk_size = math.gcd(N, default_chunk_size)
    x_blocks = x.view(-1, H).split(chunk_size)
    y_blocks = y.view(-1).split(chunk_size)

    for x_block, y_block in zip(x_blocks, y_blocks):
        loss += checkpoint(_inner_function, x_block, y_block, A, num_blocks=len(y_blocks), use_reentrant=False)  # type: ignore
    return loss


def torch_checkpoint(x, y, A, default_chunk_size=4096):
    loss = 0.0
    _, H = A.shape
    N = x.view(-1, H).shape[0]
    chunk_size = min(default_chunk_size, N)
    if chunk_size % N != 0:
        chunk_size = math.gcd(N, default_chunk_size)
    x_blocks = x.view(-1, H).split(chunk_size)
    y_blocks = y.view(-1).split(chunk_size)

    for x_block, y_block in zip(x_blocks, y_blocks):
        loss += checkpoint(_inner_function, x_block, y_block, A, num_blocks=len(y_blocks), use_reentrant=False)  # type: ignore
    return loss


maxauto_baseline = torch.compile(baseline_torch, fullgraph=True, mode="max-autotune-no-cudagraphs")


from variants.highmem import linear_cross_entropy as linear_cross_entropy_highmem
from variants.double_recomp2 import linear_cross_entropy as linear_cross_entropy_double_recomp
from variants.double_recomp3 import linear_cross_entropy as linear_cross_entropy_parallel_recomp
from variants.manyway_recomp import linear_cross_entropy as linear_cross_entropy_manyway
from variants.malek_xent import linear_cross_entropy as efficient_xent

from variants.malek_like_with_my_fwd import linear_cross_entropy as linear_cross_entropy_z_chunks
from variants.malek_like_with_my_fwdNV import linear_cross_entropy as linear_cross_entropy_z_chunksNV

# from malek_like_with_my_fwdNV3 import linear_cross_entropy as linear_cross_entropy_z_chunks_full_fused
from variants.malek_like_with_my_fwdNV5 import linear_cross_entropy as linear_cross_entropy_z_chunksNV_v2
from variants.final_fusion import linear_cross_entropy as linear_cross_entropy_z_chunks_full_fused
from variants.final_fusion_parallel import linear_cross_entropy as linear_cross_entropy_z_chunks_full_fused_parallel
from variants.final_fusion_parallel_splitN_V import (
    linear_cross_entropy as linear_cross_entropy_z_chunks_full_fused_parallel_split,
)
from variants.final_fusion_parallel_splitN_V_usable import linear_cross_entropy as linear_cross_entropy_final
from variants.final_fusion_usable_variant import linear_cross_entropy as linear_cross_entropy_final_var
from variants.final_fusion_VII import linear_cross_entropy as linear_cross_entropy_final_with_all_tricks
from variants.final_fusion_deployed import linear_cross_entropy as deployed_fusion

# # Benchmarking FWD + BWD


range_dict = {"H": range(8, 14, 1), "V": range(8, 18, 1), "N": range(8, 17, 1)}
# range_dict = {"H": range(8, 9, 1), "V": range(9, 10, 1), "N": range(8, 9, 1)}  # quick debug run

method_list = [
    "torch-baseline",
    # # "torch-base-float32",
    "torch-compile",
    "maxauto-compile",
    "torch-checkpoint",
    # # "torch-base-amp",
    # # "compiled-amp",
    # # "torch-compile-checkpoint",
    # # "triton",
    # # "triton-recomp",
    # # "triton-par-recomp",
    # # "triton-many-recomp",
    "malek",
    # # "triton-z-chunks-in-sram",
    # "triton-z-chunks-in-sram-NV",
    # # "triton-z-chunks-in-sram-NV-v2",
    # "triton-z-chunks-in-sram-fused",
    # "triton-z-chunks-in-sram-fusedP",
    # "triton-z-chunks-in-sram-fusedP-split",
    # "triton-z-chunks-in-sram-final",
    # "triton-z-chunks-in-sram-final-var",
    # "triton-z-chunks-in-sram-fused-2k",
    # "triton-z-chunks-in-sramNV-v2-2k",
    # "triton-z-chunks-in-sramNV-05k",
    # "triton-z-chunks-in-sram-fusedP-2k",
    # "triton-z-chunks-in-sram-fusedP-05k",
    # "triton-z-chunks-in-sram-vii",
    "triton-z-chunks-in-sram-deployed",
]


def provider_lookup(provider, x, y, At, A):
    fn = None
    if provider == "torch-baseline":
        fn = lambda: baseline_torch(x, y, A)
    if provider == "torch-compile":
        fn = lambda: compiled_baseline(x, y, A)
    if provider == "maxauto-compile":
        fn = lambda: maxauto_baseline(x, y, A)
    if provider == "torch-compile-checkpoint":
        fn = lambda: torch_compiled_checkpoint(x, y, A)
    if provider == "torch-checkpoint":
        fn = lambda: torch_checkpoint(x, y, A)
    if provider == "triton":
        fn = lambda: linear_cross_entropy_highmem(x, y, At)
    if provider == "triton-recomp":
        fn = lambda: linear_cross_entropy_double_recomp(x, y, At)
    if provider == "triton-par-recomp":
        fn = lambda: linear_cross_entropy_parallel_recomp(x, y, At)
    if provider == "malek":
        fn = lambda: efficient_xent(x, y, A)
    if provider == "triton-many-recomp":
        fn = lambda: linear_cross_entropy_manyway(x, y, At)
    if provider == "triton-z-chunks-in-sram":
        fn = lambda: linear_cross_entropy_z_chunks(x, y, At)
    if provider == "triton-z-chunks-in-sram-NV":
        fn = lambda: linear_cross_entropy_z_chunksNV_v2(x, y, At)
    if provider == "triton-z-chunks-in-sram-fused":
        fn = lambda: linear_cross_entropy_z_chunks_full_fused(x, y, At, N_chunk_size=4096)
    if provider == "triton-z-chunks-in-sram-fusedP":
        fn = lambda: linear_cross_entropy_z_chunks_full_fused_parallel(x, y, At, N_chunk_size=4096)
    if provider == "triton-z-chunks-in-sram-fusedP-2k":
        fn = lambda: linear_cross_entropy_z_chunks_full_fused_parallel(x, y, At, N_chunk_size=2048)
    if provider == "triton-z-chunks-in-sram-fusedP-05k":
        fn = lambda: linear_cross_entropy_z_chunks_full_fused_parallel(x, y, At, N_chunk_size=512)
    if provider == "triton-z-chunks-in-sram-fusedP-split":
        fn = lambda: linear_cross_entropy_z_chunks_full_fused_parallel_split(x, y, At, N_chunk_size=4096)
    if provider == "triton-z-chunks-in-sram-fused-2k":
        fn = lambda: linear_cross_entropy_z_chunks_full_fused(x, y, At, N_chunk_size=2048)
    if provider == "triton-z-chunks-in-sramNV-2k":
        fn = lambda: linear_cross_entropy_z_chunksNV_v2(x, y, At, N_chunk_size=2048)
    if provider == "triton-z-chunks-in-sramNV-05k":
        fn = lambda: linear_cross_entropy_z_chunksNV(x, y, At, N_chunk_size=512)
    if provider == "torch-base-amp":
        fn = lambda: torch_mixed_precision(x.float(), y, A.float())
    if provider == "compiled-amp":
        fn = lambda: compiled_amp(x.float(), y, A.float())
    if provider == "torch-base-float32":
        fn = lambda: baseline_torch(x.float(), y, A.float())
    if provider == "triton-z-chunks-in-sram-final":
        fn = lambda: linear_cross_entropy_final(x, y, At, N_chunk_size=4096)
    if provider == "triton-z-chunks-in-sram-final-var":
        fn = lambda: linear_cross_entropy_final_var(x, y, At, N_chunk_size=4096)
    if provider == "triton-z-chunks-in-sram-vii":
        fn = lambda: linear_cross_entropy_final_with_all_tricks(x, y, At, N_chunk_size=4096)
    if provider == "triton-z-chunks-in-sram-deployed":
        fn = lambda: deployed_fusion(x, y, At, N_chunk_size=4096)

    if fn is None:
        raise ValueError(f"No implementation found for {provider}")
    return fn


configs = []
for mode in ["fwd-bwd"]:  # ,
    for variable in ["H", "N", "V"]:
        for acc in ["fp16", "bf16", "fp16-mixed"]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=[variable],  # Argument names to use as an x-axis for the plot.
                    x_vals=[(2**i) // f for i in range_dict[variable]],  # Different possible values for `x_name`.
                    x_log=True,  # x axis is logarithmic.
                    line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
                    line_vals=method_list,
                    line_names=method_list,
                    ylabel="TFLOP/s",  # Label name for the y-axis.
                    plot_name=f"{acc} {mode}-Linear+Loss Performance over {variable}. Defaults: N=B*S=16384, H=2048, V=131072",
                    args={"mode": mode, "acc": acc},  # Values for function arguments not in `x_names` and `y_name`.
                )
            )

print("Starting benchmarks now:")


@triton.testing.perf_report(configs)
def benchmark(H=2048, V=131072, N=(4096 * 4), provider="torch", mode="fwd", acc="fp16"):  # type: ignore
    print(provider, N, H, V, mode, acc)

    if acc == "fp16":
        dtype = torch.float16
    elif acc == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    x = torch.randn(N, H, requires_grad=True, device=device, dtype=dtype)  # B S H
    y = torch.randint(0, V, (N,), device=device)  # vocab ** B S
    A = torch.randn(V, H, requires_grad=True, device=device, dtype=dtype)
    At = A.detach().clone().T.contiguous()
    At.requires_grad_()

    fn = provider_lookup(provider, x, y, At, A)
    try:
        torch.cuda.empty_cache()
        if mode == "fwd":

            @torch.no_grad
            def test_fn():
                fn()

        elif mode == "bwd":
            loss = fn()
            test_fn = lambda: loss.backward(retain_graph=True)
        elif mode == "fwd-bwd":

            def test_fn():
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16 if "bf16" in acc else torch.float16, enabled="mixed" in acc
                ):
                    loss = fn()
                loss.backward()

        else:
            test_fn = fn

        quantiles = [0.5, 0.2, 0.8]
        if ("base" in provider) and ((N * V) >= (32768 * 131072)):
            ms, min_ms, max_ms = float("NaN"), float("NaN"), float("NaN")  # give up on my 48gb card
        elif "compile" in provider and ((N * V) >= (32768 * 131072)):
            ms, min_ms, max_ms = float("NaN"), float("NaN"), float("NaN")  # give up on my 48gb card
        # elif "compile" in provider and ((N * V) >= (16384 * 131072)):
        #     ms, min_ms, max_ms = float("NaN"), float("NaN"), float("NaN")  # give up on my 48gb card
        else:

            ms, min_ms, max_ms = triton.testing.do_bench(test_fn, quantiles=quantiles, warmup=1000, rep=5000)
            ms, min_ms, max_ms = triton.testing.do_bench(test_fn, quantiles=quantiles, warmup=1000, rep=5000)
    except Exception as e:  # in any failure case
        print(f"error {e} when computing speed {provider} for N={N}, H={H}, V={V}")
        ms, min_ms, max_ms = float("NaN"), float("NaN"), float("NaN")
        pass  # raise

    flop = 2 * (N * H * V) + 3 * N * V
    if mode == "bwd":
        flop *= 2
    if mode == "fwd-bwd":
        flop *= 3

    perf = lambda ms: flop * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(print_data=True, show_plots=True, save_path="bench")
del benchmark


# # # # Bench memory


def benchmark_with_memory_reporting(func, quantiles, *args, **kwargs):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device=device)
    initial_memory = torch.cuda.memory_allocated(device=device)

    ms, min_ms, max_ms = triton.testing.do_bench(lambda: func(*args, **kwargs), quantiles=quantiles, warmup=50, rep=200)

    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated(device=device)
    memory_used = peak_memory - initial_memory

    return ms, min_ms, max_ms, memory_used


configs = []
for mode in ["fwd-bwd"]:
    for acc in ["fp16", "bf16", "fp16-mixed"]:
        for variable in ["H", "N", "V"]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=[variable],  # Argument names to use as an x-axis for the plot.
                    x_vals=[2**i for i in range_dict[variable]],  # Different possible values for `x_name`.
                    x_log=True,  # x axis is logarithmic.
                    line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
                    ylabel="Peak Memory in GB (excluding inputs)",  # Label name for the y-axis.
                    line_vals=method_list,
                    line_names=method_list,
                    args={"mode": mode, "acc": acc},  # Values for function arguments not in `x_names` and `y_name`.
                    plot_name=f"{acc} {mode}-Linear+Loss Memory Peak over {variable}. Defaults: N=B*S=16384, H=2048, V=131072",
                )
            )


@triton.testing.perf_report(configs)
def benchmark(H=2048, V=131072, N=(4096 * 4), provider="torch", mode="fwd", acc="fp16"):  # type: ignore
    print(provider, N, H, V, mode, acc)

    if acc == "fp16":
        dtype = torch.float16
    elif acc == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    x = torch.randn(N, H, requires_grad=True, device=device, dtype=dtype)  # B S H
    y = torch.randint(0, V, (N,), device=device)  # vocab ** B S
    A = torch.randn(V, H, requires_grad=True, device=device, dtype=dtype)
    At = A.detach().clone().T.contiguous()
    At.requires_grad_()

    fn = provider_lookup(provider, x, y, At, A)
    torch.cuda.empty_cache()
    try:
        if mode == "fwd":

            @torch.no_grad
            def test_fn():
                fn()

        elif mode == "bwd":
            loss = fn()
            test_fn = lambda: loss.backward(retain_graph=True)
        elif mode == "fwd-bwd":

            def test_fn():
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16 if "bf16" in acc else torch.float16, enabled="mixed" in acc
                ):
                    loss = fn()
                loss.backward()

        else:
            test_fn = fn

        quantiles = [0.5, 0.2, 0.8]
        if ("base" in provider) and ((N * V) >= (32768 * 131072)):
            max_memory_allocated = float("NaN")  # give up on my 48gb card
        elif "compile" in provider and ((N * V) >= (32768 * 131072)):
            max_memory_allocated = float("NaN")  # give up on my 48gb card
        else:
            # some things might need more mem during warmup, which is ...
            ms, min_ms, max_ms, max_memory_allocated = benchmark_with_memory_reporting(test_fn, quantiles=quantiles)
            ms, min_ms, max_ms, max_memory_allocated = benchmark_with_memory_reporting(test_fn, quantiles=quantiles)
    except Exception as e:  # in any failure case
        print(f"error {e} when computing memory {provider} for N={N}, H={H}, V={V}")
        max_memory_allocated = float("NaN")
        pass  # raise

    return max_memory_allocated / 1024**3, 0, 0


benchmark.run(print_data=True, show_plots=True, save_path="bench-mem")
del benchmark

# Bench accuracy


def get_reference_vals(x, y, A):
    x = x.detach()
    x.requires_grad_()

    A = A.detach()
    A.requires_grad_()

    x.grad, A.grad = None, None
    loss = torch_checkpoint(x.double(), y, A.double(), default_chunk_size=512)
    loss.backward()  # type: ignore

    reference_A_grad = A.grad.float().detach().clone()  # type: ignore
    reference_x_grad = x.grad.float().detach().clone()  # type: ignore
    reference_loss = loss.detach().float().clone()  # type: ignore

    del loss

    x.grad, A.grad = None, None
    return reference_loss.detach(), reference_A_grad.detach(), reference_x_grad.detach()


def simple_bench(fn, x, At, A, reference_loss, reference_x_grad, reference_A_grad):
    loss_triton = fn()
    with torch.autocast(device_type="cuda", enabled=False):
        loss_triton.backward()  # warmup
    torch.cuda.synchronize()
    x.grad, At.grad, A.grad = None, None, None
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()  # type: ignore
    loss_triton = fn()
    with torch.autocast(device_type="cuda", enabled=False):
        loss_triton.backward()
    end_event.record()  # type: ignore
    torch.cuda.synchronize()
    estimate_ms_bwd = start_event.elapsed_time(end_event)
    # print(f"fwd-bwd : {estimate_ms_bwd}ms")
    # print(f"fwd error: {torch.dist(loss_triton, reference_loss).item()}")
    with torch.autocast(device_type="cuda", enabled=False):
        loss_error = torch.dist(loss_triton, reference_loss).item()
        if At.grad is not None:
            A_error = torch.dist(reference_A_grad.T, At.grad).item()
        else:
            A_error = torch.dist(reference_A_grad, A.grad).item()  # type: ignore
        x_error = torch.dist(reference_x_grad, x.grad).item()  # type: ignore
        return loss_error, A_error, x_error


range_dict = {"H": range(8, 14, 1), "V": range(8, 18, 1), "N": range(8, 16, 1)}

configs = []
for mode in ["fwd", "bwd"]:
    for acc in ["fp16", "bf16"]:  # , "fp16-mixed"
        for variable in ["H", "N", "V"]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=[variable],  # Argument names to use as an x-axis for the plot.
                    x_vals=[2**i for i in range_dict[variable]],  # Different possible values for `x_name`.
                    x_log=True,  # x axis is logarithmic.
                    y_log=True,
                    line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
                    line_vals=method_list,
                    line_names=method_list,
                    ylabel="Average Abs. Error (torch.dist to reference)",  # Label name for the y-axis.
                    plot_name=f" {mode} {acc} Linear+Loss Accuracy over {variable}. Defaults: N=B*S=16384, H=2048, V=131072",
                    args={"mode": mode, "acc": acc},  # Values for function arguments not in `x_names` and `y_name`.
                )
            )


@triton.testing.perf_report(configs)
def benchmark(H=2048, V=131072, N=(4096 * 4), provider="torch", mode="fwd", acc="fp16"):
    print(provider, N, H, V, mode, acc)

    if acc == "fp16":
        dtype = torch.float16
    elif acc == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    y = torch.randint(0, V, (N,), device=device)  # vocab ** B S
    A = torch.randn(V, H, requires_grad=True, device=device, dtype=dtype)
    At = A.detach().clone().T.contiguous()
    At.requires_grad_()

    x = 0.1 * A[y].clone().detach() + torch.randn(N, H, device=device, dtype=dtype)  # simulate low loss
    x.requires_grad_()
    reference_loss, reference_A_grad, reference_x_grad = get_reference_vals(x, y, A)
    x.grad, A.grad, At.grad = None, None, None

    if ("base" in provider) and ((N * V) >= (32768 * 131072)):
        return float("NaN")
    elif "compile" in provider and ((N * V) >= (32768 * 131072)):
        return float("NaN")
    elif ("compile" in provider) and ("mixed" in acc) and ((N * V) >= (16384 * 131072)):
        return float("NaN")
    else:
        try:
            torch.cuda.empty_cache()
            fn = provider_lookup(provider, x, y, At, A)

            torch.cuda.empty_cache()

            with torch.autocast(
                "cuda", dtype=torch.bfloat16 if "bf16" in acc else torch.float16, enabled="mixed" in acc
            ):
                loss_error, A_error, x_error = simple_bench(
                    fn, x, At, A, reference_loss, reference_x_grad, reference_A_grad
                )
            if mode == "fwd":
                return loss_error
            else:
                return (x_error + A_error) / 2
        except Exception as e:
            print(f"error {e} when computing acc {provider} for N={N}, H={H}, V={V}")
            # raise
            return float("NaN")


benchmark.run(print_data=True, show_plots=True, save_path="bench-acc")
