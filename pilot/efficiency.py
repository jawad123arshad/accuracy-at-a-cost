"""
Efficiency metrics used for C2 (pilot benchmarking protocol) and C3
(descriptive efficiency-performance comparison): parameter count, inference
latency, peak GPU memory, and training wall-clock time.

FLOPs are estimated via `thop` if it's installed (pip install thop); if not
available, that column is left as NaN rather than failing the run, since
FLOPs are secondary to the timing/memory numbers for this pilot.
"""

import time
import torch


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model, sample_batch):
    try:
        from thop import profile
    except ImportError:
        return float("nan")
    try:
        model.eval()
        with torch.no_grad():
            macs, _ = profile(
                model,
                inputs=(sample_batch["input_ids"], sample_batch["lengths"]),
                verbose=False,
            )
        return float(macs) * 2  # MACs -> FLOPs (rough, ~2 FLOPs per MAC)
    except Exception:
        return float("nan")


def measure_inference_latency(model, sample_batch, device, n_warmup=5, n_iters=30):
    model.eval()
    input_ids = sample_batch["input_ids"].to(device)
    lengths = sample_batch["lengths"].to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_ids, lengths)
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            model(input_ids, lengths)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return (elapsed / n_iters) * 1000.0  # ms per batch


def reset_peak_memory(device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(device):
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return float("nan")
