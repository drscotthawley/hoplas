#! /usr/bin/env python3

"""
Results: 

## Speed: Implicit speedup over PHMLinear (>1.0x = implicit wins)

| Config (n, in, out, batch) | CPU    | CUDA   | MPS    |
|----------------------------|--------|--------|--------|
| n=4, 128×128,  b=64        | 2.41x  | 0.86x  | 0.78x  |
| n=4, 256×256,  b=64        | 1.57x  | 0.87x  | 0.62x  |
| n=4, 512×512,  b=64        | 1.51x  | 0.87x  | 0.69x  |
| n=4, 1024×1024, b=64       | 1.57x  | 0.87x  | 0.11x  |
| n=4, 2048×2048, b=32       | 1.54x  | 0.86x  | 0.03x  |
| n=8, 512×512,  b=64        | 1.89x  | 0.88x  | 0.34x  |

## Memory: Implicit savings over PHMLinear

| Config (n, in, out, batch) | CPU†   | CUDA   | MPS†   |
|----------------------------|--------|--------|--------|
| n=4, 128×128,  b=64        | ~41%   |  50%   | ~41%   |
| n=4, 256×256,  b=64        | ~41%   |  55%   | ~41%   |
| n=4, 512×512,  b=64        | ~44%   |  58%   | ~44%   |
| n=4, 1024×1024, b=64       | ~46%   |  59%   | ~46%   |
| n=4, 2048×2048, b=32       | ~46%   |  60%   | ~46%   |
| n=8, 512×512,  b=64        | ~44%   |  76%   | ~44%   |

† CPU and MPS memory measured via tracemalloc (Python allocations only, not tensor VRAM — treat as approximate).
CUDA measured via `torch.cuda.max_memory_allocated` (accurate).
"""



import argparse
import time
import tracemalloc

import torch

from hoplas.ph_layers import PHMLinear, PHMLinear_Implicit


fastest_device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default=fastest_device, help="Device to use (cpu, cuda, mps)")
args = parser.parse_args()
device = torch.device(args.device)
print('device:', device)

# ─── Correctness check ───

def check_correctness(n, in_f, out_f, batch):
    torch.manual_seed(42)
    kron_layer = PHMLinear(n, in_f, out_f).to(device)
    impl_layer = PHMLinear_Implicit(n, in_f, out_f).to(device)
    # Copy params so both layers are identical
    with torch.no_grad():
        impl_layer.a.copy_(kron_layer.a)
        impl_layer.s.copy_(kron_layer.s)
        impl_layer.bias.copy_(kron_layer.bias)
    x = torch.randn(batch, in_f, device=device)
    y_kron = kron_layer(x)
    y_impl = impl_layer(x)
    maxdiff = (y_kron - y_impl).abs().max().item()
    print(f"  Correctness (n={n}, in={in_f}, out={out_f}, batch={batch}): max diff = {maxdiff:.2e}")
    assert maxdiff < 1e-5, f"Results diverge! maxdiff={maxdiff}"


# ─── Timing benchmark ───

def benchmark_speed(layer, x, label, warmup=20, iters=200):
    # Warmup
    for _ in range(warmup):
        _ = layer(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = layer(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    print(f"  {label:25s}: {elapsed*1e3:8.3f} ms/fwd")
    return elapsed


# ─── Memory benchmark ───

def benchmark_memory_cpu(layer_cls, n, in_f, out_f, batch, label):
    """Measure peak memory of a forward pass on CPU using tracemalloc."""
    torch.manual_seed(42)
    layer = layer_cls(n, in_f, out_f).to(device)
    x = torch.randn(batch, in_f, device=device)
    # Warmup
    _ = layer(x)

    tracemalloc.start()
    _ = layer(x)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  {label:25s}: peak alloc = {peak / 1024:.1f} KB")
    return peak


def benchmark_memory_cuda(layer_cls, n, in_f, out_f, batch, label):
    """Measure peak memory of a forward pass on CUDA."""
    torch.manual_seed(42)
    layer = layer_cls(n, in_f, out_f).to(device)
    x = torch.randn(batch, in_f, device=device)
    # Warmup
    _ = layer(x)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()
    _ = layer(x)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - mem_before
    print(f"  {label:25s}: peak VRAM = {peak / 1024:.1f} KB")
    return peak


# ─── Run benchmarks ───

configs = [
    # (n, in_features, out_features, batch_size)
    (4,  128,  128,  64),
    (4,  256,  256,  64),
    (4,  512,  512,  64),
    (4, 1024, 1024,  64),
    (4, 2048, 2048,  32),
    (8,  512,  512,  64),
]

print("=" * 60)
print("CORRECTNESS CHECKS")
print("=" * 60)
for n, in_f, out_f, batch in configs:
    check_correctness(n, in_f, out_f, batch)

print()
print("=" * 60)
print("SPEED BENCHMARKS")
print("=" * 60)
for n, in_f, out_f, batch in configs:
    print(f"\nn={n}, in={in_f}, out={out_f}, batch={batch}")
    torch.manual_seed(42)
    kron_layer = PHMLinear(n, in_f, out_f).to(device)
    impl_layer = PHMLinear_Implicit(n, in_f, out_f).to(device)
    x = torch.randn(batch, in_f, device=device)
    t_kron = benchmark_speed(kron_layer, x, "Kron (materialized)")
    t_impl = benchmark_speed(impl_layer, x, "Implicit (einsum)")
    speedup = t_kron / t_impl
    print(f"  {'Speedup':25s}: {speedup:.2f}x")

print()
print("=" * 60)
print("MEMORY BENCHMARKS")
print("=" * 60)
bench_mem = benchmark_memory_cuda if device.type == "cuda" else benchmark_memory_cpu
for n, in_f, out_f, batch in configs:
    print(f"\nn={n}, in={in_f}, out={out_f}, batch={batch}")
    m_kron = bench_mem(PHMLinear, n, in_f, out_f, batch, "Kron (materialized)")
    m_impl = bench_mem(PHMLinear_Implicit, n, in_f, out_f, batch, "Implicit (einsum)")
    savings = (1 - m_impl / m_kron) * 100 if m_kron > 0 else 0
    print(f"  {'Memory savings':25s}: {savings:.1f}%")
