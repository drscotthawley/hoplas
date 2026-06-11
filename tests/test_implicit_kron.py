#! /usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
import time
import tracemalloc

device = torch.device("cpu")  # Change to "mps" or "cuda" as needed
print('device:', device)

# ─── Layer implementations ───

class PHMLinear_Kron(nn.Module):
    """Original: batched Kronecker product, materialized weight."""
    def __init__(self, n, in_features, out_features):
        super().__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
        self.S = nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.zeros((n, out_features // n, in_features // n))))
        fan_in = in_features
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def kronecker_product1(self, a, b):
        siz1 = torch.Size(torch.tensor(a.shape[-2:]) * torch.tensor(b.shape[-2:]))
        res = a.unsqueeze(-1).unsqueeze(-3) * b.unsqueeze(-2).unsqueeze(-4)
        siz0 = res.shape[:-4]
        return res.reshape(siz0 + siz1)

    def forward(self, input):
        weight = torch.sum(self.kronecker_product1(self.A, self.S), dim=0)
        input = input.to(dtype=weight.dtype)
        return F.linear(input, weight=weight, bias=self.bias)


class PHMLinear_Implicit(nn.Module):
    """Implicit: einsum contraction, no materialized weight."""
    def __init__(self, n, in_features, out_features):
        super().__init__()
        self.n = n
        self.in_features = in_features
        self.out_features = out_features
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.zeros((n, n, n))))
        self.S = nn.Parameter(torch.nn.init.xavier_uniform_(
            torch.zeros((n, out_features // n, in_features // n))))
        fan_in = in_features
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        input = input.to(dtype=self.A.dtype)
        X = input.reshape(input.shape[0], self.n, -1)
        Y = torch.einsum('iab,ijk,Bbk->Baj', self.A, self.S, X)
        return Y.reshape(input.shape[0], -1) + self.bias


# ─── Correctness check ───

def check_correctness(n, in_f, out_f, batch):
    torch.manual_seed(42)
    kron_layer = PHMLinear_Kron(n, in_f, out_f).to(device)
    impl_layer = PHMLinear_Implicit(n, in_f, out_f).to(device)
    # Copy params so both layers are identical
    with torch.no_grad():
        impl_layer.A.copy_(kron_layer.A)
        impl_layer.S.copy_(kron_layer.S)
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
    kron_layer = PHMLinear_Kron(n, in_f, out_f).to(device)
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
    m_kron = bench_mem(PHMLinear_Kron, n, in_f, out_f, batch, "Kron (materialized)")
    m_impl = bench_mem(PHMLinear_Implicit, n, in_f, out_f, batch, "Implicit (einsum)")
    savings = (1 - m_impl / m_kron) * 100 if m_kron > 0 else 0
    print(f"  {'Memory savings':25s}: {savings:.1f}%")
