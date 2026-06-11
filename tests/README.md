## Comparing Implicit Kronecker vs. Original (Across Device Types)

### TLDR: 
Save memory with implicit method. On CPU it's faster but on CUDA and MPS it can be slower -- much slower on MPS


### Speed: Implicit speedup over PHMLinear (>1.0x = implicit wins)

| Config (n, in, out, batch) | CPU    | CUDA   | MPS    |
|----------------------------|--------|--------|--------|
| n=4, 128×128,  b=64        | 2.41x  | 0.86x  | 0.78x  |
| n=4, 256×256,  b=64        | 1.57x  | 0.87x  | 0.62x  |
| n=4, 512×512,  b=64        | 1.51x  | 0.87x  | 0.69x  |
| n=4, 1024×1024, b=64       | 1.57x  | 0.87x  | 0.11x  |
| n=4, 2048×2048, b=32       | 1.54x  | 0.86x  | 0.03x  |
| n=8, 512×512,  b=64        | 1.89x  | 0.88x  | 0.34x  |

### Memory: Implicit savings over PHMLinear

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

