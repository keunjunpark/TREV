# TREV Profiling Guide

## Why profile before rewriting?

PyTorch calls the same NVIDIA GPU kernels (cuBLAS, cuSOLVER, cuRAND) that raw CUDA would use. Rewriting in native CUDA gives you:

- **Fused kernels** - combine multiple small ops into one launch
- **Less Python overhead** - matters only if you have many tiny ops
- **Custom memory layout** - avoids PyTorch allocator overhead

But you lose maintainability, debuggability, and portability. So **find the actual bottleneck first**.

## Quick Start

```bash
cd /home/keunjun/code/TREV
pip install -e .   # if not already installed

# Run full profiling suite (default: 8 qubits, rank 64, depth 3)
python profiling/profile_bottleneck.py

# Custom config
python profiling/profile_bottleneck.py --n 12 --rank 128 --depth 5
```

## What the profiler measures

| Section | What it tests | What to look for |
|---------|--------------|------------------|
| **1. Gate-level** | Single `build_tensor` | Baseline gate application time |
| **2. Batch build** | `build_tensor_batch` at various B | How build scales with batch size |
| **3. SVD microbenchmark** | Isolated SVD timing | Is SVD the bottleneck? Compare to total build time |
| **4. Measurement** | `efficient_contraction` vs `full_contraction` | Which measurement dominates |
| **5. Gradient** | Full `BatchParameterShiftGradient.run()` | End-to-end gradient cost |
| **6. Kernel overhead** | 1000 tiny matmuls vs 1 big one | Is Python dispatch overhead significant? |
| **7. torch.compile** | Compiled vs eager `build_tensor` | Free speedup from kernel fusion? |
| **8. CUDA Graphs** | Graph-captured `build_tensor` | Can we eliminate all launch overhead? |

## How to interpret results

### Scenario A: SVD dominates (>50% of build time)
- SVD is already running cuSOLVER on GPU
- Native CUDA won't help here
- **Try instead**: reduce bond dimension, use randomized SVD, or skip SVD for some gates

### Scenario B: Many small kernel launches (overhead section shows big gap)
- Python -> CUDA dispatch is the bottleneck
- **Try**: `torch.compile()` (fuses kernels automatically) or CUDA Graphs
- Native CUDA kernels would also help here

### Scenario C: Measurement dominates
- Efficient contraction's transfer matrix loop is the bottleneck
- **Try**: custom fused CUDA kernel for the transfer matrix multiply
- Or use `torch.compile` on the measurement function

### Scenario D: torch.compile already gives >2x speedup
- Don't bother with native CUDA. PyTorch can fuse the kernels for you.
- Focus on algorithmic improvements instead.

## Recommended optimization order (before going native CUDA)

1. **`torch.compile()`** - zero-effort kernel fusion, try it first
2. **CUDA Graphs** - captures and replays entire GPU operation sequence
3. **Algorithmic improvements** - reduce bond dimension, smarter SVD truncation
4. **Custom CUDA kernels for hot spots only** - use `torch.utils.cpp_extension` to write CUDA just for the critical inner loop, keep PyTorch for everything else
5. **Full native CUDA rewrite** - only if all above fail to meet performance targets

## Using PyTorch's built-in profiler (more detail)

For deeper analysis, use PyTorch's profiler with Chrome trace:

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
    record_shapes=True,
) as prof:
    circuit.build_tensor(theta)

# Print top CUDA kernels
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export Chrome trace (open in chrome://tracing)
prof.export_chrome_trace("profiling/trace_build.json")
```

This shows you exactly which CUDA kernels are called, how long each takes, and where the gaps (idle time) are.
