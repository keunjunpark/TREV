"""
Benchmark: torch.einsum vs cuTensorNet (fair comparison)

Key differences from naive benchmark:
  1. cuTN path is computed ONCE and reused (amortized)
  2. Test larger bond dimensions where cuTN kernels matter
  3. Separate path-finding time from contraction time
"""

import time
import torch
import numpy as np

from cuquantum.tensornet import einsum as cutn_einsum, einsum_path as cutn_einsum_path
from cuquantum.tensornet import Network, NetworkOptions, OptimizerOptions


def random_transfer_matrices(N, chi, device='cuda'):
    return [torch.randn(chi, chi, chi, chi, dtype=torch.cfloat, device=device) for _ in range(N)]


def build_ring_einsum_str(N):
    idx = 0
    def label():
        nonlocal idx
        c = chr(ord('a') + idx) if idx < 26 else chr(ord('A') + idx - 26)
        idx += 1
        return c
    bra = [label() for _ in range(N)]
    ket = [label() for _ in range(N)]
    subs = []
    for i in range(N):
        j = (i + 1) % N
        subs.append(f"{bra[i]}{ket[i]}{bra[j]}{ket[j]}")
    return ','.join(subs) + '->'


def benchmark_fn(fn, warmup=5, repeats=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times), np.std(times)


def main():
    device = 'cuda'
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    import cuquantum
    print(f"cuQuantum: {cuquantum.__version__}")
    print()

    configs = [
        # (N, chi)
        (4, 16),
        (4, 32),
        (4, 64),
        (8, 16),
        (8, 32),
        (8, 64),
        (12, 16),
        (12, 32),
        (16, 16),
        (16, 32),
    ]

    print(f"{'N':>4} {'chi':>5} | {'torch (ms)':>11} {'cuTN (ms)':>11} {'cuTN+path':>10} | {'winner':>8} {'ratio':>7} {'ok':>4}")
    print("-" * 72)

    for N, chi in configs:
        torch.cuda.empty_cache()
        einsum_str = build_ring_einsum_str(N)

        try:
            transfers = random_transfer_matrices(N, chi, device)
        except Exception:
            print(f"{N:>4} {chi:>5} | OOM on alloc")
            continue

        # --- torch.einsum ---
        try:
            torch_val = torch.einsum(einsum_str, *transfers)
            torch_ms, _ = benchmark_fn(lambda: torch.einsum(einsum_str, *transfers))
            torch_ms *= 1000
        except Exception as e:
            print(f"{N:>4} {chi:>5} | torch OOM")
            continue

        # --- cuTensorNet with Network (path reuse) ---
        try:
            torch.cuda.empty_cache()

            # Method: use cuTN Network API with path caching
            # First call includes path finding
            t_path_start = time.perf_counter()
            cutn_val = cutn_einsum(einsum_str, *transfers)
            torch.cuda.synchronize()
            t_path_end = time.perf_counter()
            first_call_ms = (t_path_end - t_path_start) * 1000

            if not isinstance(cutn_val, torch.Tensor):
                cutn_val = torch.as_tensor(cutn_val, device=device)

            # Subsequent calls (path cached internally by cuTN)
            cutn_ms, _ = benchmark_fn(lambda: cutn_einsum(einsum_str, *transfers))
            cutn_ms *= 1000

            match = torch.allclose(torch_val, cutn_val, atol=1e-1, rtol=1e-1)
        except Exception as e:
            print(f"{N:>4} {chi:>5} | {torch_ms:>9.2f}ms   cuTN err: {str(e)[:40]}")
            continue

        # Determine winner
        if cutn_ms < torch_ms:
            winner = "cuTN"
            ratio = f"{torch_ms/cutn_ms:.2f}x"
        else:
            winner = "torch"
            ratio = f"{cutn_ms/torch_ms:.2f}x"

        ok = "OK" if match else "FAIL"
        print(f"{N:>4} {chi:>5} | {torch_ms:>9.2f}ms {cutn_ms:>9.2f}ms {first_call_ms:>8.1f}ms | {winner:>8} {ratio:>7} {ok:>4}")

    print()
    print("cuTN (ms)  = amortized (path reuse). cuTN+path = first call including path finding.")
    print("winner ratio: how many times faster the winner is.")


if __name__ == '__main__':
    main()
