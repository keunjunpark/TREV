"""
Profile: torch.compile() vs eager mode on TREV's efficient contraction.

Tests the main expectation value hotpath with and without compilation.
Also tests CUDA graphs for the fixed-shape workload.
"""

import time
import torch
import numpy as np

from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.efficient_contraction import (
    expectation_value_batch,
    _kron_contract_right_4d,
)


def build_maxcut_hamiltonian(n):
    h = Hamiltonian(num_qubits=n)
    for i in range(n):
        j = (i + 1) % n
        h.add_pauli('I' * n, 0.5)
        pauli = ['I'] * n
        pauli[i] = 'Z'
        pauli[j] = 'Z'
        h.add_pauli(''.join(pauli), -0.5)
    return h


def build_tensor(n, chi, device='cuda'):
    circuit = Circuit(num_qubit=n, rank=chi, device=device)
    for i in range(n):
        circuit.h(i)
    for _ in range(2):
        for i in range(n):
            circuit.cx(i, (i + 1) % n)
        for i in range(n):
            circuit.ry(i)
            circuit.rz(i)
    theta = torch.randn(circuit.params_size, device=device)
    return circuit.build_tensor(theta)


def time_fn(fn, warmup=5, repeats=30):
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
    med = np.median(times) * 1000
    std = np.std(times) * 1000
    return med, std


def main():
    device = 'cuda'
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    configs = [
        (8, 10),
        (8, 32),
        (12, 10),
        (12, 32),
        (16, 10),
        (16, 32),
    ]

    # =========================================================
    # Part 1: Full expectation_value_batch — eager vs compiled
    # =========================================================
    print("=" * 70)
    print("Part 1: expectation_value_batch  (eager vs torch.compile)")
    print("=" * 70)

    compiled_ev = torch.compile(expectation_value_batch)

    print(f"{'N':>4} {'chi':>5} | {'eager (ms)':>11} {'compile (ms)':>13} | {'speedup':>8}")
    print("-" * 55)

    for N, chi in configs:
        torch.cuda.empty_cache()
        ham = build_maxcut_hamiltonian(N)
        tensor = build_tensor(N, chi, device)

        # Eager
        eager_val = expectation_value_batch(tensor, ham, device=device)
        eager_ms, _ = time_fn(lambda: expectation_value_batch(tensor, ham, device=device))

        # Compiled (first call triggers compilation)
        try:
            comp_val = compiled_ev(tensor, ham, device=device)
            comp_ms, _ = time_fn(lambda: compiled_ev(tensor, ham, device=device))
            match = torch.allclose(eager_val, comp_val, atol=1e-4)
            speedup = eager_ms / comp_ms
            tag = "OK" if match else "MISMATCH"
            print(f"{N:>4} {chi:>5} | {eager_ms:>9.2f}ms {comp_ms:>11.2f}ms | {speedup:>7.2f}x  {tag}")
        except Exception as e:
            print(f"{N:>4} {chi:>5} | {eager_ms:>9.2f}ms   compile err: {str(e)[:40]}")

    # =========================================================
    # Part 2: Inner kernel — _kron_contract_right_4d
    # =========================================================
    print()
    print("=" * 70)
    print("Part 2: _kron_contract_right_4d inner kernel (eager vs compile)")
    print("=" * 70)

    compiled_kron = torch.compile(_kron_contract_right_4d)

    chi_values = [10, 32, 64]
    print(f"{'chi':>5} | {'eager (us)':>11} {'compile (us)':>13} | {'speedup':>8}")
    print("-" * 50)

    for chi in chi_values:
        torch.cuda.empty_cache()
        Prod = torch.randn(chi, chi, chi, chi, dtype=torch.cfloat, device=device)
        A0 = torch.randn(chi, chi, dtype=torch.cfloat, device=device)
        A1 = torch.randn(chi, chi, dtype=torch.cfloat, device=device)

        # Eager
        _ = _kron_contract_right_4d(Prod, A0, A1, op=3)
        eager_us, _ = time_fn(lambda: _kron_contract_right_4d(Prod, A0, A1, op=3), warmup=10, repeats=50)
        eager_us *= 1000  # ms -> us

        # Compiled
        try:
            _ = compiled_kron(Prod, A0, A1, op=3)
            comp_us, _ = time_fn(lambda: compiled_kron(Prod, A0, A1, op=3), warmup=10, repeats=50)
            comp_us *= 1000
            speedup = eager_us / comp_us
            print(f"{chi:>5} | {eager_us:>9.1f}us {comp_us:>11.1f}us | {speedup:>7.2f}x")
        except Exception as e:
            print(f"{chi:>5} | {eager_us:>9.1f}us   compile err: {str(e)[:40]}")

    # =========================================================
    # Part 3: CUDA Graphs
    # =========================================================
    print()
    print("=" * 70)
    print("Part 3: CUDA Graphs (capture full contraction, replay)")
    print("=" * 70)

    print(f"{'N':>4} {'chi':>5} | {'eager (ms)':>11} {'graph (ms)':>11} | {'speedup':>8}")
    print("-" * 52)

    for N, chi in configs:
        torch.cuda.empty_cache()
        ham = build_maxcut_hamiltonian(N)
        tensor = build_tensor(N, chi, device)

        # Eager baseline
        eager_ms, _ = time_fn(lambda: expectation_value_batch(tensor, ham, device=device))

        # CUDA Graph capture
        try:
            # Warmup for graph capture
            torch.cuda.synchronize()
            for _ in range(3):
                expectation_value_batch(tensor, ham, device=device)
            torch.cuda.synchronize()

            # Capture
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                graph_result = expectation_value_batch(tensor, ham, device=device)

            # Replay benchmark
            def replay():
                g.replay()
                return graph_result

            graph_ms, _ = time_fn(replay)
            speedup = eager_ms / graph_ms
            print(f"{N:>4} {chi:>5} | {eager_ms:>9.2f}ms {graph_ms:>9.2f}ms | {speedup:>7.2f}x")
        except Exception as e:
            print(f"{N:>4} {chi:>5} | {eager_ms:>9.2f}ms   graph err: {str(e)[:50]}")

    print()
    print("speedup > 1 means the optimization is faster than eager PyTorch.")


if __name__ == '__main__':
    main()
