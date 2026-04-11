"""
Profile: AutogradGradient — vectorized contraction with auto term chunking.

Tests correctness, speed vs sequential, speed vs parameter-shift,
and the effect of term_chunk size on performance.
"""

import time
import torch
import numpy as np

from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.optimization.gradients.autograd_gradient import (
    _build_tensor_diff,
    _contraction_diff,
    _contraction_diff_vectorized,
    _auto_term_chunk,
    autograd_gradient,
)
from TREV.optimization.gradients.batch_parameter_shift import (
    BatchParameterShiftGradient,
)
from TREV.measure.enums import MeasureMethod


# ── Helpers ──

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


def build_circuit(n, chi, layers=2, device='cuda'):
    circuit = Circuit(num_qubit=n, rank=chi, device=device)
    for i in range(n):
        circuit.h(i)
    for _ in range(layers):
        for i in range(n):
            circuit.cx(i, (i + 1) % n)
        for i in range(n):
            circuit.ry(i)
            circuit.rz(i)
    theta = torch.randn(circuit.params_size, device=device)
    return circuit, theta


def time_fn(fn, warmup=3, repeats=10):
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


def autograd_seq(theta, circuit, ham, dtype=torch.cfloat):
    """Original sequential autograd (for comparison)."""
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    with torch.enable_grad():
        t = theta.detach().to(real_dtype).clone().requires_grad_(True)
        tensor = _build_tensor_diff(t, circuit, dtype)
        loss = _contraction_diff(tensor, ham, dtype)
        loss.backward()
    return t.grad.float()


# ── Main ──

def main():
    device = 'cuda'
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    free_b, total_b = torch.cuda.mem_get_info(device)
    print(f"GPU memory: {free_b/1e9:.1f} GB free / {total_b/1e9:.1f} GB total")
    print(f"PyTorch: {torch.__version__}")
    print()

    configs = [
        (8, 4),
        (8, 10),
        (12, 4),
        (12, 10),
    ]

    # =========================================================
    # Part 1: Auto term_chunk selection
    # =========================================================
    print("=" * 70)
    print("Part 1: Auto term_chunk selection")
    print("=" * 70)

    print(f"{'N':>4} {'chi':>5} {'T':>5} | {'auto_chunk':>11} {'mem/term':>12}")
    print("-" * 50)

    for N, chi in configs:
        ham = build_maxcut_hamiltonian(N)
        T = len(ham.paulis)
        chunk = _auto_term_chunk(N, chi, torch.cfloat, device)
        elem_size = 8
        mem_per_term = 6 * N * (chi ** 4) * elem_size
        print(f"{N:>4} {chi:>5} {T:>5} | {min(chunk, T):>11} {mem_per_term/1e6:>10.1f}MB")

    # =========================================================
    # Part 2: Correctness — sequential vs vectorized (chunked)
    # =========================================================
    print()
    print("=" * 70)
    print("Part 2: Correctness — sequential vs vectorized gradient")
    print("=" * 70)

    ps_ref = BatchParameterShiftGradient(
        shift=np.pi / 2, batch_size=None, shots=0,
        measure_method=MeasureMethod.EFFICIENT_CONTRACTION, depth=1,
    )

    print(f"{'N':>4} {'chi':>5} | {'vs param-shift':>15} {'seq vs vec':>12}")
    print("-" * 45)

    for N, chi in configs:
        torch.cuda.empty_cache()
        ham = build_maxcut_hamiltonian(N)
        circuit, theta = build_circuit(N, chi, device=device)

        g_ps = ps_ref.run(theta, circuit, ham)
        g_seq = autograd_seq(theta, circuit, ham, torch.cfloat)
        g_vec = autograd_gradient(theta, circuit, ham, torch.cfloat)

        d_ps = (g_vec - g_ps).abs().max().item()
        d_sv = (g_seq - g_vec).abs().max().item()
        print(f"{N:>4} {chi:>5} | {d_ps:>15.2e} {d_sv:>12.2e}")

    # =========================================================
    # Part 3: Speed — sequential vs vectorized vs param-shift
    # =========================================================
    print()
    print("=" * 70)
    print("Part 3: Speed comparison (cfloat)")
    print("=" * 70)

    print(f"{'N':>4} {'chi':>5} {'T':>4} {'P':>5} | {'sequential':>11} {'vectorized':>11} {'param-shift':>12} | {'vec/seq':>8} {'vec/ps':>7}")
    print("-" * 85)

    for N, chi in configs:
        torch.cuda.empty_cache()
        ham = build_maxcut_hamiltonian(N)
        circuit, theta = build_circuit(N, chi, device=device)
        P = circuit.params_size
        T = len(ham.paulis)

        try:
            # Sequential
            autograd_seq(theta, circuit, ham)
            seq_med, _ = time_fn(lambda: autograd_seq(theta, circuit, ham))

            # Vectorized (auto chunk)
            autograd_gradient(theta, circuit, ham, torch.cfloat)
            vec_med, _ = time_fn(lambda: autograd_gradient(theta, circuit, ham, torch.cfloat))

            # Parameter-shift
            ps_ref.run(theta, circuit, ham)
            ps_med, _ = time_fn(lambda: ps_ref.run(theta, circuit, ham))

            sv = seq_med / vec_med
            vp = ps_med / vec_med
            print(
                f"{N:>4} {chi:>5} {T:>4} {P:>5} | "
                f"{seq_med:>9.1f}ms {vec_med:>9.1f}ms {ps_med:>10.1f}ms | "
                f"{sv:>7.2f}x {vp:>6.2f}x"
            )
        except Exception as e:
            print(f"{N:>4} {chi:>5} {T:>4} {P:>5} |  error: {str(e)[:60]}")

    # =========================================================
    # Part 4: Effect of term_chunk size
    # =========================================================
    print()
    print("=" * 70)
    print("Part 4: term_chunk sweep (N=12, chi=10)")
    print("=" * 70)

    N, chi = 12, 10
    torch.cuda.empty_cache()
    ham = build_maxcut_hamiltonian(N)
    circuit, theta = build_circuit(N, chi, device=device)
    T = len(ham.paulis)

    chunks_to_test = [1, 4, 8, 12, T]
    print(f"  T={T} terms, P={circuit.params_size} params")
    print(f"  {'chunk':>6} | {'time(ms)':>10} {'peak_mem(MB)':>13}")
    print(f"  {'-'*6}-+-{'-'*10}-{'-'*13}")

    for tc in chunks_to_test:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        # Warmup
        autograd_gradient(theta, circuit, ham, torch.cfloat, term_chunk=tc)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats(device)
        med, _ = time_fn(
            lambda tc=tc: autograd_gradient(theta, circuit, ham, torch.cfloat, term_chunk=tc),
            warmup=2, repeats=5,
        )
        peak = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"  {tc:>6} | {med:>8.1f}ms {peak:>11.1f}MB")

    print()


if __name__ == '__main__':
    main()
