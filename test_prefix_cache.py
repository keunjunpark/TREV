"""
Test prefix caching: accuracy + speedup + memory across circuit types.
Run: python test_prefix_cache.py
"""
import time, torch
from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.efficient_contraction import expectation_value_batch as exact_ev
from TREV.optimization.gradients.batch_parameter_shift import (
    batch_gradient, batch_gradient_cached,
)
from TREV.measure.enums import MeasureMethod

device = 'cuda'
torch.manual_seed(42)


def build_ham(N):
    h = Hamiltonian(num_qubits=N)
    for i in range(N):
        j = (i+1) % N; p = ['I']*N; p[i]='Z'; p[j]='Z'
        h.add_pauli(''.join(p), 0.5)
    h.add_pauli('I'*N, 1.0)
    return h


def hea_circuit(N, chi, layers):
    """Hardware-efficient ansatz: [RY,RZ] + CNOT chain, repeated."""
    c = Circuit(num_qubit=N, rank=chi, device=device)
    for i in range(N): c.h(i)
    for _ in range(layers):
        for i in range(N): c.cx(i, (i+1)%N)
        for i in range(N): c.ry(i); c.rz(i)
    return c


def qaoa_circuit(N, chi, p_layers):
    """QAOA-style: [CNOT-RZ-CNOT per edge] + [RX mixer], repeated."""
    c = Circuit(num_qubit=N, rank=chi, device=device)
    for i in range(N): c.h(i)
    for _ in range(p_layers):
        for i in range(N):
            j = (i+1) % N
            c.cx(i, j); c.rz(j); c.cx(i, j)
        for i in range(N): c.rx(i)
    return c


def deep_circuit(N, chi, layers):
    """Deep circuit: more entangling layers."""
    c = Circuit(num_qubit=N, rank=chi, device=device)
    for i in range(N): c.h(i)
    for _ in range(layers):
        for i in range(N): c.cx(i, (i+1)%N)
        for i in range(N): c.ry(i)
        for i in range(0, N-1, 2): c.cx(i, i+1)
        for i in range(N): c.rz(i)
    return c


def time_fn(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return result, ms, peak_mb


def test_circuit(name, circuit, N, chi):
    h = build_ham(N)
    P = circuit.params_size
    theta = torch.randn(P, device=device)
    B = min(8, P)
    shift = 0.5 * 3.14159

    # Show checkpoint structure
    cps = circuit.get_prefix_checkpoints()
    seg_info = ', '.join(f'θ[{lo}:{hi}]@gate{g}' for g, (lo,hi) in cps)

    # Warmup
    batch_gradient(theta, circuit, h, B, 0, shift, 1, 0, False, MeasureMethod.EFFICIENT_CONTRACTION)

    # Original gradient
    torch.cuda.empty_cache()
    grad_orig, ms_orig, mem_orig = time_fn(
        lambda: batch_gradient(theta, circuit, h, B, 0, shift, 1, 0, False,
                              MeasureMethod.EFFICIENT_CONTRACTION))

    # Cached gradient
    torch.cuda.empty_cache()
    # warmup cached path
    batch_gradient_cached(theta, circuit, h, B, 0, shift, MeasureMethod.EFFICIENT_CONTRACTION)
    torch.cuda.empty_cache()
    grad_cached, ms_cached, mem_cached = time_fn(
        lambda: batch_gradient_cached(theta, circuit, h, B, 0, shift,
                                     MeasureMethod.EFFICIENT_CONTRACTION))

    # Accuracy
    cos = torch.nn.functional.cosine_similarity(
        grad_orig.unsqueeze(0), grad_cached.unsqueeze(0)).item()
    max_diff = (grad_orig - grad_cached).abs().max().item()

    speedup = ms_orig / ms_cached if ms_cached > 0 else 0
    ok = cos > 0.999 and max_diff < 1e-3

    print(f"  {name}")
    print(f"    N={N} chi={chi} P={P} B={B}")
    print(f"    segments: {seg_info}")
    print(f"    original:  {ms_orig:>7.1f}ms  peak={mem_orig:>6.1f}MB")
    print(f"    cached:    {ms_cached:>7.1f}ms  peak={mem_cached:>6.1f}MB")
    print(f"    speedup:   {speedup:.2f}x")
    print(f"    cos_sim:   {cos:.6f}  max_diff: {max_diff:.2e}  {'PASS' if ok else 'FAIL'}")
    print()
    return ok


all_pass = True
print("=" * 60)
print("Prefix Caching: accuracy + speedup + memory")
print("=" * 60)

# HEA circuits
all_pass &= test_circuit("HEA 2-layer", hea_circuit(8, 4, 2), 8, 4)
all_pass &= test_circuit("HEA 2-layer", hea_circuit(8, 10, 2), 8, 10)
all_pass &= test_circuit("HEA 4-layer", hea_circuit(8, 4, 4), 8, 4)

# QAOA circuits
all_pass &= test_circuit("QAOA p=2", qaoa_circuit(8, 4, 2), 8, 4)
all_pass &= test_circuit("QAOA p=3", qaoa_circuit(6, 10, 3), 6, 10)

# Deep circuit
all_pass &= test_circuit("Deep 3-layer", deep_circuit(8, 4, 3), 8, 4)

print("=" * 60)
print(f"ALL {'PASSED' if all_pass else 'FAILED'}")
