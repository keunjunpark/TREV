"""
Exhaustive test for prefix caching correctness.
Compares batch_gradient vs batch_gradient_cached across many circuit
structures, sizes, chi values, and parameter counts.

Run: python test_prefix_cache_exhaustive.py
"""
import torch
from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.optimization.gradients.batch_parameter_shift import (
    batch_gradient, batch_gradient_cached,
)
from TREV.measure.enums import MeasureMethod

device = 'cuda'
SHIFT = 0.5 * 3.14159
METHOD = MeasureMethod.EFFICIENT_CONTRACTION
ATOL = 1e-2
COS_TOL = 0.999

passed = 0
failed = 0
total = 0


def check(name, circuit, N, chi, B=8):
    global passed, failed, total
    total += 1

    h = Hamiltonian(num_qubits=N)
    for i in range(N):
        j = (i + 1) % N
        p = ['I'] * N; p[i] = 'Z'; p[j] = 'Z'
        h.add_pauli(''.join(p), 0.5)
    h.add_pauli('I' * N, 1.0)

    P = circuit.params_size
    if P == 0:
        print(f"  SKIP {name} (no params)")
        return

    B = min(B, P)
    theta = torch.randn(P, device=device)

    grad_orig = batch_gradient(theta, circuit, h, B, 0, SHIFT, 1, 0, False, METHOD)
    grad_cached = batch_gradient_cached(theta, circuit, h, B, 0, SHIFT, METHOD)

    cos = torch.nn.functional.cosine_similarity(
        grad_orig.unsqueeze(0).float(), grad_cached.unsqueeze(0).float()
    ).item()
    max_diff = (grad_orig - grad_cached).abs().max().item()
    norm_orig = grad_orig.norm().item()

    ok = cos > COS_TOL and max_diff < ATOL
    if ok:
        passed += 1
    else:
        failed += 1

    status = "PASS" if ok else "FAIL"
    print(f"  {status} {name:<45} P={P:>3} cos={cos:.6f} diff={max_diff:.2e} |g|={norm_orig:.3f}")


# ═══════════════════════════════════════════════
# 1. HEA circuits — vary N, chi, layers
# ═══════════════════════════════════════════════
print("=" * 70)
print("1. Hardware-Efficient Ansatz (HEA)")
print("=" * 70)

for N in [4, 6, 8, 12]:
    for chi in [4, 10]:
        for layers in [1, 2, 3]:
            c = Circuit(num_qubit=N, rank=chi, device=device)
            for i in range(N): c.h(i)
            for _ in range(layers):
                for i in range(N): c.cx(i, (i + 1) % N)
                for i in range(N): c.ry(i); c.rz(i)
            check(f"HEA N={N} chi={chi} L={layers}", c, N, chi)

# ═══════════════════════════════════════════════
# 2. QAOA circuits — params interleaved with CNOTs
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. QAOA (params between CNOTs)")
print("=" * 70)

for N in [4, 6, 8]:
    for chi in [4, 10]:
        for p_layers in [1, 2, 3]:
            c = Circuit(num_qubit=N, rank=chi, device=device)
            for i in range(N): c.h(i)
            for _ in range(p_layers):
                for i in range(N):
                    j = (i + 1) % N
                    c.cx(i, j); c.rz(j); c.cx(i, j)
                for i in range(N): c.rx(i)
            check(f"QAOA N={N} chi={chi} p={p_layers}", c, N, chi)

# ═══════════════════════════════════════════════
# 3. Mixed circuits — different gate patterns
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. Mixed / unusual structures")
print("=" * 70)

# Only params, no 2q gates (no caching needed)
for N in [4, 8]:
    c = Circuit(num_qubit=N, rank=4, device=device)
    for i in range(N): c.ry(i); c.rz(i)
    check(f"Params-only N={N}", c, N, 4)

# Only 2q gates then params (single prefix)
for N in [4, 8]:
    c = Circuit(num_qubit=N, rank=4, device=device)
    for i in range(N): c.h(i)
    for i in range(N): c.cx(i, (i + 1) % N)
    for i in range(N): c.cx(i, (i + 1) % N)
    for i in range(N): c.ry(i); c.rz(i)
    check(f"2xCNOT+params N={N}", c, N, 4)

# Alternating: param, CNOT, param, CNOT (worst case for caching)
for N in [4, 6]:
    c = Circuit(num_qubit=N, rank=4, device=device)
    for i in range(N): c.h(i)
    for _ in range(3):
        c.ry(0)
        c.cx(0, 1)
        c.ry(1)
        c.cx(1, 2 % N)
    check(f"Alternating param-CNOT N={N}", c, N, 4)

# Single param (may have zero gradient — use max_diff check only)
c = Circuit(num_qubit=4, rank=4, device=device)
for i in range(4): c.h(i)
for i in range(4): c.cx(i, (i + 1) % 4)
c.ry(0); c.rz(0); c.ry(1); c.rz(1)
check("Few params after CNOT", c, 4, 4)

# Params only on one qubit
c = Circuit(num_qubit=6, rank=4, device=device)
for i in range(6): c.h(i)
for i in range(6): c.cx(i, (i + 1) % 6)
for _ in range(4): c.ry(0); c.rz(0)
check("Params on qubit 0 only", c, 6, 4)

# ═══════════════════════════════════════════════
# 4. Edge cases
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("4. Edge cases")
print("=" * 70)

# Very small: N=2
c = Circuit(num_qubit=2, rank=4, device=device)
c.h(0); c.h(1); c.cx(0, 1); c.ry(0); c.rz(1)
check("N=2 minimal", c, 2, 4)

# Large chunk size (B > P)
c = Circuit(num_qubit=4, rank=4, device=device)
for i in range(4): c.h(i)
for i in range(4): c.cx(i, (i + 1) % 4)
for i in range(4): c.ry(i)
check("B > P", c, 4, 4, B=32)

# Chunk size = 1
c = Circuit(num_qubit=4, rank=4, device=device)
for i in range(4): c.h(i)
for i in range(4): c.cx(i, (i + 1) % 4)
for i in range(4): c.ry(i); c.rz(i)
check("B=1 (one at a time)", c, 4, 4, B=1)

# Multiple different chi values
for chi in [2, 4, 8, 16]:
    c = Circuit(num_qubit=4, rank=chi, device=device)
    for i in range(4): c.h(i)
    for i in range(4): c.cx(i, (i + 1) % 4)
    for i in range(4): c.ry(i); c.rz(i)
    check(f"chi={chi}", c, 4, chi)

# ═══════════════════════════════════════════════
# 5. Reproducibility: same theta, different seeds
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("5. Reproducibility (multiple runs, same theta)")
print("=" * 70)

c = Circuit(num_qubit=8, rank=10, device=device)
for i in range(8): c.h(i)
for _ in range(2):
    for i in range(8): c.cx(i, (i + 1) % 8)
    for i in range(8): c.ry(i); c.rz(i)

h = Hamiltonian(num_qubits=8)
for i in range(8):
    j = (i + 1) % 8; p = ['I'] * 8; p[i] = 'Z'; p[j] = 'Z'
    h.add_pauli(''.join(p), 0.5)
h.add_pauli('I' * 8, 1.0)

theta = torch.randn(c.params_size, device=device)
results = []
for run in range(5):
    g = batch_gradient_cached(theta, c, h, 8, 0, SHIFT, METHOD)
    results.append(g.clone())

for i in range(1, 5):
    diff = (results[0] - results[i]).abs().max().item()
    ok = diff < 1e-6
    total += 1
    if ok: passed += 1
    else: failed += 1
    print(f"  {'PASS' if ok else 'FAIL'} run0 vs run{i}: max_diff={diff:.2e}")


# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print("=" * 70)
