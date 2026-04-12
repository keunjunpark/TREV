#!/usr/bin/env python3
"""Quick correctness + speed test for EC optimizations."""
import torch
import time
import math
from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.enums import MeasureMethod
from TREV.optimization.gradients.batch_parameter_shift import (
    expectation_value_batch_efficient_contraction,
    expectation_value_batch_right_suffix,
    BatchParameterShiftGradient,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name()}')

# ── Correctness tests ──
print('\n=== Correctness ===')

tests = [
    ('Z-only (deep)', 8, 2, 30, [('Z'+'I'*7, 1.0), ('IZ'+'I'*6, 0.5)]),
    ('ZZ terms',      6, 4, 5,  [('ZZ'+'I'*4, -1.0), ('IZZ'+'I'*3, -0.5), ('Z'+'I'*5, 0.3)]),
    ('X/Y terms',     4, 4, 3,  [('XXII', 0.5), ('YYII', 0.5), ('ZIIZ', 1.0), ('IXYI', 0.3)]),
    ('XXYY multi',    6, 4, 5,  [('XXYYII', 0.3), ('ZZII'+'I'*2, 1.0), ('I'*2+'XI'*2, -0.5)]),
]

all_ok = True
for name, N, rank, depth, terms in tests:
    c = Circuit(num_qubit=N, rank=rank, device=DEVICE)
    for d in range(depth):
        for q in range(N): c.ry(q)
        for q in range(N-1): c.cx(q, q+1)
    theta = torch.randn(c.params_size, device=DEVICE)
    h = Hamiltonian(num_qubits=N)
    for pauli, coeff in terms:
        h.add_pauli(pauli, coeff)

    ec = c.get_expectation_value(theta, h, MeasureMethod.EFFICIENT_CONTRACTION)
    rss = c.get_expectation_value(theta, h, MeasureMethod.RIGHT_SUFFIX_SAMPLING, shots=200000)
    diff = abs(float(ec) - float(rss))
    ok = diff < 0.05
    all_ok = all_ok and ok
    print(f'  {name:20s}  EC={float(ec):+.6f}  RSS={float(rss):+.6f}  diff={diff:.6f}  {"OK" if ok else "FAIL"}')

print(f'\nCorrectness: {"ALL PASS" if all_ok else "SOME FAILED"}')

# ── Speed benchmark ──
print('\n=== Speed (TSP-like, N=9, rank=8) ===')

N, rank, depth = 9, 8, 3
c = Circuit(num_qubit=N, rank=rank, device=DEVICE)
for d in range(depth):
    for q in range(N): c.ry(q); c.rz(q)
    for q in range(0, N-1, 2): c.cx(q, q+1)
    for q in range(1, N-1, 2): c.cx(q, q+1)

h = Hamiltonian(num_qubits=N)
for i in range(N-1):
    p = ['I']*N; p[i]='Z'; p[i+1]='Z'
    h.add_pauli(''.join(p), -1.0)
for i in range(N):
    p = ['I']*N; p[i]='X'
    h.add_pauli(''.join(p), -0.5)
for i in range(N-2):
    p = ['I']*N; p[i]='Z'; p[i+2]='Z'
    h.add_pauli(''.join(p), -0.3)
print(f'  Hamiltonian: {len(h.coefficients)} terms')

theta = torch.randn(c.params_size, device=DEVICE)
P = theta.numel()
print(f'  Circuit: P={P} params')

for method_name, measure_method, shots in [
    ('EC',     MeasureMethod.EFFICIENT_CONTRACTION, 0),
    ('RSS-1k', MeasureMethod.RIGHT_SUFFIX_SAMPLING, 1000),
]:
    grad = BatchParameterShiftGradient(
        shift=math.pi/2, batch_size=P, shots=shots,
        measure_method=measure_method, depth=1,
    )
    # Warmup
    grad.run(theta, c, h)
    if DEVICE == 'cuda': torch.cuda.synchronize()

    times = []
    for _ in range(5):
        if DEVICE == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        grad.run(theta, c, h)
        if DEVICE == 'cuda': torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times)//2]
    print(f'  {method_name:8s}: {med*1000:.0f}ms/gradient (median of 5)')
