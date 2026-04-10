"""
Profile: gradient breakdown — build vs contraction per chunk.
Run: python profile_full.py
"""
import time, torch
from contextlib import contextmanager
from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.optimization.gradients.batch_parameter_shift import (
    expectation_value_batch_efficient_contraction,
)
from TREV.measure.enums import MeasureMethod

device = 'cuda'

class T:
    def __init__(self): self.r = {}
    @contextmanager
    def t(self, name):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        yield
        torch.cuda.synchronize(); self.r.setdefault(name, []).append((time.perf_counter()-t0)*1000)
    def show(self):
        tot = sum(sum(v) for v in self.r.values())
        for n, v in self.r.items():
            s = sum(v); print(f"    {n:<40} {s:>7.1f}ms {s/tot*100:>5.1f}%  x{len(v)}")
        print(f"    {'TOTAL':<40} {tot:>7.1f}ms\n")

torch.manual_seed(42)

for N, chi in [(8,4), (8,10), (12,10)]:
    c = Circuit(num_qubit=N, rank=chi, device=device)
    for i in range(N): c.h(i)
    for _ in range(2):
        for i in range(N): c.cx(i,(i+1)%N)
        for i in range(N): c.ry(i); c.rz(i)
    h = Hamiltonian(num_qubits=N)
    for i in range(N):
        j=(i+1)%N; p=['I']*N; p[i]='Z'; p[j]='Z'
        h.add_pauli(''.join(p), 0.5)
    h.add_pauli('I'*N, 1.0)

    P = c.params_size
    B = 8
    theta = torch.randn(P, device=device)
    base = theta.unsqueeze(0)

    # warmup
    params = torch.randn(B, P, device=device)
    expectation_value_batch_efficient_contraction(params, c, h, shots=0)

    print(f"N={N} chi={chi} P={P} B={B} chunks={((P+B-1)//B)}")

    # Simulate gradient loop with timing
    tm = T()
    shift = 1.5708
    for start in range(0, P, B):
        stop = min(start + B, P)
        idx = torch.arange(start, stop, device=device)

        with tm.t("param_batch_build"):
            plus = base.repeat(len(idx), 1)
            minus = plus.clone()
            plus[torch.arange(len(idx)), idx] += shift
            minus[torch.arange(len(idx)), idx] -= shift
            batch = torch.cat([plus, minus], dim=0)

        with tm.t("build_tensor_batch"):
            ring = c.build_tensor_batch(batch, batch.shape[0])

        with tm.t("contraction"):
            exp_vals = expectation_value_batch_efficient_contraction(batch, c, h, shots=0)

    tm.show()
