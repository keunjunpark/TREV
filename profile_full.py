"""
Profile: find remaining bottlenecks after all optimizations.
Run: python profile_full.py
"""
import time, torch
from contextlib import contextmanager
from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.optimization.gradients.batch_parameter_shift import (
    expectation_value_batch_efficient_contraction,
    batch_gradient, batch_gradient_cached,
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
            s = sum(v); print(f"    {n:<45} {s:>7.1f}ms {s/tot*100:>5.1f}%  x{len(v)}")
        print(f"    {'TOTAL':<45} {tot:>7.1f}ms")

torch.manual_seed(42)

def hea(N, chi, L):
    c = Circuit(num_qubit=N, rank=chi, device=device)
    for i in range(N): c.h(i)
    for _ in range(L):
        for i in range(N): c.cx(i,(i+1)%N)
        for i in range(N): c.ry(i); c.rz(i)
    return c

def ham(N):
    h = Hamiltonian(num_qubits=N)
    for i in range(N):
        j=(i+1)%N; p=['I']*N; p[i]='Z'; p[j]='Z'
        h.add_pauli(''.join(p), 0.5)
    h.add_pauli('I'*N, 1.0)
    return h

for N, chi, L in [(8,4,2), (8,10,2), (12,4,2), (12,10,2)]:
    c = hea(N, chi, L)
    h = ham(N)
    P = c.params_size
    B = min(8, P)
    theta = torch.randn(P, device=device)
    shift = 1.5708

    # warmup
    batch_gradient_cached(theta, c, h, B, 0, shift, MeasureMethod.EFFICIENT_CONTRACTION)

    print(f"N={N} chi={chi} L={L} P={P} B={B}")
    print("-" * 60)

    # --- 1. Gradient: original vs cached ---
    tm = T()
    with tm.t("gradient_original"):
        batch_gradient(theta, c, h, B, 0, shift, 1, 0, False, MeasureMethod.EFFICIENT_CONTRACTION)
    with tm.t("gradient_cached"):
        batch_gradient_cached(theta, c, h, B, 0, shift, MeasureMethod.EFFICIENT_CONTRACTION)
    tm.show()
    print()

    # --- 2. Breakdown: cached gradient internals ---
    # Simulate what batch_gradient_cached does
    cps = c.get_prefix_checkpoints()
    params_batch = torch.randn(2*B, P, device=device)

    tm = T()
    for seg_gate, (plo, phi) in cps:
        with tm.t(f"prefix_build (to gate {seg_gate})"):
            prefix = c.build_prefix_batch(theta, 1, seg_gate)

        with tm.t(f"prefix_expand+clone"):
            pexp = prefix.expand(2*B, -1, -1, -1, -1).clone()

        with tm.t(f"replay_from_gate_{seg_gate}"):
            ring = c.build_from_prefix_batch(pexp, params_batch, 2*B, seg_gate)

        with tm.t(f"contraction"):
            # inline the contraction
            ctype = torch.cfloat
            paulis = h.get_bool_pauli_tensor().to(device)
            coeffs = torch.as_tensor(h.coefficients, dtype=ctype, device=device)
            Tc, _ = paulis.shape
            chi2 = chi*chi
            A0 = ring[...,0].to(ctype)
            A1 = ring[...,1].to(ctype)
            BT_val = 2*B*Tc
            eye = torch.eye(chi2, dtype=ctype, device=device).unsqueeze(0).expand(BT_val,-1,-1)
            Prod = eye
            for i in range(N):
                A0_exp = A0[:,i].unsqueeze(1).expand(-1,Tc,-1,-1).reshape(BT_val,chi,chi)
                A1_exp = A1[:,i].unsqueeze(1).expand(-1,Tc,-1,-1).reshape(BT_val,chi,chi)
                P4 = Prod.reshape(BT_val,chi2,chi,chi)
                t = torch.matmul(P4, A0_exp.unsqueeze(1))
                r0 = torch.matmul(A0_exp.conj().mT.unsqueeze(1), t).reshape(BT_val,chi2,chi2)
                t = torch.matmul(P4, A1_exp.unsqueeze(1))
                r1 = torch.matmul(A1_exp.conj().mT.unsqueeze(1), t).reshape(BT_val,chi2,chi2)
                mi = paulis[:,i].view(1,Tc,1,1).expand(2*B,Tc,chi2,chi2).reshape(BT_val,chi2,chi2)
                Prod = torch.where(mi, r0-r1, r0+r1)
            Prod = Prod.reshape(2*B,Tc,chi2,chi2)
            tr = Prod.diagonal(0,2,3).sum(-1)
            _ = (tr * coeffs.view(1,Tc)).sum(1)
        break  # one segment is enough to see the breakdown

    print(f"  Per-segment breakdown (segment 0, θ[{cps[0][1][0]}:{cps[0][1][1]}]):")
    tm.show()
    print()

    # --- 3. Micro: individual operations ---
    tm = T()
    ring = c.build_tensor_batch(params_batch[:B], B)
    A0_i = ring[:,0,:,:,0].to(torch.cfloat)
    A1_i = ring[:,0,:,:,1].to(torch.cfloat)
    Prod_flat = torch.randn(B*Tc, chi2, chi2, dtype=torch.cfloat, device=device)

    with tm.t("matmul (chi^5 kron step)"):
        for _ in range(N):
            P4 = Prod_flat.reshape(B*Tc, chi2, chi, chi)
            A0e = A0_i.unsqueeze(1).expand(-1,Tc,-1,-1).reshape(B*Tc,chi,chi)
            t = torch.matmul(P4, A0e.unsqueeze(1))
            torch.matmul(A0e.conj().mT.unsqueeze(1), t)

    with tm.t("torch.where (I/Z select)"):
        mi = torch.randint(0,2,(B*Tc,chi2,chi2), device=device, dtype=torch.bool)
        r0 = Prod_flat; r1 = Prod_flat
        for _ in range(N):
            torch.where(mi, r0, r1)

    with tm.t("SVD (one 2q gate)"):
        mps = torch.randn(B, 2*chi, 2*chi, dtype=torch.cfloat, device=device)
        torch.linalg.svd(mps, full_matrices=False)

    with tm.t("1q gate einsum"):
        state = torch.randn(B, chi, chi, 2, dtype=torch.cfloat, device=device)
        gate = torch.randn(B, 2, 2, dtype=torch.cfloat, device=device)
        for _ in range(N):
            torch.einsum('bij,bklj->bikl', gate, state)

    print(f"  Micro-benchmarks (per-call, B={B}):")
    tm.show()
    print("=" * 60)
    print()
