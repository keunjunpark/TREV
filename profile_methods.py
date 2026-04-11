"""
Profile: Breakdown of time spent in efficient_contraction and right_suffix_sampling.

Shows where the wall-clock time goes for each method at different (N, chi) scales.
Designed to run on a laptop GPU (small configs).
"""

import time
import torch
import numpy as np
from contextlib import contextmanager

from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.efficient_contraction import _kron_contract_right_4d
from TREV.hamiltonian.hamiltonian import rotate_tensor_for_measurement
from TREV.measure.right_suffix_sampling import (
    precompute_double_layer_and_right_suffix,
)


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
    return circuit, theta


class Timer:
    def __init__(self, device='cuda'):
        self.device = device
        self.records = {}

    @contextmanager
    def track(self, name):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(elapsed)

    def summary(self):
        total = sum(sum(v) for v in self.records.values())
        print(f"  {'section':<30} {'time (ms)':>10} {'%':>7}")
        print(f"  {'-'*30} {'-'*10} {'-'*7}")
        for name, vals in self.records.items():
            t = sum(vals)
            pct = t / total * 100 if total > 0 else 0
            print(f"  {name:<30} {t:>9.2f}ms {pct:>6.1f}%")
        print(f"  {'TOTAL':<30} {total:>9.2f}ms {'100.0':>6}%")


# ── Profile: Efficient Contraction ──

def profile_efficient_contraction(tensor, hamiltonian, device='cuda'):
    """Break down efficient_contraction into timed stages."""
    timer = Timer(device)
    cdtype = tensor.dtype if tensor.is_complex() else torch.cfloat

    with timer.track("get_pauli_op_tensor"):
        op_tensor = hamiltonian.get_pauli_op_tensor().to(device)
    Tc, N = op_tensor.shape

    with timer.track("prepare coefficients"):
        coeffs = torch.as_tensor(
            [c.item() if hasattr(c, 'item') else c for c in hamiltonian.coefficients],
            dtype=cdtype, device=device,
        )

    chi = tensor.shape[1]

    with timer.track("cache A0/A1 slices"):
        sites = []
        for i in range(N):
            A = tensor[i].to(device=device, dtype=cdtype)
            sites.append((A[:, :, 0].contiguous(), A[:, :, 1].contiguous()))

    eye4 = torch.eye(chi * chi, dtype=cdtype, device=device).reshape(chi, chi, chi, chi)

    with timer.track("left prefix (all-I)"):
        L_pre = [None] * (N + 1)
        acc = eye4
        for i in range(N):
            L_pre[i] = acc
            A0, A1 = sites[i]
            acc = _kron_contract_right_4d(acc, A0, A1)
        L_pre[N] = acc

    with timer.track("right suffix (all-I)"):
        R_suf_T = [None] * (N + 1)
        R_suf_T[N] = eye4
        acc = eye4
        for i in range(N - 1, -1, -1):
            A0, A1 = sites[i]
            acc = _kron_contract_right_4d(acc, A0.mT, A1.mT)
            R_suf_T[i] = acc

    with timer.track("per-term contraction loop"):
        total = torch.zeros((), dtype=cdtype, device=device)
        for t in range(Tc):
            non_i_sites = torch.where(op_tensor[t] != 0)[0].tolist()
            if len(non_i_sites) == 0:
                total += coeffs[t] * (L_pre[N] * R_suf_T[N]).sum()
                continue
            s_first = non_i_sites[0]
            s_last = non_i_sites[-1]
            run = L_pre[s_first].clone()
            for i in range(s_first, s_last + 1):
                A0, A1 = sites[i]
                op_i = op_tensor[t, i].item()
                run = _kron_contract_right_4d(run, A0, A1, op=op_i)
            total += coeffs[t] * (run * R_suf_T[s_last + 1]).sum()

    return timer


# ── Profile: Right Suffix Sampling ──

def profile_right_suffix_sampling(tensor, hamiltonian, shots=1000, chunk_size=128, device='cuda'):
    """Break down right_suffix_sampling into timed stages."""
    timer = Timer(device)

    N = tensor.shape[0]
    cdtype = tensor.dtype if tensor.is_complex() else torch.cfloat

    with timer.track("QWC grouping"):
        groups = hamiltonian.get_qwc_groups()

    with timer.track("get_pauli_op_tensor"):
        op_tensor = hamiltonian.get_pauli_op_tensor().to(device)

    all_coeffs = hamiltonian.coefficients
    shots_per_group = max(1, shots // len(groups))

    gen = torch.Generator(device=device)
    gen.manual_seed(42)

    grand_total = 0.0

    for gi, group in enumerate(groups):
        idx = group['term_indices']

        with timer.track(f"rotate_tensor (group {gi})"):
            rotated = rotate_tensor_for_measurement(tensor, group['basis'])
            rot_cores = [rotated[i] for i in range(N)]

        with timer.track(f"precompute_double_layer+R_suf (group {gi})"):
            Es, R_suf, d2, _, _ = precompute_double_layer_and_right_suffix(rot_cores)
            chi = int(d2**0.5)
            chi2 = chi * chi

        with timer.track(f"cast+reshape R_suf (group {gi})"):
            Es = [(E0.to(cdtype), E1.to(cdtype)) for (E0, E1) in Es]
            R_suf_cast = [Ri.to(cdtype) for Ri in R_suf]
            R_bl = [R_suf_cast[i].view(chi, chi, chi, chi).permute(3, 1, 2, 0)
                    .contiguous().reshape(chi2, chi2)
                    for i in range(N)]

        with timer.track(f"prepare A0/A1 (group {gi})"):
            A0 = [rot_cores[i][:, :, 0].to(cdtype).contiguous() for i in range(N)]
            A1 = [rot_cores[i][:, :, 1].to(cdtype).contiguous() for i in range(N)]

        group_nonI = (op_tensor[idx] != 0).to(device=device, dtype=torch.bool)
        group_coeffs = torch.tensor(
            [all_coeffs[t] for t in idx], dtype=torch.float64, device=device
        )

        total = torch.zeros((), dtype=torch.float64, device=device)
        done = 0

        for s0 in range(0, shots_per_group, chunk_size):
            s1 = min(s0 + chunk_size, shots_per_group)
            B = s1 - s0

            with timer.track("sampling loop (per-site)"):
                X = torch.eye(chi, dtype=cdtype, device=device).expand(B, chi, chi).clone()
                bits = torch.empty((B, N), dtype=torch.bool, device=device)

                for i in range(N):
                    M0 = X @ A0[i]
                    M1 = X @ A1[i]
                    Ri = R_bl[i]
                    v0 = M0.reshape(B, chi2)
                    v1 = M1.reshape(B, chi2)
                    y0 = torch.matmul(v0, Ri.mT)
                    y1 = torch.matmul(v1, Ri.mT)
                    w0 = (v0.conj() * y0).sum(-1).real
                    w1 = (v1.conj() * y1).sum(-1).real
                    den = (w0 + w1).clamp_min(1e-300)
                    p0 = (w0 / den).to(torch.float64)
                    si = (torch.rand((B,), generator=gen, device=device) >= p0)
                    bits[:, i] = si
                    X = torch.where(si.view(B, 1, 1), M1, M0)

            with timer.track("scoring (parity + coeffs)"):
                bf = bits.to(torch.float32)
                cnt = bf @ group_nonI.to(torch.float32).T
                sgn = torch.where((cnt.remainder_(2.0) > 0.5), -1.0, 1.0).to(torch.float64)
                Eb = sgn @ group_coeffs
                total += Eb.sum()
                done += B

        grand_total += (total / max(1, done)).item()

    return timer


# ── Main ──

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

    for N, chi in configs:
        torch.cuda.empty_cache()
        ham = build_maxcut_hamiltonian(N)
        circuit, theta = build_tensor(N, chi, device)
        tensor = circuit.build_tensor(theta)

        # Warmup
        from TREV.measure.efficient_contraction import expectation_value_batch
        expectation_value_batch(tensor, ham, device=device)

        print("=" * 65)
        print(f"  N={N}, chi={chi}, Hamiltonian terms={len(ham.paulis)}")
        print("=" * 65)

        # Profile efficient contraction
        print(f"\n  --- Efficient Contraction ---")
        timer_ec = profile_efficient_contraction(tensor, ham, device)
        timer_ec.summary()

        # Profile right suffix sampling
        print(f"\n  --- Right Suffix Sampling (shots=1000, chunk=128) ---")
        timer_rs = profile_right_suffix_sampling(tensor, ham, shots=1000, chunk_size=128, device=device)
        timer_rs.summary()
        print()


if __name__ == '__main__':
    main()
