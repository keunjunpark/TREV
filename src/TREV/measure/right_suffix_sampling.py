from typing import Optional, List

import torch

from ..hamiltonian.hamiltonian import Hamiltonian, rotate_tensor_for_measurement
def kron(A, B):  # alias
    return torch.kron(A, B)

def E_site(core):  # core: [chi,2,chi] -> (E, (E0,E1)) in double layer
    B0, B1 = core[:,:,0], core[:,:,1]
    E0 = kron(B0, B0.conj())
    E1 = kron(B1, B1.conj())
    return (E0 + E1), (E0, E1)

@torch.no_grad()
def precompute_double_layer_and_right_suffix(cores):
    """
    Returns:
      Es: [(E0,E1)] length n, each [chi^2,chi^2]
      R_suf: [n] where R_suf[i] = E_{i+1} ... E_{n-1} (identity if i==n-1)
      d2, device, dtype
    """
    device = cores[0].device
    dtype  = torch.complex128 if torch.is_complex(cores[0]) else torch.float64

    E_list, Es = [], []
    for c in cores:
        c = c.to(dtype)
        Ei, (Ei0, Ei1) = E_site(c)
        E_list.append(Ei)
        Es.append((Ei0, Ei1))

    n  = len(E_list)
    d2 = E_list[0].shape[0]
    I  = torch.eye(d2, dtype=E_list[0].dtype, device=device)

    # Build right suffixes with a left->right pass on the reversed list,
    # but crucially LEFT-multiply to preserve forward order:
    # Rpref_rev[j] = E_{n-1} ... E_{n-j}
    Rpref_rev = [None] * (n + 1)
    acc = I
    Rpref_rev[0] = acc
    E_rev = E_list[::-1]  # [E_{n-1}, E_{n-2}, ..., E_0]
    for j in range(1, n + 1):
        acc = E_rev[j - 1] @ acc        # <-- left-multiply (critical)
        Rpref_rev[j] = acc

    # Map back: for site i, R_suf[i] = E_{i+1} ... E_{n-1} = Rpref_rev[n-(i+1)]
    R_suf = [Rpref_rev[n - (i + 1)] for i in range(n)]
    return Es, R_suf, d2, device, dtype


@torch.no_grad()
def expectation_value(
    cores, hamiltonian, shots=10_000, chunk_size=128, term_chunk=4096, seed=None
):
    groups = hamiltonian.get_qwc_groups()
    op_tensor = hamiltonian.get_pauli_op_tensor().to(cores[0].device)  # (T, N) uint8
    all_coeffs = hamiltonian.coefficients
    n = len(cores)
    shots_per_group = max(1, shots // len(groups))
    device = cores[0].device
    cdtype = cores[0].dtype

    # Stack cores into tensor for rotation
    tensor = torch.stack(cores, dim=0)  # (N, chi, chi, 2)

    # RNG
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    grand_total = 0.0

    for group in groups:
        idx = group['term_indices']

        # Rotate cores for this group's measurement basis
        rotated = rotate_tensor_for_measurement(tensor, group['basis'])
        rot_cores = [rotated[i] for i in range(n)]

        # Non-I mask and coefficients for this group's terms
        group_nonI = (op_tensor[idx] != 0).to(device=device, dtype=torch.bool)  # (G, N)
        group_coeffs = torch.tensor(
            [all_coeffs[t] for t in idx], dtype=torch.float64, device=device
        )

        # Precompute double layer and right suffix for rotated cores
        Es, R_suf, d2, _, _ = precompute_double_layer_and_right_suffix(rot_cores)
        chi = int(d2**0.5)
        chi2 = chi * chi

        Es = [(E0.to(cdtype), E1.to(cdtype)) for (E0, E1) in Es]
        R_suf_cast = [Ri.to(cdtype) for Ri in R_suf]
        R_bl = [R_suf_cast[i].view(chi, chi, chi, chi).permute(3, 1, 2, 0)
                .contiguous().reshape(chi2, chi2)
                for i in range(n)]
        del R_suf, R_suf_cast

        A0 = [rot_cores[i][:, :, 0].to(cdtype).contiguous() for i in range(n)]
        A1 = [rot_cores[i][:, :, 1].to(cdtype).contiguous() for i in range(n)]

        total = torch.zeros((), dtype=torch.float64, device=device)
        done = 0

        for s0 in range(0, shots_per_group, chunk_size):
            s1 = min(s0 + chunk_size, shots_per_group)
            B = s1 - s0
            X = torch.eye(chi, dtype=cdtype, device=device).expand(B, chi, chi).clone()
            bits = torch.empty((B, n), dtype=torch.bool, device=device)

            for i in range(n):
                M0 = X @ A0[i]; M1 = X @ A1[i]

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

                nX = torch.linalg.norm(X.reshape(B, -1), dim=1).clamp_min(1e-300).view(B, 1, 1)

            # Score using non-I mask for this group
            bf = bits.to(torch.float32)
            cnt = bf @ group_nonI.to(torch.float32).T  # (B, G)
            sgn = torch.where((cnt.remainder_(2.0) > 0.5), -1.0, 1.0).to(torch.float64)
            Eb = sgn @ group_coeffs  # (B,)
            total += Eb.sum()
            done += B

        grand_total += (total / max(1, done)).item()

    return grand_total

@torch.no_grad()
def expectation_value_batch(
    param_batch: torch.Tensor,           # (B_total, P)
    circuit,                             # .device and build_tensor_batch(params,B)->(B,N,χ,χ,2)
    hamiltonian,                         # .get_bool_pauli_tensor()->(T,N) bool; .coefficients (len T)
    shots: int = 10_000,
    chunk_size: Optional[int] = None,    # shots chunk
    term_chunk: int = 4096,              # Hamiltonian term chunk
    seed: int | None = None,
    *,
    param_chunk: Optional[int] = None,   # split B into chunks to fit memory
    normalize_every: int = 8,            # periodic normalization of X for stability
    use_complex64: bool = True,          # internal complex precision
) -> torch.Tensor:
    """
    Batched Monte Carlo ⟨ψ(θ)|H|ψ(θ)⟩ via right-suffix sampling.
    Vectorizes over parameter sets (B) and samples (S), and streams Hamiltonian terms in chunks.

    Returns:
        (B_total,) float64 tensor of estimated expectations (on CPU).
    """
    device = getattr(circuit, "device", param_batch.device)

    B_total = int(param_batch.shape[0])
    if param_chunk is None or param_chunk >= B_total:
        batch_slices = [(0, B_total)]
    else:
        batch_slices = [(s, min(s + param_chunk, B_total)) for s in range(0, B_total, param_chunk)]

    if chunk_size is None:
        chunk_size = shots

    # QWC groups and Hamiltonian data
    groups = hamiltonian.get_qwc_groups()
    op_tensor = hamiltonian.get_pauli_op_tensor().to(device)  # (T, N) uint8
    all_coeffs = hamiltonian.coefficients
    N = op_tensor.shape[1]
    shots_per_group = max(1, shots // len(groups))

    # RNG
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    out_parts = []

    for lo, hi in batch_slices:
        param_view = param_batch[lo:hi]
        B = int(param_view.shape[0])
        ring = circuit.build_tensor_batch(param_view, B)     # (B, N, chi, chi, 2)
        if use_complex64:
            ctype = ring.dtype if ring.is_complex() else torch.cfloat
        else:
            ctype = torch.complex128
        _, N_chk, chi_l, chi_r, d = ring.shape
        assert N_chk == N and d == 2 and chi_l == chi_r, "Mismatch in circuit vs. Hamiltonian."
        chi = chi_l
        chi2 = chi * chi

        grand_totals = torch.zeros(B, dtype=torch.float64, device=device)

        for group in groups:
            idx = group['term_indices']
            nonI_mask = (op_tensor[idx] != 0).to(device=device, dtype=torch.bool)  # (G, N)
            group_coeffs = torch.as_tensor(
                [all_coeffs[t] for t in idx], dtype=torch.float64, device=device
            )
            G = len(idx)

            # Rotate ring for this group's measurement basis
            rotated = rotate_tensor_for_measurement(ring, group['basis'])

            A0_sites = [rotated[:, i, :, :, 0].to(ctype).contiguous() for i in range(N)]
            A1_sites = [rotated[:, i, :, :, 1].to(ctype).contiguous() for i in range(N)]

            # Build R_suf via Kronecker-free O(chi^5) contraction
            acc = torch.eye(chi2, dtype=ctype, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
            R_suf = [None] * N
            for i in range(N - 1, -1, -1):
                R_suf[i] = acc
                A0i, A1i = A0_sites[i], A1_sites[i]
                acc_view = acc.view(B, chi, chi, chi2)
                temp = torch.matmul(A0i.conj().unsqueeze(1), acc_view)
                new_acc = torch.matmul(A0i, temp.reshape(B, chi, chi * chi2))
                del temp
                temp = torch.matmul(A1i.conj().unsqueeze(1), acc_view)
                new_acc += torch.matmul(A1i, temp.reshape(B, chi, chi * chi2))
                del temp
                acc = new_acc.view(B, chi, chi, chi2).reshape(B, chi2, chi2).contiguous()
                del new_acc
            del acc

            # Convert R_suf from kron convention to bilinear form
            for i in range(N):
                R_suf[i] = (R_suf[i].view(B, chi, chi, chi, chi)
                            .permute(0, 3, 1, 4, 2)
                            .conj()
                            .contiguous()
                            .reshape(B, chi2, chi2))

            # Monte Carlo accumulation over shot-chunks
            totals = torch.zeros(B, dtype=torch.float64, device=device)
            done = torch.zeros((), dtype=torch.int64, device=device)

            Ichi = torch.eye(chi, dtype=ctype, device=device)

            for s0 in range(0, shots_per_group, chunk_size):
                s1 = min(s0 + chunk_size, shots_per_group)
                S = s1 - s0

                X = Ichi.expand(B, S, chi, chi).clone()
                bits = torch.empty((B, S, N), dtype=torch.bool, device=device)

                for i in range(N):
                    A0i = A0_sites[i]
                    A1i = A1_sites[i]
                    Ri = R_suf[i]

                    M0 = torch.matmul(X, A0i.unsqueeze(1))
                    M1 = torch.matmul(X, A1i.unsqueeze(1))

                    v0 = M0.reshape(B, S, chi2)
                    v1 = M1.reshape(B, S, chi2)
                    y0 = torch.matmul(v0, Ri.mT)
                    y1 = torch.matmul(v1, Ri.mT)
                    w0 = (v0.conj() * y0).sum(-1).real
                    w1 = (v1.conj() * y1).sum(-1).real
                    den = (w0 + w1).clamp_min(1e-300)
                    p1 = (w1 / den)

                    u = torch.rand((B, S), generator=gen, device=device)
                    si = (u < p1)
                    bits[:, :, i] = si

                    si_view = si.view(B, S, 1, 1)
                    X = torch.where(si_view, M1, M0)

                    if normalize_every > 0 and (i % normalize_every) == 0 and i != 0:
                        nX = torch.linalg.norm(X.reshape(B, S, -1), dim=-1).clamp_min(1e-300).view(B, S, 1, 1)
                        X = X / nX

                # Score using non-I mask for this group
                bf = bits.to(torch.float32).reshape(B * S, N)
                cnt = bf @ nonI_mask.to(torch.float32).T  # (B*S, G)
                parity = (cnt.remainder_(2.0) > 0.5)
                sgn = torch.where(parity, -1.0, 1.0)
                Eb = (sgn * group_coeffs.view(1, -1)).sum(dim=1)  # (B*S,)

                Eb = Eb.view(B, S)
                totals += Eb.sum(dim=1)
                done += S

            if done > 0:
                grand_totals += (totals / done.clamp_min(1)).detach()

            del R_suf, A0_sites, A1_sites

        out_parts.append(grand_totals)

        del ring
        torch.cuda.empty_cache()

    return torch.cat(out_parts, dim=0).cpu()

@torch.no_grad()
def argmax_bitstring_tr_right_suffix(cores, bit_order="LE", normalize_every=8):
    """
    Return the highest-probability bitstring for a tensor ring using right-suffix
    perfect-sampling weights (no random sampling).
    - bit_order: "LE" (site 0 = LSB) or "BE" (flip at the end if you want MSB-first)
    """
    # Build double-layer blocks and right suffixes (exact same as your sampler)
    Es, R_suf, d2, device, dtype = precompute_double_layer_and_right_suffix(cores)
    n = len(Es)

    # Single-sample left env in double-layer space
    L = torch.eye(d2, dtype=dtype, device=device)

    bits = torch.zeros((n,), dtype=torch.long, device=device)
    logp = torch.zeros((), dtype=torch.float64, device=device)
    eps  = 1e-300

    for i in range(n):
        Ei0, Ei1 = Es[i]          # [d2,d2] each
        T0 = L @ Ei0              # [d2,d2]
        T1 = L @ Ei1

        Ri = R_suf[i]             # [d2,d2]
        # Weights w_s = Tr( (L E_i(s)) * R_i ), real and ≥ 0 up to numerics
        w0 = torch.trace(T0 @ Ri).real
        w1 = torch.trace(T1 @ Ri).real
        w0c = torch.clamp(w0, min=0.0)
        w1c = torch.clamp(w1, min=0.0)
        den = (w0c + w1c).clamp_min(eps)

        # Greedy choice: argmax conditional
        if w1c >= w0c:
            si = 1
            L  = T1
            num = w1c
        else:
            si = 0
            L  = T0
            num = w0c

        bits[i] = si
        # accumulate exact log-prob
        logp += (num.double() + eps).log() - (den.double()).log()

        # occasional stabilization
        if normalize_every and (i % normalize_every == 0 and i != 0):
            nL = torch.linalg.norm(L.reshape(-1)).clamp_min(eps)
            L /= nL

    # If you want MSB-first output, flip once
    if bit_order.upper() == "BE":
        bits = bits.flip(dims=[0])
    bitstring = ''.join(str(int(x.item())) for x in bits)
    return bitstring