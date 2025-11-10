from typing import Optional, List

import torch

from ..hamiltonian.hamiltonian import Hamiltonian
def kron(A, B):  # alias
    return torch.kron(A, B)

def E_site(core):  # core: [chi,2,chi] -> (E, (E0,E1)) in double layer
    B0, B1 = core[:,:,0], core[:,:,1]
    E0 = kron(B0, B0.conj())
    E1 = kron(B1, B1.conj())
    # E0 = B0
    # E1 = B1
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
    # Build in the *same* basis as your working code
    Es, R_suf, d2, device, _ = precompute_double_layer_and_right_suffix(cores)
    cdtype = cores[0].dtype           # keep dtype consistent with cores
    n, chi = len(Es), int(d2**0.5)

    # Cast once
    Es    = [(E0.to(cdtype), E1.to(cdtype)) for (E0,E1) in Es]
    R_suf = [Ri.to(cdtype) for Ri in R_suf]

    for i in range(n):
        print(f"site {i} rsuf")
        print(R_suf[i])
    

    print("first rsuf")
    print(R_suf[0])
    print("last rsuf")
    print(R_suf[-1])
    # # Detect recipe
    # orient, perms, err = detect_full_recipe(cores, Es, R_suf, trials=8)
    # print(f"Detected single-layer sampling recipe: orient={orient}, max |Δw|={err:.2e}")
    # print("  Suffix perms per site:", perms)
    # Prep per-site R4 with detected permutation
    R4 = [ R_suf[i].view(chi,chi,chi,chi).permute(2,3,0,1).contiguous() for i in range(n) ]
    # Single-layer slices
    A0 = [cores[i][:,:,0].to(cdtype).contiguous() for i in range(n)]
    A1 = [cores[i][:,:,1].to(cdtype).contiguous() for i in range(n)]

    # H (streaming; no (B×T) buffer)
    coeffs = torch.as_tensor(hamiltonian.coefficients, dtype=torch.float64, device=device)
    zmask  = hamiltonian.get_bool_pauli_tensor().to(device=device, dtype=torch.bool)
    T = int(coeffs.numel())

    # RNG
    gen = torch.Generator(device=device)
    if seed is not None: gen.manual_seed(seed)

    total = torch.zeros((), dtype=torch.float64, device=device); done = 0
    for s0 in range(0, shots, chunk_size):
        s1 = min(s0 + chunk_size, shots); B = s1 - s0
        X    = torch.eye(chi, dtype=cdtype, device=device).expand(B,chi,chi).clone()
        bits = torch.empty((B,n), dtype=torch.bool, device=device)

        for i in range(n):

            M0 = X @ A0[i];  M1 = X @ A1[i]


            w0 = torch.einsum('sab,scd,acbd->s', M0, M0.conj(), R4[i]).real
            w1 = torch.einsum('sab,scd,acbd->s', M1, M1.conj(), R4[i]).real

            den = (w0 + w1).clamp_min(1e-300)
            p0  = (w0 / den).to(torch.float64)
            si  = (torch.rand((B,), generator=gen, device=device) >= p0)
            bits[:, i] = si
            X = torch.where(si.view(B,1,1), M1, M0)

            # if (i % 8) == 0 and i != 0:
            nX = torch.linalg.norm(X.reshape(B,-1), dim=1).clamp_min(1e-300).view(B,1,1)
            #X  = X / nX

        # stream scoring
        Eb = torch.zeros((B,), dtype=torch.float64, device=device)
        bf = bits.to(torch.float32)
        for t0 in range(0, T, term_chunk):
            t1 = min(t0 + term_chunk, T)
            Zblk = zmask[t0:t1, :]; Cblk = coeffs[t0:t1]
            cnt = bf @ Zblk.to(torch.float32).T
            sgn = torch.where((cnt.remainder_(2.0) > 0.5), -1.0, 1.0).to(torch.float64)
            Eb += sgn @ Cblk
        total += Eb.sum(); done += B

    return (total / max(1, done)).item()


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