import time
from typing import List, Optional
import os

import torch
from torch import Tensor, no_grad

from TREV.optimization.gradients.set_batch_size import auto_batch_size

from ...circuit import Circuit
from ...hamiltonian.hamiltonian import Hamiltonian
from ...measure.enums import MeasureMethod
from ...optimization.gradients.gradient import Gradient


def _gpu_info(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    name = torch.cuda.get_device_name(device)
    try:
        free_b, total_b = torch.cuda.mem_get_info(device)
        gb = 1024**3
        return f"{name} (free {free_b/gb:.2f} GB / total {total_b/gb:.2f} GB)"
    except Exception:
        return name


def _dispatch_expectation(param_batch, circuit, hamiltonian, shots, measure_method):
    """Route to the correct batched expectation value backend."""
    if measure_method == MeasureMethod.EFFICIENT_CONTRACTION:
        return expectation_value_batch_efficient_contraction(param_batch, circuit, hamiltonian, shots)
    elif measure_method == MeasureMethod.RIGHT_SUFFIX_SAMPLING:
        return expectation_value_batch_right_suffix(param_batch, circuit, hamiltonian, shots)
    else:
        return expectation_value_batch(param_batch, circuit, hamiltonian, shots)


class BatchParameterShiftGradient(Gradient):
    def __init__(self, shift, batch_size, shots, measure_method: MeasureMethod, depth:int, is_partial:bool=False):
        super().__init__(measure_method)
        self.shift = shift
        self.batch_size = batch_size  # may be None
        self.shots = shots
        self.depth = depth
        self.curr_depth = 0
        self.is_partial = is_partial
        self._autotuned = False

        # optional: control printing via env var
        self._verbose = True

    def run(self, theta: torch.Tensor, circuit: Circuit, hamiltonian: Hamiltonian):
        if (self.batch_size is None) and (not self._autotuned):
            device = torch.device(circuit.device) if isinstance(circuit.device, str) else circuit.device
            P = theta.numel()
            base = theta.detach().to(device).unsqueeze(0)

            def run_batch_fn(bs: int):
                idx = torch.arange(0, min(bs, P), device=device)
                C = idx.numel()
                if C == 0:
                    return
                arange_C = torch.arange(C, device=device)
                # Build a (2C, P) batch without repeat/cat
                param_batch = base.expand(2 * C, -1).clone()
                param_batch[arange_C, idx] += self.shift
                param_batch[C + arange_C, idx] -= self.shift
                _dispatch_expectation(param_batch, circuit, hamiltonian, self.shots, self.measure_method)

            self.batch_size = auto_batch_size(
                run_batch_fn,
                device,
                min_bs=1,
                max_bs=min(4096, P),
                safety_frac=0.85,
                warmup=1,
                use_amp=False,
            )

            self._autotuned = True

            if self._verbose:
                print(
                    f"[TREV] Auto batch_size selected: {self.batch_size} "
                    f"(measure={self.measure_method.name}, total_theta={P}, device={_gpu_info(device)})"
                    f"\n"
                    , flush=True
                )

        val = batch_gradient(theta, circuit, hamiltonian, self.batch_size, self.shots,
                             self.shift, self.depth, self.curr_depth, self.is_partial, self.measure_method)
        self.curr_depth = (self.curr_depth + 1) % self.depth
        return val


def batch_gradient(
        params:     torch.Tensor,           # (P,)
        circuit : Circuit,
        hamiltonian: Hamiltonian,
        chunk_size: int,
        shots: int,
        shift:float,
        depth:int,
        curr_depth:int,
        is_partial:bool,
        measure_method: MeasureMethod
) -> torch.Tensor:
    """
    Memory-frugal parameter-shift gradient.

    params   : (P,)  -- single circuit's parameters
    chunk_size  : how many theta-indices to shift at once
    returns     : (P,)  -- gradient d<O>/d_theta
    """
    with torch.no_grad():
        device = circuit.device
        P      = params.numel()
        grad   = torch.empty(P, device=device, dtype=torch.float32)
        base   = params.detach().to(device).unsqueeze(0)  # (1, P)

        if is_partial:
            dP = P // depth
            start = dP * curr_depth
            stop = min(start + dP, P)
            C = stop - start

            idx = torch.arange(start, stop, device=device)
            arange_C = torch.arange(C, device=device)

            batch = base.expand(2 * C, -1).clone()  # (2C, P)
            batch[arange_C, idx] += shift
            batch[C + arange_C, idx] -= shift

            exp_vals = _dispatch_expectation(batch, circuit, hamiltonian, shots, measure_method)
            grad[start:stop] = 0.5 * (exp_vals[:C] - exp_vals[C:])
        else:
            for start in range(0, P, chunk_size):
                stop   = min(start + chunk_size, P)
                C      = stop - start
                idx    = torch.arange(start, stop, device=device)
                arange_C = torch.arange(C, device=device)

                batch = base.expand(2 * C, -1).clone()  # (2C, P)
                batch[arange_C, idx] += shift
                batch[C + arange_C, idx] -= shift

                exp_vals = _dispatch_expectation(batch, circuit, hamiltonian, shots, measure_method)
                grad[start:stop] = 0.5 * (exp_vals[:C] - exp_vals[C:])
        return grad

def expectation_value_batch(
    param_batch: torch.Tensor,
    circuit: Circuit,
    hamiltonian: Hamiltonian,
    shots: int,
    seed: int | None = None
) -> torch.Tensor:
    """
    Fully parallel run_circuit with batched parameter inputs.
    param_batch: (B, 2L)
    returns: (B,) expectations
    """
    with torch.no_grad():
        if seed is not None:
            # Ensure deterministic sampling across devices
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        B = param_batch.shape[0]
        device = circuit.device
        if device == 'cuda':
            torch.cuda.synchronize()

        ring_tensor_batch = circuit.build_tensor_batch(param_batch, B)
        B, N = ring_tensor_batch.shape[:2]
        # (T, N) bool -> index as paulis[:, site_i] to get (T,) mask for that site
        paulis = hamiltonian.get_bool_pauli_tensor().to(device=device)
        coeffs = torch.as_tensor(hamiltonian.coefficients, dtype=torch.float64, device=device)
        T = int(coeffs.numel())

        q0 = torch.tensor([[1], [0]], dtype=torch.cfloat, device=device)
        q1 = torch.tensor([[0], [1]], dtype=torch.cfloat, device=device)

        shot_chunk = shots
        totals = torch.zeros(B, dtype=torch.float64, device=device)
        done = 0

        for s0 in range(0, shots, shot_chunk):
            s1 = min(s0 + shot_chunk, shots)
            S = s1 - s0
            if S == 0:
                continue

            batch_prev: torch.Tensor | None = None
            bits = torch.empty((B, S, N), dtype=torch.bool, device=device)

            for i in range(N):
                curr_tens = ring_tensor_batch[:, i].contiguous()

                if i == 0:
                    qubit_0 = torch.einsum('bijk,kl->bijl', curr_tens, q0).squeeze(-1)
                    qubit_1 = torch.einsum('bijk,kl->bijl', curr_tens, q1).squeeze(-1)
                    batch_qubit_0 = qubit_0.unsqueeze(1).expand(-1, S, -1, -1).contiguous()
                    batch_qubit_1 = qubit_1.unsqueeze(1).expand(-1, S, -1, -1).contiguous()
                else:
                    contracted = torch.einsum('bsij,bjkl->bsikl', batch_prev, curr_tens)
                    batch_qubit_0 = torch.einsum('bsijk,kl->bsijl', contracted, q0).squeeze(-1).contiguous()
                    batch_qubit_1 = torch.einsum('bsijk,kl->bsijl', contracted, q1).squeeze(-1).contiguous()

                # Compute probabilities safely
                prob_0 = torch.einsum('bsij,bsij->bs', batch_qubit_0.conj(), batch_qubit_0).real
                prob_1 = torch.einsum('bsij,bsij->bs', batch_qubit_1.conj(), batch_qubit_1).real
                total = prob_0 + prob_1

                zero_mask = total == 0
                prob_0 = torch.where(zero_mask, torch.full_like(prob_0, 0.5), prob_0)
                prob_1 = torch.where(zero_mask, torch.full_like(prob_1, 0.5), prob_1)
                total = prob_0 + prob_1
                p0 = prob_0 / total

                # Deterministic sampling if seed given
                rnd = torch.rand(B, S, device=device)
                choose_1 = rnd > p0
                bits[:, :, i] = choose_1

                batch_prev = torch.where(
                    choose_1.unsqueeze(-1).unsqueeze(-1),
                    batch_qubit_1,
                    batch_qubit_0
                )

            # Compute energies for this shot chunk without storing (B,S,T)
            bf = bits.to(torch.float32).reshape(B * S, N)
            Eb = torch.zeros((B * S,), dtype=torch.float64, device=device)

            term_chunk = 4096
            for t0 in range(0, T, term_chunk):
                t1 = min(t0 + term_chunk, T)
                Zblk = paulis[t0:t1, :]
                Cblk = coeffs[t0:t1]
                cnt = bf @ Zblk.to(torch.float32).T
                parity = (cnt.remainder_(2.0) > 0.5)
                sgn = torch.where(parity, -1.0, 1.0)
                Eb += (sgn * Cblk.view(1, -1)).sum(dim=1)

            totals += Eb.view(B, S).sum(dim=1)
            done += S

        if done == 0:
            return torch.zeros((B,), dtype=torch.float32, device=device)

        expectations = (totals / done).float()
        return expectations.detach()



@torch.no_grad()
def expectation_value_batch_efficient_contraction(
    param_batch: Tensor,                 # (B, P)
    circuit,                             # .device and build_tensor_batch(params,B)->(B,N,l,r,2)
    hamiltonian,                         # .get_bool_pauli_tensor()->(T,N); .coefficients (len T)
    shots: int,                          # kept for API compatibility (ignored)
    *,
    term_chunk: Optional[int] = None,    # chunk across Hamiltonian terms
    cache_double_layers: bool = True,    # cache E_I/E_Z per site (avoids recompute across term chunks)
    param_chunk: Optional[int] = None,   # split B across chunks to fit memory
    use_complex64: bool = True,          # internal complex precision
) -> Tensor:
    """
    Exact batched <psi|H|psi> via double-layer contraction in MATRIX form.

    - No rank-5 tensors kept around.
    - Working set ~ O(B * chi^4) only for the CURRENT site/term-chunk.
    - Returns (B,) float32 expectations.
    """
    device = getattr(circuit, "device", param_batch.device)

    # Split param batch if desired
    B_total = int(param_batch.shape[0])
    if param_chunk is None or param_chunk >= B_total:
        batch_slices = [(0, B_total)]
    else:
        batch_slices = [(s, min(s + param_chunk, B_total)) for s in range(0, B_total, param_chunk)]

    # Hamiltonian
    paulis = hamiltonian.get_bool_pauli_tensor().to(device)  # (T, N)
    coeffs = torch.as_tensor(hamiltonian.coefficients, dtype=torch.cfloat if use_complex64 else torch.cdouble, device=device)
    T, N = paulis.shape
    if term_chunk is None:
        term_chunk = T

    out_parts = []

    for lo, hi in batch_slices:
        param_view = param_batch[lo:hi]
        B = int(param_view.shape[0])

        # Build cores
        ring = circuit.build_tensor_batch(param_view, B).to(device)  # (B,N,l,r,2)
        _, N_check, l, r, d = ring.shape
        assert N_check == N and d == 2, "MPS/circuit shape mismatch with Hamiltonian"
        ctype = torch.complex64 if use_complex64 else torch.complex128

        # Local ops
        Z = torch.tensor([[1, 0], [0, -1]], dtype=ctype, device=device)
        I = torch.eye(2, dtype=ctype, device=device)

        ll = l * l
        rr = r * r

        def site_double_layers_matrix(Ab: Tensor):
            """
            Ab: (B,l,r,2)
            Returns E_I, E_Z each (B, ll, rr).
            """
            AO_I = torch.einsum('blrd,dk->blrk', Ab, I)  # (B,l,r,2)
            AO_Z = torch.einsum('blrd,dk->blrk', Ab, Z)  # (B,l,r,2)
            EI5 = torch.einsum('blrd,bLRd->blLrR', Ab.conj(), AO_I)
            EZ5 = torch.einsum('blrd,bLRd->blLrR', Ab.conj(), AO_Z)
            EI  = EI5.reshape(B, ll, rr)
            EZ  = EZ5.reshape(B, ll, rr)
            return EI, EZ

        # Cache per-site double-layer matrices (avoids recompute across term chunks)
        cached = None
        if cache_double_layers:
            cached = []
            for i in range(N):
                EI, EZ = site_double_layers_matrix(ring[:, i].to(ctype))
                cached.append((EI, EZ))

        totals = torch.zeros(B, dtype=ctype, device=device)

        for t0 in range(0, T, term_chunk):
            t1 = min(t0 + term_chunk, T)
            mask = paulis[t0:t1]      # (Tc, N)
            coefs = coeffs[t0:t1]     # (Tc,)
            Tc = mask.size(0)

            if cache_double_layers:
                EI0, EZ0 = cached[0]
            else:
                EI0, EZ0 = site_double_layers_matrix(ring[:, 0].to(ctype))

            m0 = mask[:, 0].view(1, Tc, 1, 1)  # broadcast
            Prod = torch.where(
                m0,
                EZ0.unsqueeze(1).expand(-1, Tc, -1, -1),
                EI0.unsqueeze(1).expand(-1, Tc, -1, -1),
            ).contiguous()  # (B,Tc,ll,rr)

            # Sweep remaining sites
            for i in range(1, N):
                if cache_double_layers:
                    EIi, EZi = cached[i]
                else:
                    EIi, EZi = site_double_layers_matrix(ring[:, i].to(ctype))

                mi = mask[:, i].view(1, Tc, 1, 1)
                Ei = torch.where(
                    mi,
                    EZi.unsqueeze(1).expand(-1, Tc, -1, -1),
                    EIi.unsqueeze(1).expand(-1, Tc, -1, -1),
                )  # (B,Tc,ll,rr)

                Prod = torch.matmul(
                    Prod.reshape(B * Tc, ll, rr),
                    Ei.reshape(B * Tc, rr, rr),
                ).reshape(B, Tc, ll, rr)

            # Ring closure: trace over diagonal
            trace_vals = Prod.diagonal(offset=0, dim1=2, dim2=3).sum(dim=-1)  # (B,Tc)

            totals += (trace_vals * coefs.view(1, Tc)).sum(dim=1)

            del Prod, trace_vals

        out_parts.append(totals.real.float())  # (B,)

        del ring, totals, cached

    # concat across param chunks
    return torch.cat(out_parts, dim=0)


@torch.no_grad()
def expectation_value_batch_right_suffix(
    param_batch: torch.Tensor,           # (B_total, P)
    circuit,                             # .device and build_tensor_batch(params,B)->(B,N,chi,chi,2)
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
    Batched Monte Carlo <psi(theta)|H|psi(theta)> via right-suffix sampling.
    Vectorizes over parameter sets (B) and samples (S), and streams Hamiltonian terms in chunks.

    Returns:
        (B_total,) float64 tensor of estimated expectations (on CPU).
    """
    device = getattr(circuit, "device", param_batch.device)
    ctype = torch.complex64 if use_complex64 else torch.complex128

    B_total = int(param_batch.shape[0])
    if param_chunk is None or param_chunk >= B_total:
        batch_slices = [(0, B_total)]
    else:
        batch_slices = [(s, min(s + param_chunk, B_total)) for s in range(0, B_total, param_chunk)]

    if chunk_size is None:
        chunk_size = shots

    # Hamiltonian data (shared for all param chunks)
    coeffs = torch.as_tensor(hamiltonian.coefficients, dtype=torch.float64, device=device)  # (T,)
    zmask  = hamiltonian.get_bool_pauli_tensor().to(device=device, dtype=torch.bool)        # (T, N)
    T, N   = int(coeffs.numel()), int(zmask.shape[1])

    # RNG
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    out_parts = []

    for lo, hi in batch_slices:
        # ---- Build Tensor-Ring cores for this param sub-batch
        param_view = param_batch[lo:hi]                      # (B, P)
        B = int(param_view.shape[0])
        ring = circuit.build_tensor_batch(param_view, B)     # (B, N, chi, chi, 2)
        _, N_chk, chi_l, chi_r, d = ring.shape
        assert N_chk == N and d == 2 and chi_l == chi_r, "Mismatch in circuit vs. Hamiltonian."
        chi = chi_l
        chi2 = chi * chi

        # ---- Vectorized precomputation over all B at once (no per-b Python loop) ----
        A0_sites = [ring[:, i, :, :, 0].to(ctype).contiguous() for i in range(N)]  # each (B, chi, chi)
        A1_sites = [ring[:, i, :, :, 1].to(ctype).contiguous() for i in range(N)]  # each (B, chi, chi)

        # Batched double-layer: E = A0* (x) A0 + A1* (x) A1, shape (B, chi^2, chi^2)
        def _batched_E(A0i, A1i):
            E0 = torch.einsum('bij,bkl->bikjl', A0i.conj(), A0i).reshape(B, chi2, chi2)
            E1 = torch.einsum('bij,bkl->bikjl', A1i.conj(), A1i).reshape(B, chi2, chi2)
            return E0 + E1

        # Build right suffix products: R_suf[i] = E_{i+1} @ ... @ E_{N-1}
        acc = torch.eye(chi2, dtype=ctype, device=device).unsqueeze(0).expand(B, -1, -1).clone()
        R_suf = [None] * N
        for i in range(N - 1, -1, -1):
            R_suf[i] = acc                                    # (B, chi^2, chi^2)
            Ei = _batched_E(A0_sites[i], A1_sites[i])
            acc = torch.bmm(Ei, acc)
        del acc

        # ---- Monte Carlo accumulation over shot-chunks
        totals = torch.zeros(B, dtype=torch.float64, device=device)
        done   = torch.zeros((), dtype=torch.int64, device=device)

        Ichi = torch.eye(chi, dtype=ctype, device=device)

        for s0 in range(0, shots, chunk_size):
            s1 = min(s0 + chunk_size, shots)
            S  = s1 - s0

            # X: (B,S,chi,chi), start as identity per (B,S)
            X = Ichi.expand(B, S, chi, chi).clone()
            # bits: (B,S,N) bool
            bits = torch.empty((B, S, N), dtype=torch.bool, device=device)

            # Sweep sites
            for i in range(N):
                A0i = A0_sites[i].unsqueeze(1)                # (B, 1, chi, chi)
                A1i = A1_sites[i].unsqueeze(1)                # (B, 1, chi, chi)

                M0 = torch.matmul(X, A0i)                     # (B, S, chi, chi)
                M1 = torch.matmul(X, A1i)

                # Matrix-form weights: w = v^dag R v  (no R4 5D tensors needed)
                Ri = R_suf[i]                                  # (B, chi^2, chi^2)
                v0 = M0.reshape(B, S, chi2)
                v1 = M1.reshape(B, S, chi2)
                y0 = torch.einsum('bij,bsj->bsi', Ri, v0)     # batched mat-vec
                y1 = torch.einsum('bij,bsj->bsi', Ri, v1)
                w0 = (v0.conj() * y0).sum(dim=-1).real         # (B, S)
                w1 = (v1.conj() * y1).sum(dim=-1).real         # (B, S)

                den = (w0 + w1).clamp_min(1e-300)
                p1  = w1 / den                                 # (B, S)

                u   = torch.rand((B, S), generator=gen, device=device)
                si  = (u < p1)                                 # True => choose 1, else 0
                bits[:, :, i] = si

                si_view = si.view(B, S, 1, 1)
                X = torch.where(si_view, M1, M0)               # select next prefix

                # Periodic normalization for numerical stability
                if normalize_every > 0 and (i % normalize_every) == 0 and i != 0:
                    nX = torch.linalg.norm(X.reshape(B, S, -1), dim=-1).clamp_min(1e-300).view(B, S, 1, 1)
                    X = X / nX

            # ---- Energy for this shot-chunk, streamed over term-chunks
            # bits: (B,S,N)
            bf = bits.to(torch.float32).reshape(B * S, N)  # (B*S, N)
            Eb = torch.zeros((B * S,), dtype=torch.float64, device=device)

            for t0 in range(0, T, term_chunk):
                t1   = min(t0 + term_chunk, T)
                Zblk = zmask[t0:t1, :]                     # (Tc, N) bool
                Cblk = coeffs[t0:t1]                       # (Tc,) float64

                cnt = bf @ Zblk.to(torch.float32).T        # (B*S, Tc)
                parity = (cnt.remainder_(2.0) > 0.5)       # True if odd number of Z -> -1
                sgn = torch.where(parity, -1.0, 1.0)       # (B*S, Tc)
                Eb += (sgn * Cblk.view(1, -1)).sum(dim=1)  # (B*S,)

            # Reduce samples -> per-B totals
            Eb = Eb.view(B, S)
            totals += Eb.sum(dim=1)
            done   += S

        out_parts.append((totals / done.clamp_min(1)).detach())  # (B,)

        # Free per-chunk buffers
        del ring, R_suf, A0_sites, A1_sites

    # Concatenate across parameter chunks and move to CPU
    return torch.cat(out_parts, dim=0).cpu()


@torch.no_grad()
def expectation_value_batch_correct_sampling(
    param_batch: Tensor,                 # (B, P)
    circuit,                             # .device and build_tensor_batch(params,B)->(B,n,chi,chi,2)
    hamiltonian,                         # .coefficients (len T), .get_bool_pauli_tensor()->(T,n)
    shots: int,                          # used (this is a sampling backend)
    *,
    shot_chunk: int = None,              # chunk over shots
    term_chunk: int = 4096,              # chunk over H terms
    seed: Optional[int] = None,
    use_fp32_env: bool = True,           # halve memory for environments
) -> Tensor:
    """
    Batched perfect-sampling expectation using right-suffix (R_suf) weights,
    memory-optimized (no 5D R4 tensors). Returns (B,) float64.
    """
    device = getattr(circuit, "device", param_batch.device)
    B = int(param_batch.shape[0])

    # ----- 1) Build cores -----
    cores = circuit.build_tensor_batch(param_batch, B).to(device)  # (B,n,chi,chi,2)
    if cores.dim() != 5 or cores.shape[-1] != 2:
        raise ValueError("circuit.build_tensor_batch must return (B, n, chi, chi, 2)")
    _, n, chi = cores.shape[0], cores.shape[1], cores.shape[2]
    cdtype = torch.complex64 if (use_fp32_env and cores.dtype in (torch.complex64, torch.complex128)) else cores.dtype
    rtype  = torch.float64  # final return dtype

    # Per-site slices (avoid python lists of huge tensors where possible)
    A0 = cores[..., 0].contiguous()   # (B, n, chi, chi)
    A1 = cores[..., 1].contiguous()   # (B, n, chi, chi)

    # ----- 2) Hamiltonian (streamed) -----
    coeffs = torch.as_tensor(hamiltonian.coefficients, dtype=rtype, device=device)  # (T,)
    zmask  = hamiltonian.get_bool_pauli_tensor().to(device=device, dtype=torch.bool)  # (T, n)
    if zmask.shape[1] != n:
        raise ValueError(f"Pauli mask width ({zmask.shape[1]}) != number of sites ({n})")
    T = int(coeffs.numel())

    # ----- 3) Build right suffix R_suf[i] in chi^2 x chi^2, complex (B, chi^2, chi^2); no R4 -----
    chi2 = chi * chi
    env_dtype = torch.complex64 if use_fp32_env else cdtype
    Id = torch.eye(chi2, dtype=env_dtype, device=device).expand(B, chi2, chi2).clone()

    # helper to form E(A) = sum_d A*(d) (x) A(d) = A0* (x) A0 + A1* (x) A1
    def E_from_slices(A0i: Tensor, A1i: Tensor) -> Tensor:
        # A0i/A1i: (B, chi, chi)
        E0 = torch.einsum('bij,bkl->bikjl', A0i.conj(), A0i).reshape(B, chi2, chi2)
        E1 = torch.einsum('bij,bkl->bikjl', A1i.conj(), A1i).reshape(B, chi2, chi2)
        return (E0 + E1).to(env_dtype)

    R_suf = [None] * n
    acc = Id
    # Right-to-left pass; store R_suf for each i
    for i in range(n - 1, -1, -1):
        R_suf[i] = acc
        Ei = E_from_slices(A0[:, i], A1[:, i])   # (B, chi2, chi2)
        acc = torch.bmm(Ei, acc)                 # (B, chi2, chi2)
        del Ei
    del acc, Id

    # ----- 4) Sampler over shots (in chunks) -----
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    if shot_chunk is None:
        shot_chunk = shots

    total = torch.zeros(B, dtype=rtype, device=device)
    done  = 0

    # Pre-allocate per-chunk work buffers to avoid reallocs
    eye_chi = torch.eye(chi, dtype=cdtype, device=device)
    for s0 in range(0, shots, shot_chunk):
        s1 = min(s0 + shot_chunk, shots)
        S  = s1 - s0

        X    = eye_chi.expand(B, S, chi, chi).clone()    # (B,S,chi,chi)
        bits = torch.empty((B, S, n), dtype=torch.bool, device=device)

        for i in range(n):
            A0i = A0[:, i].unsqueeze(1)                  # (B,1,chi,chi)
            A1i = A1[:, i].unsqueeze(1)                  # (B,1,chi,chi)

            M0 = torch.matmul(X, A0i)                    # (B,S,chi,chi)
            M1 = torch.matmul(X, A1i)                    # (B,S,chi,chi)

            # --- Memory-optimized weights via chi^2 bilinear form (NO R4) ---
            v0 = M0.reshape(B, S, chi2)                  # (B,S,chi^2)
            v1 = M1.reshape(B, S, chi2)

            Ri = R_suf[i]                                # (B,chi^2,chi^2), complex
            # y = R * v  without expanding R along S
            y0 = torch.einsum('bij,bsj->bsi', Ri, v0)    # (B,S,chi^2)
            y1 = torch.einsum('bij,bsj->bsi', Ri, v1)

            # w = v* . y
            w0 = (v0.conj() * y0).sum(dim=-1).real       # (B,S)
            w1 = (v1.conj() * y1).sum(dim=-1).real

            den = (w0 + w1).clamp_min(1e-300)
            p0  = (w0 / den).to(rtype)                   # (B,S)
            si  = (torch.rand((B, S), generator=gen, device=device) >= p0)
            bits[:, :, i] = si
            X = torch.where(si.unsqueeze(-1).unsqueeze(-1), M1, M0)

            del M0, M1, v0, v1, y0, y1  # free per-site temporaries

        # ---- 5) Streamed scoring over terms ----
        Eb = torch.zeros((B, S), dtype=rtype, device=device)
        bf = bits.to(torch.float32)
        for t0 in range(0, T, term_chunk):
            t1 = min(t0 + term_chunk, T)
            Zblk = zmask[t0:t1, :]                            # (Tc,n)
            Cblk = coeffs[t0:t1]                              # (Tc,)
            cnt  = torch.einsum('bsn,tn->bst', bf, Zblk.float())
            sgn  = torch.where((cnt.remainder_(2.0) > 0.5), -1.0, 1.0).to(rtype)
            Eb  += torch.einsum('bst,t->bs', sgn, Cblk)

        total += Eb.sum(dim=1)    # sum shots
        done  += S

        del X, bits, Eb, bf

    # free big envs
    del R_suf, A0, A1, cores

    return total / max(1, done)   # (B,), float64
