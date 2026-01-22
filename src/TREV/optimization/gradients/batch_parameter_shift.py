import time
from typing import List, Optional

# # .../measure/efficient_contraction.py
# from __future__ import annotations
from typing import Optional


import torch
from torch import Tensor, no_grad

from TREV.measure.contraction import precompute_double_layer_and_right_suffix

from ...circuit import Circuit
from ...hamiltonian.hamiltonian import Hamiltonian
from ...measure.enums import MeasureMethod
from ...optimization.gradients.gradient import Gradient


class BatchParameterShiftGradient(Gradient):
    def __init__(self, shift, batch_size, shots, measure_method: MeasureMethod, depth:int, is_partial:bool = False):
        super().__init__(measure_method)
        self.shift = shift
        self.batch_size = batch_size
        self.shots = shots
        self.depth = depth
        self.curr_depth = 0
        self.is_partial = is_partial
    def run(self, theta: torch.Tensor, circuit: Circuit, hamiltonian: Hamiltonian):
        val = batch_gradient(theta, circuit, hamiltonian, self.batch_size, self.shots, self.shift, self.depth, self.curr_depth, self.is_partial, self.measure_method)
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
    Memory‑frugal parameter‑shift gradient.

    params   : (P,)  – single circuit’s parameters
    chunk_size  : how many θ‑indices to shift at once
    *circuit_*  : whatever run_circuit_batched needs
    returns     : (P,)  – gradient d⟨O⟩/dθ
    """
    with torch.no_grad():
        device = circuit.device
        P      = params.numel()
        grad   = torch.empty(P, device=device, dtype=torch.float32)
        base   = params.unsqueeze(0)            # (1, P)  acts as “B = 1”

        if is_partial:
            dP = int(P/depth)
            start = dP*curr_depth
            stop = min(start + dP, P)

            idx = torch.arange(start, stop, device=device)

            eye = torch.eye(len(idx), device=device) * shift  # (C,C)
            plus = base.repeat(len(idx), 1).to(device)  # (C,P)
            minus = plus.clone().to(device)

            plus[torch.arange(len(idx)), idx] += shift
            minus[torch.arange(len(idx)), idx] -= shift
            batch = torch.cat([plus, minus], dim=0).to(device)  # (2C,P)
            if measure_method == MeasureMethod.EFFICIENT_CONTRACTION:
                exp_vals = expectation_value_batch_efficient_contraction(batch, circuit, hamiltonian, shots)
            elif measure_method == MeasureMethod.RIGHT_SUFFIX_SAMPLING:
                exp_vals = expectation_value_batch_right_suffix(batch, circuit, hamiltonian, shots)
            else:
                exp_vals = expectation_value_batch(batch, circuit, hamiltonian, shots)
            fwd, bwd = exp_vals[:len(idx)], exp_vals[len(idx):]

            grad[start:stop] = 0.5 * (fwd - bwd)
        else:
            for start in range(0, P, chunk_size):
                stop   = min(start + chunk_size, P)
                idx    = torch.arange(start, stop, device=device)
                plus   = base.repeat(len(idx), 1).to(device)                     # (C,P)
                minus  = plus.clone().to(device)

                plus [torch.arange(len(idx)), idx] += shift
                minus[torch.arange(len(idx)), idx] -= shift
                batch  = torch.cat([plus, minus], dim=0).to(device)                 # (2C,P)

                if measure_method == MeasureMethod.EFFICIENT_CONTRACTION:
                    exp_vals = expectation_value_batch_efficient_contraction(batch, circuit, hamiltonian, shots)
                elif measure_method == MeasureMethod.RIGHT_SUFFIX_SAMPLING:
                    
                    exp_vals = expectation_value_batch_right_suffix(batch, circuit, hamiltonian, shots)
                else:
                    exp_vals = expectation_value_batch(batch, circuit, hamiltonian, shots)
                fwd, bwd = exp_vals[:len(idx)], exp_vals[len(idx):]

                grad[start:stop] = 0.5 * (fwd - bwd)
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
        if circuit.device == 'cuda':
            torch.cuda.synchronize()

        batch_coefs = (
            torch.tensor(hamiltonian.coefficients, dtype=torch.complex64, device=circuit.device)
            .unsqueeze(0).unsqueeze(0).expand(B, shots, -1).clone()
        )

        ring_tensor_batch = circuit.build_tensor_batch(param_batch, B)
        B, N = ring_tensor_batch.shape[:2]
        paulis_tensor = hamiltonian.get_bool_pauli_tensor_old().to(device=circuit.device)

        q0 = torch.tensor([[1], [0]], dtype=torch.cfloat, device=circuit.device)
        q1 = torch.tensor([[0], [1]], dtype=torch.cfloat, device=circuit.device)

        assert all(len(p) >= N for p in paulis_tensor), \
            f"Each Pauli string must have at least {N} qubits!"

        batch_prev: torch.Tensor | None = None

        for i in range(N):
            curr_tens = ring_tensor_batch[:, i].contiguous()

            if i == 0:
                qubit_0 = torch.einsum('bijk,kl->bijl', curr_tens, q0).squeeze(-1)
                qubit_1 = torch.einsum('bijk,kl->bijl', curr_tens, q1).squeeze(-1)
                batch_qubit_0 = qubit_0.unsqueeze(1).expand(-1, shots, -1, -1).contiguous()
                batch_qubit_1 = qubit_1.unsqueeze(1).expand(-1, shots, -1, -1).contiguous()
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
            rnd = torch.rand(B, shots, device=circuit.device)
            choose_1 = rnd > p0

            batch_prev = torch.where(
                choose_1.unsqueeze(-1).unsqueeze(-1),
                batch_qubit_1,
                batch_qubit_0
            )

            mask = paulis_tensor[i].unsqueeze(0).unsqueeze(1).expand(B, shots, -1)
            flip_mask = mask & choose_1.unsqueeze(-1)
            batch_coefs = torch.where(flip_mask, -batch_coefs, batch_coefs)

        expectations = batch_coefs.sum(dim=2).mean(dim=1)
        return expectations.detach().real.float()

    

@torch.no_grad()
def expectation_value_batch_efficient_contraction(
    param_batch: Tensor,                 # (B, P)
    circuit,                             # .device and build_tensor_batch(params,B)->(B,N,l,r,2)
    hamiltonian,                         # .get_bool_pauli_tensor()->(T,N); .coefficients (len T)
    shots: int,                          # kept for API compatibility (ignored)
    *,
    term_chunk: Optional[int] = None,    # chunk across Hamiltonian terms
    cache_double_layers: bool = False,   # True = cache E_I/E_Z per site (uses more memory)
    param_chunk: Optional[int] = None,   # split B across chunks to fit memory
    use_complex64: bool = True,          # internal complex precision
) -> Tensor:
    """
    Exact batched ⟨ψ|H|ψ⟩ via STREAMED double-layer contraction in MATRIX form.

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
        batch_slices = []
        for s in range(0, B_total, param_chunk):
            batch_slices.append((s, min(s + param_chunk, B_total)))

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
            Ab: (B,el,r,2)
            Returns E_I, E_Z each (B, ll, rr) where rows index (l,l'), cols index (r,r').
            We compute 5D then reshap immediately, NO caching of 5D tensors.
            """
            AO_I = torch.einsum('blrd,dk->blrk', Ab, I)  # (B,l,r,2)
            AO_Z = torch.einsum('blrd,dk->blrk', Ab, Z)  # (B,l,r,2)
            # Ei_5d(b,l,L,r,R) = sum_d Ab*[b,l,r,d] * AO_[b,L,R,d]
            EI5 = torch.einsum('blrd,bLRd->blLrR', Ab.conj(), AO_I)
            EZ5 = torch.einsum('blrd,bLRd->blLrR', Ab.conj(), AO_Z)
            EI  = EI5.reshape(B, ll, rr).contiguous()
            EZ  = EZ5.reshape(B, ll, rr).contiguous()
            return EI, EZ

        # Optional cache of per-site matrices to avoid recompute across term-chunks
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

            # Initialize transfer product for each (b,t): (B,Tc,ll,rr)
            # We’ll multiply Ei on the RIGHT: Prod = Ei0 @ Ei1 @ ... @ Ei_{N-1}
            # Final value = trace(Prod) for ring closure.
            # Start with site 0:
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

                # Prod(b,t,ll,rr) @ Ei(b,t,rr,rr_next) -> here rr_next == rr by construction
                # Use batched matmul by fusing (B,Tc)
                Prod = torch.matmul(
                    Prod.reshape(B * Tc, ll, rr),
                    Ei.reshape(B * Tc, rr, rr),
                ).reshape(B, Tc, ll, rr).contiguous()

            # Ring closure: trace over (ll,rr) diagonal of product matrix
            # trace = einsum over diagonal; better: use .diagonal then sum
            trace_vals = Prod.diagonal(offset=0, dim1=2, dim2=3).sum(dim=-1)  # (B,Tc)

            totals += (trace_vals * coefs.view(1, Tc)).sum(dim=1)

            # free working buffers of this term-chunk
            del Prod, trace_vals, m0
            if not cache_double_layers:
                # if NOT caching, the per-site E_i matrices were freed each iteration automatically
                pass
            torch.cuda.empty_cache()

        out_parts.append(totals.real.float())  # (B,)

        # free per param-chunk tensors
        del ring, totals, cached
        torch.cuda.empty_cache()

    # concat across param chunks
    return torch.cat(out_parts, dim=0)

import torch
from typing import Optional

@torch.no_grad()
def expectation_value_batch_right_suffix(
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
        ring = circuit.build_tensor_batch(param_view, B)     # (B, N, χ, χ, 2)
        _, N_chk, chi_l, chi_r, d = ring.shape
        assert N_chk == N and d == 2 and chi_l == chi_r, "Mismatch in circuit vs. Hamiltonian."
        chi = chi_l

        # ---- Precompute right-suffix objects per-parameter, stack across B
        # We only need the *R4* suffix tensors for sampling:
        #   R4[i] shape (χ,χ,χ,χ) per parameter -> stack to (B,χ,χ,χ,χ) per site.
        # We also prepare A0/A1 per site stacked across B.
        R4_stack = []
        A0_stack = []
        A1_stack = []
        # precompute_double_layer_and_right_suffix expects a single-(N,χ,χ,2) ring per parameter
        for b in range(B):
            Es, R_suf, d2, _, _ = precompute_double_layer_and_right_suffix(ring[b])
            # Cast once to consistent dtype and layout
            R4_b = [Ri.to(ctype).view(chi, chi, chi, chi).permute(2, 3, 0, 1).contiguous()
                    for Ri in R_suf]  # -> (χ,χ,χ,χ) with indices (a,c,b,d) order used below
            A0_b = [ring[b, i, :, :, 0].to(ctype).contiguous() for i in range(N)]  # (χ,χ)
            A1_b = [ring[b, i, :, :, 1].to(ctype).contiguous() for i in range(N)]  # (χ,χ)
            R4_stack.append(R4_b)
            A0_stack.append(A0_b)
            A1_stack.append(A1_b)

        # Now stack across B for each site i -> tensors:
        #   R4_sites[i] : (B, χ,χ,χ,χ); A0_sites[i]/A1_sites[i] : (B, χ,χ)
        R4_sites = [torch.stack([R4_stack[b][i] for b in range(B)], dim=0) for i in range(N)]
        A0_sites = [torch.stack([A0_stack[b][i] for b in range(B)], dim=0) for i in range(N)]
        A1_sites = [torch.stack([A1_stack[b][i] for b in range(B)], dim=0) for i in range(N)]

        # ---- Monte Carlo accumulation over shot-chunks
        totals = torch.zeros(B, dtype=torch.float64, device=device)
        done   = torch.zeros((), dtype=torch.int64, device=device)

        Ichi = torch.eye(chi, dtype=ctype, device=device)

        for s0 in range(0, shots, chunk_size):
            s1 = min(s0 + chunk_size, shots)
            S  = s1 - s0

            # X: (B,S,χ,χ), start as identity per (B,S)
            X = Ichi.expand(B, S, chi, chi).clone()
            # bits: (B,S,N) bool
            bits = torch.empty((B, S, N), dtype=torch.bool, device=device)

            # Sweep sites
            for i in range(N):
                A0i = A0_sites[i]                    # (B, χ, χ)
                A1i = A1_sites[i]                    # (B, χ, χ)
                R4i = R4_sites[i]                    # (B, χ, χ, χ, χ)

                # Broadcast A* to (B,S,χ,χ) for batched matmul
                A0i_bs = A0i.unsqueeze(1)            # (B,1,χ,χ)
                A1i_bs = A1i.unsqueeze(1)            # (B,1,χ,χ)

                M0 = torch.matmul(X, A0i_bs)         # (B,S,χ,χ)
                M1 = torch.matmul(X, A1i_bs)         # (B,S,χ,χ)

                # Weights w0, w1 ∝ ⟨Mσ| R4 |Mσ⟩  (σ in {0,1}) — keep real part for probs
                # Indices: M0 -> (B,S,a,b); M0.conj -> (B,S,c,d); R4 -> (B,a,c,b,d)  => (B,S)
                # w0 = torch.einsum('bsab,bscd,bacbd->bs', M0, M0.conj(), R4i).real
                # w1 = torch.einsum('bsab,bscd,bacbd->bs', M1, M1.conj(), R4i).real
                w0 = torch.einsum('xsab,xscd,xacbd->xs', M0, M0.conj(), R4i).real
                w1 = torch.einsum('xsab,xscd,xacbd->xs', M1, M1.conj(), R4i).real
                den = (w0 + w1).clamp_min(1e-300)
                p1  = (w1 / den)                      # (B,S)

                u   = torch.rand((B, S), generator=gen, device=device)
                si  = (u < p1)                        # True => choose 1, else 0
                bits[:, :, i] = si

                si_view = si.view(B, S, 1, 1)
                X = torch.where(si_view, M1, M0)      # select next prefix

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
        del ring, R4_stack, A0_stack, A1_stack, R4_sites, A0_sites, A1_sites
        torch.cuda.empty_cache()

    # Concatenate across parameter chunks and move to CPU
    return torch.cat(out_parts, dim=0).cpu()



import torch

def _maybe_compile(fn):
    try:
        # cheap compile setting; avoids the heavy max-autotune path
        return torch.compile(fn, mode="reduce-overhead", fullgraph=False, dynamic=False)
    except Exception:
        return fn

def _site_sweep_impl(X, A0, A1, R4, U, normalize_every: int):
    """
    X: (S, χ, χ)  complex64
    A0/A1: (N, χ, χ) complex64
    R4: list of length N, each (χ, χ, χ, χ) complex64
    U: (S, N) float32 uniforms in [0,1)
    returns:
        bits: (S, N) bool
        X:    (S, χ, χ) complex64 (final state after N)
    """
    S, chi, _ = X.shape
    N = A0.shape[0]
    bits = torch.empty((S, N), dtype=torch.bool, device=X.device)

    for i in range(N):
        A0i = A0[i]
        A1i = A1[i]
        R4i = R4[i]

        M0 = X @ A0i                  # (S, χ, χ)
        M1 = X @ A1i

        # one einsum for both branches
        M  = torch.stack((M0, M1), dim=1)  # (S, 2, χ, χ)
        w  = torch.einsum('skab,skcd,acbd->sk', M, M.conj(), R4i).real  # (S,2)
        den = (w.sum(dim=1)).clamp_min_(1e-30)
        p0  = (w[:, 0] / den)

        si  = (U[:, i] >= p0)         # (S,) bool
        bits[:, i] = si
        X = torch.where(si.view(S,1,1), M1, M0)

        if (i % normalize_every) == 0 and i != 0:
            nX = torch.linalg.norm(X.reshape(S, -1), dim=1).clamp_min_(1e-30).view(S,1,1)
            X  = X / nX

    return bits, X


import torch
from torch import Tensor
from typing import Optional

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

    # ----- 3) Build right suffix R_suf[i] in χ²×χ², complex (B, χ², χ²); no R4 -----
    chi2 = chi * chi
    env_dtype = torch.complex64 if use_fp32_env else cdtype
    Id = torch.eye(chi2, dtype=env_dtype, device=device).expand(B, chi2, chi2).clone()

    # helper to form E(A) = Σ_d A*(d) ⊗ A(d) = A0*⊗A0 + A1*⊗A1
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
    torch.cuda.empty_cache()

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

            # --- Memory-optimized weights via χ² bilinear form (NO R4) ---
            # v = vec(M) with row-major reshape consistent with E construction
            v0 = M0.reshape(B, S, chi2)                  # (B,S,chi^2)
            v1 = M1.reshape(B, S, chi2)

            Ri = R_suf[i]                                # (B,chi^2,chi^2), complex
            # y = R * v  without expanding R along S
            y0 = torch.einsum('bij,bsj->bsi', Ri, v0)    # (B,S,chi^2)
            y1 = torch.einsum('bij,bsj->bsi', Ri, v1)

            # w = v* · y
            w0 = (v0.conj() * y0).sum(dim=-1).real       # (B,S)
            w1 = (v1.conj() * y1).sum(dim=-1).real

            den = (w0 + w1).clamp_min(1e-300)
            p0  = (w0 / den).to(rtype)                   # (B,S)
            si  = (torch.rand((B, S), generator=gen, device=device) >= p0)
            bits[:, :, i] = si
            X = torch.where(si.unsqueeze(-1).unsqueeze(-1), M1, M0)

            # Optional per-site stabilization (cheap; keeps memory flat)
            # nX = torch.linalg.norm(X.reshape(B, S, -1), dim=2).clamp_min(1e-300).view(B, S, 1, 1)
            # X = X / nX

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
        torch.cuda.empty_cache()

    # free big envs
    del R_suf, A0, A1, cores
    torch.cuda.empty_cache()

    return total / max(1, done)   # (B,), float64