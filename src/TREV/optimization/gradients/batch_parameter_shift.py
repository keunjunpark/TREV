import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import torch.multiprocessing as mp

import torch
from torch import Tensor, no_grad

from TREV.optimization.gradients.set_batch_size import auto_batch_size

from ...circuit import Circuit
from ...hamiltonian.hamiltonian import Hamiltonian
from ...measure.enums import MeasureMethod
from ...optimization.gradients.gradient import Gradient
from TREV.measure.contraction import precompute_double_layer_and_right_suffix

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


def _get_gpu_count() -> int:
    """Return the number of available CUDA GPUs, or 0 if CUDA is unavailable."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _worker_gradient_chunk(
    ranges: list,
    base: Tensor,
    circuit_clone: Circuit,
    hamiltonian: Hamiltonian,
    shift: float,
    shots: int,
    measure_method: MeasureMethod,
    chunk_size: int,
) -> list:
    """Compute gradient slices for assigned parameter index ranges on one GPU.

    Each range is a (start, stop) tuple of parameter indices.
    Returns a list of (start, stop, grad_slice_cpu) tuples.
    """
    device = circuit_clone.device
    P = base.shape[1]
    base_dev = base.to(device)
    results = []

    for start, stop in ranges:
        C = stop - start
        idx = torch.arange(start, stop, device=device)
        arange_C = torch.arange(C, device=device)

        batch = base_dev.expand(2 * C, -1).clone()
        batch[arange_C, idx] += shift
        batch[C + arange_C, idx] -= shift

        exp_vals = _dispatch_expectation(batch, circuit_clone, hamiltonian, shots, measure_method)
        grad_slice = 0.5 * (exp_vals[:C] - exp_vals[C:])
        results.append((start, stop, grad_slice.cpu()))

    return results


def _mp_worker_fn(gpu_id, ranges, base_cpu, circuit_cpu, hamiltonian,
                  shift, shots, measure_method, chunk_size, result_queue):
    """Process worker for multi-GPU gradient computation.

    Each process owns one GPU, avoids GIL contention for sampling-heavy methods.
    """
    device = f'cuda:{gpu_id}'
    circuit_clone = circuit_cpu.to_device(device)
    base_dev = base_cpu.to(device)
    results = []

    for start, stop in ranges:
        C = stop - start
        idx = torch.arange(start, stop, device=device)
        arange_C = torch.arange(C, device=device)

        batch = base_dev.expand(2 * C, -1).clone()
        batch[arange_C, idx] += shift
        batch[C + arange_C, idx] -= shift

        exp_vals = _dispatch_expectation(batch, circuit_clone, hamiltonian, shots, measure_method)
        grad_slice = 0.5 * (exp_vals[:C] - exp_vals[C:])
        results.append((start, stop, grad_slice.cpu()))

    result_queue.put(results)


class BatchParameterShiftGradient(Gradient):
    def __init__(self, shift, batch_size, shots, measure_method: MeasureMethod, depth:int, is_partial:bool=False, num_gpus: int | None = None):
        super().__init__(measure_method)
        self.shift = shift
        self.batch_size = batch_size  # may be None
        self.shots = shots
        self.depth = depth
        self.curr_depth = 0
        self.is_partial = is_partial
        self._autotuned = False

        # Multi-GPU: auto-detect if None
        if num_gpus is None:
            self._num_gpus = _get_gpu_count()
        else:
            self._num_gpus = num_gpus

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
                gpu_msg = f"[TREV] Auto batch_size selected: {self.batch_size} " \
                          f"(measure={self.measure_method.name}, total_theta={P}, device={_gpu_info(device)})"
                if self._num_gpus > 1:
                    gpu_msg += f"\n[TREV] Multi-GPU enabled: {self._num_gpus} GPUs"
                print(gpu_msg + "\n", flush=True)

        val = batch_gradient(theta, circuit, hamiltonian, self.batch_size, self.shots,
                             self.shift, self.depth, self.curr_depth, self.is_partial, self.measure_method,
                             num_gpus=self._num_gpus)
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
        measure_method: MeasureMethod,
        num_gpus: int | None = None,
) -> torch.Tensor:
    """
    Memory-frugal parameter-shift gradient.

    params   : (P,)  -- single circuit's parameters
    chunk_size  : how many theta-indices to shift at once
    num_gpus : number of GPUs to use (None or <=1 for single-GPU)
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
        elif num_gpus is not None and num_gpus > 1 and P > 0:
            # --- Multi-GPU path (multiprocessing to avoid GIL) ---
            # Build chunk ranges
            all_ranges = []
            for start in range(0, P, chunk_size):
                stop = min(start + chunk_size, P)
                all_ranges.append((start, stop))

            # Round-robin assign chunks to GPUs
            gpu_ranges: dict[int, list] = {i: [] for i in range(num_gpus)}
            for idx_r, r in enumerate(all_ranges):
                gpu_ranges[idx_r % num_gpus].append(r)

            # CPU copies for safe cross-process sharing
            circuit_cpu = circuit.to_device('cpu')
            base_cpu = base.cpu()

            # Launch worker processes (spawn context for CUDA safety)
            ctx = mp.get_context('spawn')
            result_queue = ctx.Queue()
            processes = []
            for gpu_id in range(num_gpus):
                if not gpu_ranges[gpu_id]:
                    continue
                p = ctx.Process(
                    target=_mp_worker_fn,
                    args=(gpu_id, gpu_ranges[gpu_id], base_cpu, circuit_cpu,
                          hamiltonian, shift, shots, measure_method,
                          chunk_size, result_queue),
                )
                p.start()
                processes.append(p)

            # Collect results
            for _ in processes:
                results = result_queue.get()
                for start, stop, grad_slice_cpu in results:
                    grad[start:stop] = grad_slice_cpu.to(device)

            for p in processes:
                p.join()
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



def _kron_contract_right(Prod, A0, A1, sign=1):
    """Contract Prod @ E(site) using Kronecker decomposition.

    E = conj(A0)⊗A0 + sign*conj(A1)⊗A1
    sign=+1 → E_I,  sign=-1 → E_Z.

    Prod: (B, ..., l_bra, l_ket, r_bra, r_ket)  -- last 4 dims are spatial
    A0, A1: (B, chi, chi)

    Contracts r_bra/r_ket (last 2 dims of Prod) and produces new right indices.
    A must broadcast over all dims between B and r_bra/r_ket (i.e., middle + l_bra + l_ket).
    """
    # Number of dims to broadcast over: everything between B (dim 0) and r_bra/r_ket (last 2)
    n_broadcast = Prod.dim() - 3  # = n_middle + l_bra + l_ket
    slices = (slice(None),) + (None,) * n_broadcast + (slice(None), slice(None))
    A0_e  = A0[slices]                          # (B, 1..., chi, chi)
    A0H_e = A0.conj().mT[slices]
    A1_e  = A1[slices]
    A1H_e = A1.conj().mT[slices]

    r0 = torch.matmul(A0H_e, torch.matmul(Prod, A0_e))
    r1 = torch.matmul(A1H_e, torch.matmul(Prod, A1_e))
    return r0 + sign * r1


def _make_eye4(B, chi, ctype, device):
    """Build the 4-D identity: eye4[b, i, j, i, j] = 1 for all i, j.

    Corresponds to the chi^2 x chi^2 identity matrix in reshaped form.
    """
    eye2d = torch.eye(chi * chi, dtype=ctype, device=device)  # (chi^2, chi^2)
    return eye2d.reshape(chi, chi, chi, chi).unsqueeze(0).expand(B, -1, -1, -1, -1)


@torch.no_grad()
def expectation_value_batch_efficient_contraction(
    param_batch: Tensor,                 # (B, P)
    circuit,                             # .device and build_tensor_batch(params,B)->(B,N,l,r,2)
    hamiltonian,                         # .get_bool_pauli_tensor()->(T,N); .coefficients (len T)
    shots: int,                          # kept for API compatibility (ignored)
    *,
    term_chunk: Optional[int] = None,    # unused, kept for API compat
    cache_double_layers: bool = True,    # cache A0/A1 per site
    param_chunk: Optional[int] = None,   # split B across chunks to fit memory
    use_complex64: bool = True,          # internal complex precision
) -> Tensor:
    """
    Exact batched <psi|H|psi> via identity-chain factored Kronecker contraction.

    Precomputes all-identity left-prefix and right-suffix products, then
    contracts only at the sparse Z-sites of each Hamiltonian term.

    Complexity: O((N + k*T) * B * chi^5)  instead of  O(N * T * B * chi^5)
    Memory:     O(N * B * chi^4)  instead of  O(T * B * chi^4)

    where k is the average number of Z-operators per Hamiltonian term.
    """
    device = getattr(circuit, "device", param_batch.device)

    B_total = int(param_batch.shape[0])
    if param_chunk is None or param_chunk >= B_total:
        batch_slices = [(0, B_total)]
    else:
        batch_slices = [(s, min(s + param_chunk, B_total)) for s in range(0, B_total, param_chunk)]

    # Hamiltonian
    paulis = hamiltonian.get_bool_pauli_tensor().to(device)  # (T, N) bool
    coeffs = torch.as_tensor(
        hamiltonian.coefficients,
        dtype=torch.cfloat if use_complex64 else torch.cdouble,
        device=device,
    )
    T, N = paulis.shape

    out_parts = []

    for lo, hi in batch_slices:
        param_view = param_batch[lo:hi]
        B = int(param_view.shape[0])

        # Build cores
        ring = circuit.build_tensor_batch(param_view, B).to(device)  # (B,N,l,r,2)
        _, N_check, l, r, d = ring.shape
        assert N_check == N and d == 2, "MPS/circuit shape mismatch with Hamiltonian"
        ctype = torch.complex64 if use_complex64 else torch.complex128
        chi = l

        # Cache per-site A0/A1 slices
        sites = []
        for i in range(N):
            Ab = ring[:, i].to(ctype)
            sites.append((Ab[:, :, :, 0].contiguous(), Ab[:, :, :, 1].contiguous()))
        del ring

        eye4 = _make_eye4(B, chi, ctype, device)

        # --- Precompute left prefix products under all-identity ---
        # L_pre[i] = E_I(0) @ E_I(1) @ ... @ E_I(i-1),  L_pre[0] = I
        # Uses right contraction: acc = acc @ E_I(i)
        L_pre = [None] * (N + 1)
        acc = eye4
        for i in range(N):
            L_pre[i] = acc
            A0_i, A1_i = sites[i]
            acc = _kron_contract_right(acc, A0_i, A1_i)
        L_pre[N] = acc
        del acc

        # --- Precompute TRANSPOSED right suffix products under all-identity ---
        # R_suf_T[i] = (E_I(i) @ ... @ E_I(N-1))^T = E_I(N-1)^T @ ... @ E_I(i)^T
        # R_suf_T[N] = I
        # Since E_I^T uses A^T instead of A, we pass A.mT to the right-contraction.
        # Trace formula: Tr(run @ R_suf[i]) = (run * R_suf_T[i]).sum(dims 1..4)
        R_suf_T = [None] * (N + 1)
        acc = eye4
        for i in range(N - 1, -1, -1):
            A0_i, A1_i = sites[i]
            acc = _kron_contract_right(acc, A0_i.mT, A1_i.mT)
            R_suf_T[i] = acc
        R_suf_T[N] = eye4
        del acc, eye4

        # --- Per-term contraction: only at Z-sites ---
        totals = torch.zeros(B, dtype=ctype, device=device)

        for t in range(T):
            z_sites = torch.where(paulis[t])[0].tolist()

            if len(z_sites) == 0:
                # All identity: Tr(full ring) = (L_pre[N] * R_suf_T[N]).sum
                totals += coeffs[t] * (L_pre[N] * R_suf_T[N]).sum(dim=(1, 2, 3, 4))
                continue

            s_first = z_sites[0]
            s_last = z_sites[-1]

            # Start from the precomputed left prefix up to the first Z-site
            run = L_pre[s_first].clone()

            # Contract through sites s_first..s_last (Z or I as needed)
            for i in range(s_first, s_last + 1):
                A0_i, A1_i = sites[i]
                if paulis[t, i]:
                    run = _kron_contract_right(run, A0_i, A1_i, sign=-1)
                else:
                    run = _kron_contract_right(run, A0_i, A1_i)

            # Tr(run @ R_suf[s_last+1]) = element-wise product with transposed suffix
            totals += coeffs[t] * (run * R_suf_T[s_last + 1]).sum(dim=(1, 2, 3, 4))

        out_parts.append(totals.real.float())

        del sites, L_pre, R_suf_T, totals

    return torch.cat(out_parts, dim=0)


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

        # ---- Batched R_suf via Kronecker-free O(χ⁵) contraction
        # Computes kron(A_d, conj(A_d)) @ acc as two matmuls per d,
        # avoiding O(χ⁶) explicit E construction. Same kron convention as
        # precompute_double_layer_and_right_suffix.
        chi2 = chi * chi
        A0_sites = [ring[:, i, :, :, 0].to(ctype).contiguous() for i in range(N)]
        A1_sites = [ring[:, i, :, :, 1].to(ctype).contiguous() for i in range(N)]
        del ring

        acc = torch.eye(chi2, dtype=ctype, device=device).unsqueeze(0).expand(B, -1, -1).contiguous()
        R_suf = [None] * N
        for i in range(N - 1, -1, -1):
            R_suf[i] = acc
            A0i, A1i = A0_sites[i], A1_sites[i]
            acc_view = acc.view(B, chi, chi, chi2)
            # kron(A0, conj(A0)) @ acc
            temp = torch.matmul(A0i.conj().unsqueeze(1), acc_view)
            new_acc = torch.matmul(A0i, temp.reshape(B, chi, chi * chi2))
            del temp
            # kron(A1, conj(A1)) @ acc
            temp = torch.matmul(A1i.conj().unsqueeze(1), acc_view)
            new_acc += torch.matmul(A1i, temp.reshape(B, chi, chi * chi2))
            del temp
            acc = new_acc.view(B, chi, chi, chi2).reshape(B, chi2, chi2).contiguous()
            del new_acc
        del acc

        # Convert R_suf from kron convention to bilinear form for matmul weights
        # kron: E[ij,kl] = A[i,k]*conj(A[j,l])
        # bilinear: R_bl[a·χ+b, c·χ+d] = conj(R_kron[b·χ+d, a·χ+c])
        for i in range(N):
            R_suf[i] = (R_suf[i].view(B, chi, chi, chi, chi)
                        .permute(0, 3, 1, 4, 2)
                        .conj()
                        .contiguous()
                        .reshape(B, chi2, chi2))

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
                Ri  = R_suf[i]                       # (B, χ², χ²)

                M0 = torch.matmul(X, A0i.unsqueeze(1))  # (B,S,χ,χ)
                M1 = torch.matmul(X, A1i.unsqueeze(1))  # (B,S,χ,χ)

                # Weights via bilinear form: w_d = vec(M_d)† @ R_suf @ vec(M_d)
                v0 = M0.reshape(B, S, chi2)
                v1 = M1.reshape(B, S, chi2)
                y0 = torch.matmul(v0, Ri.mT)        # (B,S,χ²)
                y1 = torch.matmul(v1, Ri.mT)        # (B,S,χ²)
                w0 = (v0.conj() * y0).sum(-1).real   # (B,S)
                w1 = (v1.conj() * y1).sum(-1).real   # (B,S)
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
        del R_suf, A0_sites, A1_sites
        torch.cuda.empty_cache()

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
