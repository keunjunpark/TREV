import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import torch.multiprocessing as mp

import torch
from torch import Tensor, no_grad

from TREV.optimization.gradients.set_batch_size import auto_batch_size

from ...circuit import Circuit
from ...hamiltonian.hamiltonian import Hamiltonian, rotate_tensor_for_measurement
from ...measure.enums import MeasureMethod
from ...optimization.gradients.gradient import Gradient
from TREV.measure.contraction import precompute_double_layer_and_right_suffix

def _validate_zi_only(hamiltonian, method_name: str):
    """Raise if Hamiltonian contains X or Y terms (sampling can't evaluate them)."""
    if not hamiltonian.has_only_zi:
        raise ValueError(
            f"{method_name} only supports Z/I Hamiltonians. "
            "Use EFFICIENT_CONTRACTION for Hamiltonians with X/Y terms."
        )


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


def _dispatch_expectation(param_batch, circuit, hamiltonian, shots, measure_method, ring_tensor=None):
    """Route to the correct batched expectation value backend."""
    if measure_method == MeasureMethod.EFFICIENT_CONTRACTION:
        return expectation_value_batch_efficient_contraction(param_batch, circuit, hamiltonian, shots, ring_tensor=ring_tensor)
    elif measure_method == MeasureMethod.RIGHT_SUFFIX_SAMPLING:
        return expectation_value_batch_right_suffix(param_batch, circuit, hamiltonian, shots, ring_tensor=ring_tensor)
    elif measure_method == MeasureMethod.PERFECT_SAMPLING:
        return expectation_value_batch(param_batch, circuit, hamiltonian, shots, ring_tensor=ring_tensor)
    else:
        return expectation_value_batch(param_batch, circuit, hamiltonian, shots, ring_tensor=ring_tensor)


def _get_gpu_count() -> int:
    """Return the number of available CUDA GPUs, or 0 if CUDA is unavailable."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _distribute_params_evenly(P, num_gpus, chunk_size):
    """Divide P parameters evenly across num_gpus, each split into chunks.

    Returns dict mapping gpu_id -> list of (start, stop) tuples.
    """
    base_per_gpu = P // num_gpus
    remainder = P % num_gpus
    gpu_ranges = {}
    offset = 0
    for gpu_id in range(num_gpus):
        count = base_per_gpu + (1 if gpu_id < remainder else 0)
        if count == 0:
            gpu_ranges[gpu_id] = []
            continue
        gpu_end = offset + count
        ranges = []
        for s in range(offset, gpu_end, chunk_size):
            ranges.append((s, min(s + chunk_size, gpu_end)))
        gpu_ranges[gpu_id] = ranges
        offset = gpu_end
    return gpu_ranges


def _mp_worker_fn(gpu_id, ranges, base_cpu, circuit_cpu, hamiltonian, shift, shots,
                  measure_method, chunk_size, grad_shared):
    """Multiprocessing worker: compute gradient slices on assigned GPU."""
    import traceback
    device = f'cuda:{gpu_id}'
    circuit_clone = circuit_cpu.to_device(device)
    base_dev = base_cpu.to(device)

    try:
        for start, stop in ranges:
            C = stop - start
            idx = torch.arange(start, stop, device=device)
            arange_C = torch.arange(C, device=device)

            batch = base_dev.expand(2 * C, -1).clone()
            batch[arange_C, idx] += shift
            batch[C + arange_C, idx] -= shift

            exp_vals = _dispatch_expectation(batch, circuit_clone, hamiltonian, shots, measure_method)
            grad_slice = 0.5 * (exp_vals[:C] - exp_vals[C:])
            grad_shared[start:stop] = grad_slice.cpu()
    except Exception as e:
        print(f"[TREV] GPU {gpu_id} worker FAILED: {e}", flush=True)
        traceback.print_exc()
    finally:
        del circuit_clone, base_dev
        torch.cuda.empty_cache()


def _persistent_worker_fn(gpu_id, circuit_cpu, hamiltonian, shift, shots,
                          measure_method, chunk_size,
                          base_shared, grad_shared, ranges_queue, done_barrier,
                          shutdown_event):
    """Persistent process worker — stays alive across iterations.

    Waits for work on ranges_queue, computes, writes to grad_shared,
    then syncs at done_barrier. Loops until shutdown_event is set.
    """
    import traceback
    device = f'cuda:{gpu_id}'
    circuit_clone = circuit_cpu.to_device(device)

    try:
        while not shutdown_event.is_set():
            # Wait for work (ranges list) or shutdown sentinel
            try:
                msg = ranges_queue.get(timeout=1.0)
            except Exception:
                continue
            if msg is None:  # shutdown sentinel
                break

            ranges = msg
            try:
                P = base_shared.shape[0]
                base_dev = base_shared.to(device).unsqueeze(0)  # (1, P)

                for start, stop in ranges:
                    C = stop - start
                    idx = torch.arange(start, stop, device=device)
                    arange_C = torch.arange(C, device=device)

                    batch = base_dev.expand(2 * C, -1).clone()
                    batch[arange_C, idx] += shift
                    batch[C + arange_C, idx] -= shift

                    exp_vals = _dispatch_expectation(
                        batch, circuit_clone, hamiltonian, shots, measure_method)
                    grad_slice = 0.5 * (exp_vals[:C] - exp_vals[C:])
                    grad_shared[start:stop] = grad_slice.cpu()
            except Exception as e:
                print(f"[TREV] GPU {gpu_id} worker FAILED: {e}", flush=True)
                traceback.print_exc()

            # Release cached GPU memory before going idle
            torch.cuda.empty_cache()

            # Signal this worker is done for this iteration
            done_barrier.wait()
    finally:
        # Cleanup GPU memory on exit
        del circuit_clone
        torch.cuda.empty_cache()


class _MultiGPUPool:
    """Persistent pool of worker processes for multi-GPU gradient computation."""

    _active_pool = None  # class-level singleton — only one pool at a time

    def __init__(self, num_gpus, circuit, hamiltonian, shift, shots,
                 measure_method, chunk_size, P):
        # Kill any previous pool first (e.g. from a different gradient object)
        if _MultiGPUPool._active_pool is not None:
            print("[TREV] Shutting down previous multi-GPU pool", flush=True)
            _MultiGPUPool._active_pool.shutdown()
        _MultiGPUPool._active_pool = self

        ctx = mp.get_context('spawn')

        circuit_cpu = circuit.to_device('cpu')
        self.num_gpus = num_gpus
        self.P = P

        # Shared-memory tensors for data exchange (no Queue serialization)
        self.base_shared = torch.zeros(P, dtype=torch.float32).share_memory_()
        self.grad_shared = torch.zeros(P, dtype=torch.float32).share_memory_()

        # Per-worker queue for sending ranges
        self.ranges_queues = [ctx.Queue() for _ in range(num_gpus)]
        # Barrier: num_gpus workers + 1 main thread
        self.done_barrier = ctx.Barrier(num_gpus + 1)
        self.shutdown_event = ctx.Event()

        self.processes = []
        for gpu_id in range(num_gpus):
            p = ctx.Process(
                target=_persistent_worker_fn,
                args=(gpu_id, circuit_cpu, hamiltonian, shift, shots,
                      measure_method, chunk_size,
                      self.base_shared, self.grad_shared,
                      self.ranges_queues[gpu_id], self.done_barrier,
                      self.shutdown_event),
            )
            p.daemon = True
            p.start()
            self.processes.append(p)
        print(f"[TREV] Persistent multi-GPU pool started: {num_gpus} workers", flush=True)

    def compute_gradient(self, params, chunk_size, device):
        P = params.numel()

        # Update shared base tensor
        self.base_shared[:P] = params.detach().cpu()

        # Free main-process GPU cache so GPU 0 worker has room
        torch.cuda.empty_cache()

        # Distribute parameters evenly across GPUs
        gpu_ranges = _distribute_params_evenly(P, self.num_gpus, chunk_size)

        # Send work to each worker
        for gpu_id in range(self.num_gpus):
            self.ranges_queues[gpu_id].put(gpu_ranges[gpu_id])

        # Wait for all workers to finish
        self.done_barrier.wait()

        # Copy result to device
        return self.grad_shared[:P].to(device).clone()

    def shutdown(self):
        if _MultiGPUPool._active_pool is self:
            _MultiGPUPool._active_pool = None
        self.shutdown_event.set()
        for q in self.ranges_queues:
            try:
                q.put(None)  # sentinel
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        self.processes.clear()
        # Release any remaining GPU cache in the main process
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        self.shutdown()


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

        self._gpu_pool = None  # lazy-initialized persistent pool

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

        # Use persistent pool for multi-GPU (avoids spawn overhead each iteration)
        if self._num_gpus > 1 and not self.is_partial:
            if self._gpu_pool is None:
                # Free main-process GPU cache so workers have room (especially GPU 0)
                torch.cuda.empty_cache()
                self._gpu_pool = _MultiGPUPool(
                    self._num_gpus, circuit, hamiltonian,
                    self.shift, self.shots, self.measure_method,
                    self.batch_size, theta.numel(),
                )
            device = circuit.device
            val = self._gpu_pool.compute_gradient(theta, self.batch_size, device)
        else:
            val = batch_gradient(theta, circuit, hamiltonian, self.batch_size, self.shots,
                                 self.shift, self.depth, self.curr_depth, self.is_partial, self.measure_method,
                                 num_gpus=1)
        self.curr_depth = (self.curr_depth + 1) % self.depth
        return val

    def __del__(self):
        if self._gpu_pool is not None:
            self._gpu_pool.shutdown()
            self._gpu_pool = None


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
            # Distribute parameters evenly across GPUs
            gpu_ranges = _distribute_params_evenly(P, num_gpus, chunk_size)

            # CPU copies for safe cross-process sharing
            circuit_cpu = circuit.to_device('cpu')
            base_cpu = base.cpu()

            # Shared-memory tensor: workers write directly, no Queue needed
            grad_shared = torch.zeros(P, dtype=torch.float32).share_memory_()

            total_chunks = sum(len(v) for v in gpu_ranges.values())
            print(f"[TREV] Spawning {num_gpus} workers, P={P}, chunk_size={chunk_size}, "
                  f"total_chunks={total_chunks}", flush=True)
            for gpu_id in range(num_gpus):
                n_ch = len(gpu_ranges[gpu_id])
                n_p = sum(s[1]-s[0] for s in gpu_ranges[gpu_id])
                print(f"[TREV]   GPU {gpu_id}: {n_ch} chunks, {n_p} params", flush=True)

            # Launch worker processes (spawn context for CUDA safety)
            ctx = mp.get_context('spawn')
            processes = []
            for gpu_id in range(num_gpus):
                if not gpu_ranges[gpu_id]:
                    continue
                p = ctx.Process(
                    target=_mp_worker_fn,
                    args=(gpu_id, gpu_ranges[gpu_id], base_cpu, circuit_cpu,
                          hamiltonian, shift, shots, measure_method,
                          chunk_size, grad_shared),
                )
                p.start()
                print(f"[TREV] GPU {gpu_id} process started (PID {p.pid})", flush=True)
                processes.append(p)

            # Wait for all workers to finish
            for p in processes:
                p.join()

            # Check for crashed workers
            for p in processes:
                if p.exitcode != 0:
                    print(f"[TREV] WARNING: Worker PID {p.pid} exited with code {p.exitcode}", flush=True)

            # Copy shared result to device
            grad[:] = grad_shared.to(device)
            # Release main process GPU cache
            torch.cuda.empty_cache()
            print(f"[TREV] All workers done, grad copied to {device}", flush=True)
        else:
            # Build prefix checkpoints once for the entire gradient computation
            checkpoints, param_to_op = circuit._build_prefix_checkpoints(base.squeeze(0))

            for start in range(0, P, chunk_size):
                stop   = min(start + chunk_size, P)
                C      = stop - start
                idx    = torch.arange(start, stop, device=device)

                # Build ring using prefix caching
                ring = circuit.build_tensor_paramshift(
                    base.squeeze(0), idx, shift, checkpoints, param_to_op
                )

                exp_vals = _dispatch_expectation(
                    None, circuit, hamiltonian, shots, measure_method, ring_tensor=ring
                )
                grad[start:stop] = 0.5 * (exp_vals[:C] - exp_vals[C:])
        return grad

def expectation_value_batch(
    param_batch: torch.Tensor,
    circuit: Circuit,
    hamiltonian: Hamiltonian,
    shots: int,
    seed: int | None = None,
    ring_tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fully parallel run_circuit with batched parameter inputs.
    param_batch: (B, 2L)
    returns: (B,) expectations
    """
    with torch.no_grad():
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        device = circuit.device
        if ring_tensor is not None:
            ring_tensor_batch = ring_tensor
            B = ring_tensor_batch.shape[0]
        else:
            B = param_batch.shape[0]
            if device == 'cuda':
                torch.cuda.synchronize()
            ring_tensor_batch = circuit.build_tensor_batch(param_batch, B)
        B, N = ring_tensor_batch.shape[:2]

        groups = hamiltonian.get_qwc_groups()
        op_tensor = hamiltonian.get_pauli_op_tensor().to(device=device)
        all_coeffs = hamiltonian.coefficients
        shots_per_group = max(1, shots // len(groups))

        q0 = torch.tensor([[1], [0]], dtype=torch.cfloat, device=device)
        q1 = torch.tensor([[0], [1]], dtype=torch.cfloat, device=device)

        grand_totals = torch.zeros(B, dtype=torch.float64, device=device)

        for group in groups:
            idx = group['term_indices']
            nonI_mask = (op_tensor[idx] != 0).to(device=device, dtype=torch.bool)  # (G, N)
            group_coeffs = torch.as_tensor(
                [all_coeffs[t] for t in idx], dtype=torch.float64, device=device
            )

            rotated = rotate_tensor_for_measurement(ring_tensor_batch, group['basis'])

            shot_chunk = shots_per_group
            totals = torch.zeros(B, dtype=torch.float64, device=device)
            done = 0

            for s0 in range(0, shots_per_group, shot_chunk):
                s1 = min(s0 + shot_chunk, shots_per_group)
                S = s1 - s0
                if S == 0:
                    continue

                batch_prev: torch.Tensor | None = None
                bits = torch.empty((B, S, N), dtype=torch.bool, device=device)

                for i in range(N):
                    curr_tens = rotated[:, i].contiguous()

                    if i == 0:
                        qubit_0 = torch.einsum('bijk,kl->bijl', curr_tens, q0).squeeze(-1)
                        qubit_1 = torch.einsum('bijk,kl->bijl', curr_tens, q1).squeeze(-1)
                        batch_qubit_0 = qubit_0.unsqueeze(1).expand(-1, S, -1, -1).contiguous()
                        batch_qubit_1 = qubit_1.unsqueeze(1).expand(-1, S, -1, -1).contiguous()
                    else:
                        contracted = torch.einsum('bsij,bjkl->bsikl', batch_prev, curr_tens)
                        batch_qubit_0 = torch.einsum('bsijk,kl->bsijl', contracted, q0).squeeze(-1).contiguous()
                        batch_qubit_1 = torch.einsum('bsijk,kl->bsijl', contracted, q1).squeeze(-1).contiguous()

                    prob_0 = torch.einsum('bsij,bsij->bs', batch_qubit_0.conj(), batch_qubit_0).real
                    prob_1 = torch.einsum('bsij,bsij->bs', batch_qubit_1.conj(), batch_qubit_1).real
                    total = prob_0 + prob_1

                    zero_mask = total == 0
                    prob_0 = torch.where(zero_mask, torch.full_like(prob_0, 0.5), prob_0)
                    prob_1 = torch.where(zero_mask, torch.full_like(prob_1, 0.5), prob_1)
                    total = prob_0 + prob_1
                    p0 = prob_0 / total

                    rnd = torch.rand(B, S, device=device)
                    choose_1 = rnd > p0
                    bits[:, :, i] = choose_1

                    batch_prev = torch.where(
                        choose_1.unsqueeze(-1).unsqueeze(-1),
                        batch_qubit_1,
                        batch_qubit_0
                    )

                # Score using non-I mask for this group
                bf = bits.to(torch.float32).reshape(B * S, N)
                G = len(idx)
                cnt = bf @ nonI_mask.to(torch.float32).T  # (B*S, G)
                parity = (cnt.remainder_(2.0) > 0.5)
                sgn = torch.where(parity, -1.0, 1.0)
                Eb = (sgn * group_coeffs.view(1, -1)).sum(dim=1)  # (B*S,)

                totals += Eb.view(B, S).sum(dim=1)
                done += S

            if done > 0:
                grand_totals += totals / done

        return grand_totals.float().detach()



def _kron_contract_right(Prod, A0, A1, op=0):
    """Contract Prod @ E(site) using Kronecker decomposition.

    op=0 (I): E = conj(A0)⊗A0 + conj(A1)⊗A1
    op=3 (Z): E = conj(A0)⊗A0 - conj(A1)⊗A1
    op=1 (X): E = conj(A0)⊗A1 + conj(A1)⊗A0
    op=2 (Y): E = -i·conj(A0)⊗A1 + i·conj(A1)⊗A0

    Prod: (B, ..., l_bra, l_ket, r_bra, r_ket)  -- last 4 dims are spatial
    A0, A1: (B, chi, chi)

    Contracts r_bra/r_ket (last 2 dims of Prod) and produces new right indices.
    A must broadcast over all dims between B and r_bra/r_ket (i.e., middle + l_bra + l_ket).
    """
    # Number of dims to broadcast over: everything between B (dim 0) and r_bra/r_ket (last 2)
    n_broadcast = Prod.dim() - 3  # = n_middle + l_bra + l_ket
    slices = (slice(None),) + (None,) * n_broadcast + (slice(None), slice(None))
    A0H_e = A0.conj().mT[slices]
    A1H_e = A1.conj().mT[slices]

    if op == 0:  # I: conj(A0)⊗A0 + conj(A1)⊗A1
        A0_e = A0[slices]
        A1_e = A1[slices]
        r0 = torch.matmul(A0H_e, torch.matmul(Prod, A0_e))
        r1 = torch.matmul(A1H_e, torch.matmul(Prod, A1_e))
        return r0 + r1
    elif op == 3:  # Z: conj(A0)⊗A0 - conj(A1)⊗A1
        A0_e = A0[slices]
        A1_e = A1[slices]
        r0 = torch.matmul(A0H_e, torch.matmul(Prod, A0_e))
        r1 = torch.matmul(A1H_e, torch.matmul(Prod, A1_e))
        return r0 - r1
    elif op == 1:  # X: conj(A0)⊗A1 + conj(A1)⊗A0
        A0_e = A0[slices]
        A1_e = A1[slices]
        r0 = torch.matmul(A0H_e, torch.matmul(Prod, A1_e))
        r1 = torch.matmul(A1H_e, torch.matmul(Prod, A0_e))
        return r0 + r1
    else:  # op == 2, Y: -i·conj(A0)⊗A1 + i·conj(A1)⊗A0
        A0_e = A0[slices]
        A1_e = A1[slices]
        r0 = torch.matmul(A0H_e, torch.matmul(Prod, A1_e))
        r1 = torch.matmul(A1H_e, torch.matmul(Prod, A0_e))
        return -1j * r0 + 1j * r1


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
    ring_tensor: Optional[Tensor] = None,  # pre-built ring tensor, skip build if provided
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
    device = getattr(circuit, "device",
                     param_batch.device if param_batch is not None else 'cpu')

    B_total = ring_tensor.shape[0] if ring_tensor is not None else int(param_batch.shape[0])
    if param_chunk is None or param_chunk >= B_total:
        batch_slices = [(0, B_total)]
    else:
        batch_slices = [(s, min(s + param_chunk, B_total)) for s in range(0, B_total, param_chunk)]

    # Hamiltonian
    op_tensor = hamiltonian.get_pauli_op_tensor().to(device)  # (T, N) uint8
    coeffs = torch.as_tensor(
        hamiltonian.coefficients,
        dtype=torch.cfloat if use_complex64 else torch.cdouble,
        device=device,
    )
    T, N = op_tensor.shape

    out_parts = []

    for lo, hi in batch_slices:
        if ring_tensor is not None:
            ring = ring_tensor[lo:hi].to(device)
            B = ring.shape[0]
        else:
            param_view = param_batch[lo:hi]
            B = int(param_view.shape[0])
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

        # --- Per-term contraction: only at non-identity sites ---
        totals = torch.zeros(B, dtype=ctype, device=device)

        for t in range(T):
            non_i_sites = torch.where(op_tensor[t] != 0)[0].tolist()

            if len(non_i_sites) == 0:
                # All identity: Tr(full ring) = (L_pre[N] * R_suf_T[N]).sum
                totals += coeffs[t] * (L_pre[N] * R_suf_T[N]).sum(dim=(1, 2, 3, 4))
                continue

            s_first = non_i_sites[0]
            s_last = non_i_sites[-1]

            # Start from the precomputed left prefix up to the first non-I site
            run = L_pre[s_first].clone()

            # Contract through sites s_first..s_last
            for i in range(s_first, s_last + 1):
                A0_i, A1_i = sites[i]
                op_i = op_tensor[t, i].item()
                run = _kron_contract_right(run, A0_i, A1_i, op=op_i)

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
    ring_tensor: Optional[Tensor] = None,  # pre-built ring tensor, skip build if provided
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
    device = getattr(circuit, "device",
                     param_batch.device if param_batch is not None else 'cpu')
    ctype = torch.complex64 if use_complex64 else torch.complex128

    B_total = ring_tensor.shape[0] if ring_tensor is not None else int(param_batch.shape[0])
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
        if ring_tensor is not None:
            ring = ring_tensor[lo:hi]
            B = ring.shape[0]
        else:
            param_view = param_batch[lo:hi]
            B = int(param_view.shape[0])
            ring = circuit.build_tensor_batch(param_view, B)     # (B, N, chi, chi, 2)
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

    # ----- 2) QWC groups and Hamiltonian -----
    groups = hamiltonian.get_qwc_groups()
    op_tensor = hamiltonian.get_pauli_op_tensor().to(device)  # (T, n) uint8
    all_coeffs = hamiltonian.coefficients
    shots_per_group = max(1, shots // len(groups))

    chi2 = chi * chi
    env_dtype = torch.complex64 if use_fp32_env else cdtype

    def E_from_slices(A0i: Tensor, A1i: Tensor) -> Tensor:
        E0 = torch.einsum('bij,bkl->bikjl', A0i.conj(), A0i).reshape(B, chi2, chi2)
        E1 = torch.einsum('bij,bkl->bikjl', A1i.conj(), A1i).reshape(B, chi2, chi2)
        return (E0 + E1).to(env_dtype)

    # ----- 3) RNG -----
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    if shot_chunk is None:
        shot_chunk = shots_per_group

    grand_total = torch.zeros(B, dtype=rtype, device=device)

    for group in groups:
        idx = group['term_indices']
        nonI_mask = (op_tensor[idx] != 0).to(device=device, dtype=torch.bool)  # (G, n)
        group_coeffs = torch.as_tensor(
            [all_coeffs[t] for t in idx], dtype=rtype, device=device
        )
        G = len(idx)

        # Rotate cores for this group's measurement basis
        rotated = rotate_tensor_for_measurement(cores, group['basis'])
        A0 = rotated[..., 0].contiguous()   # (B, n, chi, chi)
        A1 = rotated[..., 1].contiguous()

        # ----- Build R_suf for rotated cores -----
        Id = torch.eye(chi2, dtype=env_dtype, device=device).expand(B, chi2, chi2).clone()
        R_suf = [None] * n
        acc = Id
        for i in range(n - 1, -1, -1):
            R_suf[i] = acc
            Ei = E_from_slices(A0[:, i], A1[:, i])
            acc = torch.bmm(Ei, acc)
            del Ei
        del acc, Id

        # ----- Sampler over shots (in chunks) -----
        total = torch.zeros(B, dtype=rtype, device=device)
        done = 0

        eye_chi = torch.eye(chi, dtype=cdtype, device=device)
        for s0 in range(0, shots_per_group, shot_chunk):
            s1 = min(s0 + shot_chunk, shots_per_group)
            S = s1 - s0

            X = eye_chi.expand(B, S, chi, chi).clone()
            bits = torch.empty((B, S, n), dtype=torch.bool, device=device)

            for i in range(n):
                A0i = A0[:, i].unsqueeze(1)
                A1i = A1[:, i].unsqueeze(1)

                M0 = torch.matmul(X, A0i)
                M1 = torch.matmul(X, A1i)

                v0 = M0.reshape(B, S, chi2)
                v1 = M1.reshape(B, S, chi2)

                Ri = R_suf[i]
                y0 = torch.einsum('bij,bsj->bsi', Ri, v0)
                y1 = torch.einsum('bij,bsj->bsi', Ri, v1)

                w0 = (v0.conj() * y0).sum(dim=-1).real
                w1 = (v1.conj() * y1).sum(dim=-1).real

                den = (w0 + w1).clamp_min(1e-300)
                p0 = (w0 / den).to(rtype)
                si = (torch.rand((B, S), generator=gen, device=device) >= p0)
                bits[:, :, i] = si
                X = torch.where(si.unsqueeze(-1).unsqueeze(-1), M1, M0)

                del M0, M1, v0, v1, y0, y1

            # Score using non-I mask for this group
            bf = bits.to(torch.float32)
            cnt = torch.einsum('bsn,gn->bsg', bf, nonI_mask.float())  # (B, S, G)
            sgn = torch.where((cnt.remainder_(2.0) > 0.5), -1.0, 1.0).to(rtype)
            Eb = torch.einsum('bsg,g->bs', sgn, group_coeffs)

            total += Eb.sum(dim=1)
            done += S

            del X, bits, Eb, bf

        if done > 0:
            grand_total += total / done

        del R_suf, A0, A1

    del cores
    return grand_total
