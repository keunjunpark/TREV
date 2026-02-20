import torch
import torch.multiprocessing as mp
from torch import Tensor

from TREV.optimization.gradients.set_batch_size import auto_batch_size

from ...circuit import Circuit
from ...hamiltonian.hamiltonian import Hamiltonian
from ...measure.enums import MeasureMethod
from ...optimization.gradients.gradient import Gradient
from .batch_parameter_shift import (
    _dispatch_expectation,
    _get_gpu_count,
    _distribute_params_evenly,
    _gpu_info,
)


def _mp_worker_fn_fd(gpu_id, ranges, base_cpu, circuit_cpu, hamiltonian, epsilon, shots,
                     measure_method, chunk_size, grad_shared):
    """Multiprocessing worker: compute finite-diff gradient slices on assigned GPU."""
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
            batch[arange_C, idx] += epsilon
            batch[C + arange_C, idx] -= epsilon

            exp_vals = _dispatch_expectation(batch, circuit_clone, hamiltonian, shots, measure_method)
            grad_slice = (exp_vals[:C] - exp_vals[C:]) / (2 * epsilon)
            grad_shared[start:stop] = grad_slice.cpu()
    except Exception as e:
        print(f"[TREV] GPU {gpu_id} worker FAILED: {e}", flush=True)
        traceback.print_exc()
    finally:
        del circuit_clone, base_dev
        torch.cuda.empty_cache()


def _persistent_worker_fn_fd(gpu_id, circuit_cpu, hamiltonian, epsilon, shots,
                             measure_method, chunk_size,
                             base_shared, grad_shared, ranges_queue, done_barrier,
                             shutdown_event):
    """Persistent process worker for finite-diff — stays alive across iterations."""
    import traceback
    device = f'cuda:{gpu_id}'
    circuit_clone = circuit_cpu.to_device(device)

    try:
        while not shutdown_event.is_set():
            try:
                msg = ranges_queue.get(timeout=1.0)
            except Exception:
                continue
            if msg is None:
                break

            ranges = msg
            try:
                base_dev = base_shared.to(device).unsqueeze(0)

                for start, stop in ranges:
                    C = stop - start
                    idx = torch.arange(start, stop, device=device)
                    arange_C = torch.arange(C, device=device)

                    batch = base_dev.expand(2 * C, -1).clone()
                    batch[arange_C, idx] += epsilon
                    batch[C + arange_C, idx] -= epsilon

                    exp_vals = _dispatch_expectation(
                        batch, circuit_clone, hamiltonian, shots, measure_method)
                    grad_slice = (exp_vals[:C] - exp_vals[C:]) / (2 * epsilon)
                    grad_shared[start:stop] = grad_slice.cpu()
            except Exception as e:
                print(f"[TREV] GPU {gpu_id} worker FAILED: {e}", flush=True)
                traceback.print_exc()

            torch.cuda.empty_cache()
            done_barrier.wait()
    finally:
        del circuit_clone
        torch.cuda.empty_cache()


class _MultiGPUPoolFD:
    """Persistent pool of worker processes for multi-GPU finite-diff gradient computation."""

    _active_pool = None

    def __init__(self, num_gpus, circuit, hamiltonian, epsilon, shots,
                 measure_method, chunk_size, P):
        if _MultiGPUPoolFD._active_pool is not None:
            print("[TREV] Shutting down previous multi-GPU FD pool", flush=True)
            _MultiGPUPoolFD._active_pool.shutdown()
        _MultiGPUPoolFD._active_pool = self

        ctx = mp.get_context('spawn')

        circuit_cpu = circuit.to_device('cpu')
        self.num_gpus = num_gpus
        self.P = P

        self.base_shared = torch.zeros(P, dtype=torch.float32).share_memory_()
        self.grad_shared = torch.zeros(P, dtype=torch.float32).share_memory_()

        self.ranges_queues = [ctx.Queue() for _ in range(num_gpus)]
        self.done_barrier = ctx.Barrier(num_gpus + 1)
        self.shutdown_event = ctx.Event()

        self.processes = []
        for gpu_id in range(num_gpus):
            p = ctx.Process(
                target=_persistent_worker_fn_fd,
                args=(gpu_id, circuit_cpu, hamiltonian, epsilon, shots,
                      measure_method, chunk_size,
                      self.base_shared, self.grad_shared,
                      self.ranges_queues[gpu_id], self.done_barrier,
                      self.shutdown_event),
            )
            p.daemon = True
            p.start()
            self.processes.append(p)
        print(f"[TREV] Persistent multi-GPU FD pool started: {num_gpus} workers", flush=True)

    def compute_gradient(self, params, chunk_size, device):
        P = params.numel()

        self.base_shared[:P] = params.detach().cpu()
        torch.cuda.empty_cache()

        gpu_ranges = _distribute_params_evenly(P, self.num_gpus, chunk_size)

        for gpu_id in range(self.num_gpus):
            self.ranges_queues[gpu_id].put(gpu_ranges[gpu_id])

        self.done_barrier.wait()

        return self.grad_shared[:P].to(device).clone()

    def shutdown(self):
        if _MultiGPUPoolFD._active_pool is self:
            _MultiGPUPoolFD._active_pool = None
        self.shutdown_event.set()
        for q in self.ranges_queues:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self.processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        self.processes.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        self.shutdown()


class BatchFiniteDiffGradient(Gradient):
    def __init__(self, epsilon=1e-3, batch_size=None, shots=0,
                 measure_method: MeasureMethod = MeasureMethod.EFFICIENT_CONTRACTION,
                 depth: int = 1, is_partial: bool = False, num_gpus: int | None = None):
        super().__init__(measure_method)
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.shots = shots
        self.depth = depth
        self.curr_depth = 0
        self.is_partial = is_partial
        self._autotuned = False

        if num_gpus is None:
            self._num_gpus = _get_gpu_count()
        else:
            self._num_gpus = num_gpus

        self._gpu_pool = None
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
                param_batch = base.expand(2 * C, -1).clone()
                param_batch[arange_C, idx] += self.epsilon
                param_batch[C + arange_C, idx] -= self.epsilon
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

        if self._num_gpus > 1 and not self.is_partial:
            if self._gpu_pool is None:
                torch.cuda.empty_cache()
                self._gpu_pool = _MultiGPUPoolFD(
                    self._num_gpus, circuit, hamiltonian,
                    self.epsilon, self.shots, self.measure_method,
                    self.batch_size, theta.numel(),
                )
            device = circuit.device
            val = self._gpu_pool.compute_gradient(theta, self.batch_size, device)
        else:
            val = batch_gradient_finite_diff(
                theta, circuit, hamiltonian, self.batch_size, self.shots,
                self.epsilon, self.depth, self.curr_depth, self.is_partial, self.measure_method,
                num_gpus=1)
        self.curr_depth = (self.curr_depth + 1) % self.depth
        return val

    def __del__(self):
        if self._gpu_pool is not None:
            self._gpu_pool.shutdown()
            self._gpu_pool = None


def batch_gradient_finite_diff(
        params:     torch.Tensor,
        circuit:    Circuit,
        hamiltonian: Hamiltonian,
        chunk_size: int,
        shots:      int,
        epsilon:    float,
        depth:      int,
        curr_depth: int,
        is_partial: bool,
        measure_method: MeasureMethod,
        num_gpus: int | None = None,
) -> torch.Tensor:
    """
    Memory-frugal central finite-difference gradient.

    params     : (P,)  -- single circuit's parameters
    chunk_size : how many theta-indices to perturb at once
    epsilon    : finite difference step size
    num_gpus   : number of GPUs to use (None or <=1 for single-GPU)
    returns    : (P,)  -- gradient d<O>/d_theta
    """
    with torch.no_grad():
        device = circuit.device
        P      = params.numel()
        grad   = torch.empty(P, device=device, dtype=torch.float32)
        base   = params.detach().to(device).unsqueeze(0)

        if is_partial:
            dP = P // depth
            start = dP * curr_depth
            stop = min(start + dP, P)
            C = stop - start

            idx = torch.arange(start, stop, device=device)
            arange_C = torch.arange(C, device=device)

            batch = base.expand(2 * C, -1).clone()
            batch[arange_C, idx] += epsilon
            batch[C + arange_C, idx] -= epsilon

            exp_vals = _dispatch_expectation(batch, circuit, hamiltonian, shots, measure_method)
            grad[start:stop] = (exp_vals[:C] - exp_vals[C:]) / (2 * epsilon)
        elif num_gpus is not None and num_gpus > 1 and P > 0:
            gpu_ranges = _distribute_params_evenly(P, num_gpus, chunk_size)

            circuit_cpu = circuit.to_device('cpu')
            base_cpu = base.cpu()

            grad_shared = torch.zeros(P, dtype=torch.float32).share_memory_()

            total_chunks = sum(len(v) for v in gpu_ranges.values())
            print(f"[TREV] Spawning {num_gpus} workers, P={P}, chunk_size={chunk_size}, "
                  f"total_chunks={total_chunks}", flush=True)
            for gpu_id in range(num_gpus):
                n_ch = len(gpu_ranges[gpu_id])
                n_p = sum(s[1]-s[0] for s in gpu_ranges[gpu_id])
                print(f"[TREV]   GPU {gpu_id}: {n_ch} chunks, {n_p} params", flush=True)

            ctx = mp.get_context('spawn')
            processes = []
            for gpu_id in range(num_gpus):
                if not gpu_ranges[gpu_id]:
                    continue
                p = ctx.Process(
                    target=_mp_worker_fn_fd,
                    args=(gpu_id, gpu_ranges[gpu_id], base_cpu, circuit_cpu,
                          hamiltonian, epsilon, shots, measure_method,
                          chunk_size, grad_shared),
                )
                p.start()
                print(f"[TREV] GPU {gpu_id} process started (PID {p.pid})", flush=True)
                processes.append(p)

            for p in processes:
                p.join()

            for p in processes:
                if p.exitcode != 0:
                    print(f"[TREV] WARNING: Worker PID {p.pid} exited with code {p.exitcode}", flush=True)

            grad[:] = grad_shared.to(device)
            torch.cuda.empty_cache()
            print(f"[TREV] All workers done, grad copied to {device}", flush=True)
        else:
            for start in range(0, P, chunk_size):
                stop   = min(start + chunk_size, P)
                C      = stop - start
                idx    = torch.arange(start, stop, device=device)
                arange_C = torch.arange(C, device=device)

                batch = base.expand(2 * C, -1).clone()
                batch[arange_C, idx] += epsilon
                batch[C + arange_C, idx] -= epsilon

                exp_vals = _dispatch_expectation(batch, circuit, hamiltonian, shots, measure_method)
                grad[start:stop] = (exp_vals[:C] - exp_vals[C:]) / (2 * epsilon)
        return grad
