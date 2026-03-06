from typing import Callable

import torch

import sys
import time
import gc
from ..circuit import Circuit
from ..hamiltonian.hamiltonian import Hamiltonian
from ..measure.contraction import get_value_of_highest_probability, argmax_tr_noinv_BE, contract_tensor_ring
from ..measure.enums import MeasureMethod
from ..optimization.gradients.gradient import Gradient
from ..optimization.optimizer import Optimizer
from ..measure.right_suffix_sampling import argmax_bitstring_tr_right_suffix
from ..gates import contraction as _contraction
import cProfile
import time, gc, torch
from TREV.optimization.gradients.gradient import MeasureMethod

def minimize(
    circuit: Circuit,
    theta: torch.Tensor,
    hamiltonian: Hamiltonian,
    optimizer: Optimizer,
    gradient: Gradient,
    iteration: int,
    best_value_method: str,
    wall_clock_cap: float | None = None,
    param_mapping: torch.Tensor | None = None,
    param_base: torch.Tensor | None = None,
    upcast: bool = False,
):
    """
    Minimization loop with optional wall clock cap.

    When *param_mapping* (P, K) and *param_base* (P,) are supplied, the
    optimizer works in the K-dimensional subspace of logical parameters.
    ``theta`` should then be K-dimensional.  The full TREV theta is
    recovered as ``full_theta = param_base + param_mapping @ theta``.

    When *upcast* is True, SVD operations in 2-qubit gate contractions
    are performed in cdouble for numerical stability, then cast back.
    """
    _prev_upcast = _contraction.UPCAST_SVD
    _contraction.UPCAST_SVD = upcast
    with torch.no_grad():
        theta = theta.clone().to(circuit.device)
        if param_mapping is not None:
            param_mapping = param_mapping.to(circuit.device)
            param_base = param_base.to(circuit.device)
        optim = optimizer.get_optimizer([theta])
        lr = optimizer.args['lr']

        # When optimizing in a subspace, only shift TREV params that
        # actually affect the subspace gradient (non-zero Jacobian rows).
        if param_mapping is not None and hasattr(gradient, 'active_params'):
            active_mask = param_mapping.abs().sum(dim=1) > 1e-10
            active_idx = torch.where(active_mask)[0].to(circuit.device)
            gradient.active_params = active_idx
            print(f"[TREV] Subspace active params: {active_idx.numel()}/{param_mapping.shape[0]} "
                  f"({100*active_idx.numel()/param_mapping.shape[0]:.0f}%)", flush=True)

        exp_values = []
        best_result = []
        iteration_times = []

        start = time.time()
        time_after_first_iter = None

        for epoch in range(iteration):
            it_time = time.time()
            optim.zero_grad()

            # Compute full-space theta
            if param_mapping is not None:
                full_theta = param_base + param_mapping @ theta
            else:
                full_theta = theta

            _t0 = time.time()
            grad = gradient.run(full_theta, circuit, hamiltonian)

            # Project gradient back to subspace if needed
            if param_mapping is not None:
                theta.grad = param_mapping.T @ grad
            else:
                theta.grad = grad

            optim.step()
            if circuit.device == 'cuda':
                torch.cuda.synchronize()
            _t_grad = time.time() - _t0
            iteration_times.append(time.time() - it_time)

            # Recompute full_theta after optimizer step
            if param_mapping is not None:
                full_theta = param_base + param_mapping @ theta

            # Build tensor once for both exp_value and best_result
            _t0 = time.time()
            _tensor = circuit.build_tensor(full_theta)

            # --- expectation value ---
            shots = getattr(gradient, 'shots', None)
            if shots is not None:
                exp_value = circuit.get_expectation_value(full_theta, hamiltonian, gradient.measure_method, int(shots), ring_tensor=_tensor)
            else:
                exp_value = circuit.get_expectation_value(full_theta, hamiltonian, gradient.measure_method, ring_tensor=_tensor)
            _t_exp = time.time() - _t0
            exp_values.append(exp_value.item() if isinstance(exp_value, torch.Tensor) else exp_value)

            # --- best result method ---
            _t0 = time.time()
            _tensor = circuit.build_tensor(full_theta)
            if best_value_method == 'highest_probability':
                best_result.append(
                    get_value_of_highest_probability(_tensor, circuit.device)
                )
            elif best_value_method == 'argmax_tr_noinv_BE':
                best_result.append(
                    argmax_bitstring_tr_right_suffix(_tensor)
                )
            elif best_value_method == 'full_contraction':
                best_idx = contract_tensor_ring(_tensor).abs().pow(2).argmax().item()
                num_qubits = circuit.num_qubit
                best_bitstring = format(best_idx, f'0{num_qubits}b')
                best_result.append(best_bitstring[::-1])
            else:
                if gradient.measure_method in [MeasureMethod.PERFECT_SAMPLING]:
                    best_result.append(
                        get_value_of_highest_probability(_tensor, circuit.device)
                    )
                elif gradient.measure_method in [MeasureMethod.FULL_CONTRACTION, MeasureMethod.EFFICIENT_CONTRACTION]:
                    best_result.append(
                        argmax_tr_noinv_BE(_tensor, circuit.device)
                    )
                elif gradient.measure_method in [MeasureMethod.RIGHT_SUFFIX_SAMPLING]:
                    best_result.append(
                        argmax_tr_noinv_BE(_tensor, circuit.device)
                    )
                else:
                    del _tensor
                    raise NotImplementedError()
            del _tensor
            _t_best = time.time() - _t0

            #print(f"\n[TREV] Epoch {epoch}: grad={_t_grad:.2f}s, exp_value={_t_exp:.2f}s, best_result={_t_best:.2f}s", flush=True)
            if epoch == 0:
                time_after_first_iter = time.time()
            progress_bar(epoch, iteration, time_after_first_iter if epoch > 0 else None, exp_values[-1])

            # Free intermediate GPU tensors every iteration
            del grad
            if param_mapping is not None:
                del full_theta
            if circuit.device == 'cuda':
                torch.cuda.empty_cache()

            # --- early stop by wall clock ---
            if wall_clock_cap is not None:
                elapsed = time.time() - start
                if elapsed >= wall_clock_cap:
                    print(f"[INFO] Early stop at epoch {epoch} due to wall-clock cap ({elapsed:.2f}s ≥ {wall_clock_cap:.2f}s)")
                    break

        # Shut down persistent multi-GPU workers so they release GPU memory
        if hasattr(gradient, '_gpu_pool') and gradient._gpu_pool is not None:
            gradient._gpu_pool.shutdown()
            gradient._gpu_pool = None

        # Final GPU cleanup
        gc.collect()
        if circuit.device == 'cuda':
            torch.cuda.empty_cache()

        _contraction.UPCAST_SVD = _prev_upcast
        return theta, exp_values, best_result, iteration_times

def progress_bar(current, total, start_time_after_first, loss=None, bar_len=30):
    percent = float(current) / total
    arrow = '=' * int(round(percent * bar_len) - 1) + '>' if current < total else '=' * bar_len
    spaces = ' ' * (bar_len - len(arrow))

    if start_time_after_first is not None and current > 0:
        elapsed = time.time() - start_time_after_first
        iters_done = current  # iterations done since iter 0
        iters_left = total - current
        eta = (elapsed / iters_done) * iters_left
    else:
        eta = 0

    days = int(eta // 86400)
    hours = int((eta % 86400) // 3600)
    minutes = int((eta % 3600) // 60)
    seconds = int(eta % 60)
    eta_str = f"{days}d {hours:02d}h {minutes:02d}m {seconds:02d}s"

    metrics = f" | Loss: {loss:.4f}" if loss is not None else ""

    sys.stdout.write(f'\rProgress: [{arrow}{spaces}] {int(percent * 100)}% | ETA: {eta_str}{metrics}')
    sys.stdout.flush()
