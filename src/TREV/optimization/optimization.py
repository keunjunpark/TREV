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
    wall_clock_cap: float | None = None,     # <--- NEW
):
    """
    Minimization loop with optional wall clock cap.
    Stops early if elapsed real time > wall_clock_cap.
    """
    with torch.no_grad():
        theta = theta.clone().to(circuit.device)
        optim = optimizer.get_optimizer([theta])
        lr = optimizer.args['lr']

        exp_values = []
        best_result = []
        iteration_times = []

        start = time.time()

        for epoch in range(iteration):
            it_time = time.time()
            optim.zero_grad()

            grad = gradient.run(theta, circuit, hamiltonian)
            theta.grad = grad
            optim.step()
            if circuit.device == 'cuda':
                torch.cuda.synchronize()
            iteration_times.append(time.time() - it_time)

            # --- expectation value ---
            exp_value = circuit.get_expectation_value(theta, hamiltonian, gradient.measure_method)
            exp_values.append(exp_value)

            # --- best result method ---
            if best_value_method == 'highest_probability':
                best_result.append(
                    get_value_of_highest_probability(circuit.build_tensor(theta), circuit.device)
                )
            elif best_value_method == 'argmax_tr_noinv_BE':
                best_result.append(
                    argmax_bitstring_tr_right_suffix(circuit.build_tensor(theta))
                )
            elif best_value_method == 'full_contraction':
                best_idx = contract_tensor_ring(circuit.build_tensor(theta)).abs().pow(2).argmax().item()
                num_qubits = circuit.num_qubit
                best_bitstring = format(best_idx, f'0{num_qubits}b')
                best_result.append(best_bitstring[::-1])
            else:
                if gradient.measure_method in [MeasureMethod.PERFECT_SAMPLING]:
                    best_result.append(
                        get_value_of_highest_probability(circuit.build_tensor(theta), circuit.device)
                    )
                elif gradient.measure_method in [MeasureMethod.FULL_CONTRACTION, MeasureMethod.EFFICIENT_CONTRACTION]:
                    best_result.append(
                        argmax_tr_noinv_BE(circuit.build_tensor(theta), circuit.device)
                    )
                elif gradient.measure_method in [MeasureMethod.RIGHT_SUFFIX_SAMPLING]:
                    best_result.append(
                        argmax_tr_noinv_BE(circuit.build_tensor(theta), circuit.device)
                    )
                else:
                    raise NotImplementedError()

            progress_bar(epoch, iteration, start, exp_value)

            if epoch % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            # --- early stop by wall clock ---
            if wall_clock_cap is not None:
                elapsed = time.time() - start
                if elapsed >= wall_clock_cap:
                    print(f"[INFO] Early stop at epoch {epoch} due to wall-clock cap ({elapsed:.2f}s ≥ {wall_clock_cap:.2f}s)")
                    break

        return theta, exp_values, best_result, iteration_times

def progress_bar(current, total, start_time, loss=None, bar_len=30):
    percent = float(current) / total
    arrow = '=' * int(round(percent * bar_len) - 1) + '>' if current < total else '=' * bar_len
    spaces = ' ' * (bar_len - len(arrow))

    elapsed = time.time() - start_time
    eta = (elapsed / current) * (total - current) if current > 0 else 0
    eta_str = time.strftime("%M:%S", time.gmtime(eta))

    metrics = f" | Loss: {loss:.4f}" if loss is not None else ""

    sys.stdout.write(f'\rProgress: [{arrow}{spaces}] {int(percent * 100)}% | ETA: {eta_str}{metrics}')
    sys.stdout.flush()
