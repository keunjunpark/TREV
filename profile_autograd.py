"""
Profile: AutogradGradient — VQE iteration breakdown.

Shows where time goes in a full VQE iteration for autograd vs param-shift.
"""

import time
import torch
import numpy as np

from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.enums import MeasureMethod
from TREV.measure.efficient_contraction import expectation_value_batch as ev_exact
from TREV.measure.contraction import argmax_tr_noinv_BE
from TREV.optimization.gradients.autograd_gradient import (
    AutogradGradient, autograd_gradient, _auto_term_chunk,
)
from TREV.optimization.gradients.batch_parameter_shift import (
    BatchParameterShiftGradient,
)
from TREV.optimization.optimizer import Optimizer
from TREV.optimization.optimization import minimize


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


def build_circuit(n, chi, layers=2, device='cuda'):
    circuit = Circuit(num_qubit=n, rank=chi, device=device)
    for i in range(n):
        circuit.h(i)
    for _ in range(layers):
        for i in range(n):
            circuit.cx(i, (i + 1) % n)
        for i in range(n):
            circuit.ry(i)
            circuit.rz(i)
    theta = torch.randn(circuit.params_size, device=device)
    return circuit, theta


def timeit(fn, warmup=3, repeats=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times) * 1000


def main():
    device = 'cuda'
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    configs = [
        (8, 4, 2),
        (8, 10, 2),
        (12, 4, 2),
        (12, 10, 2),
    ]

    # =========================================================
    # Part 1: VQE iteration breakdown
    # =========================================================
    print("=" * 70)
    print("Part 1: VQE iteration breakdown (what takes time)")
    print("=" * 70)

    print(f"\n{'config':<25} | {'gradient':>10} {'ev_value':>10} {'argmax':>10} | {'total':>10} {'grad%':>6}")
    print("-" * 80)

    for N, chi, L in configs:
        torch.cuda.empty_cache()
        ham = build_maxcut_hamiltonian(N)
        circuit, theta = build_circuit(N, chi, L, device=device)

        # Warmup
        autograd_gradient(theta, circuit, ham, torch.cfloat)
        ev_exact(circuit.build_tensor(theta), ham, device=device)
        argmax_tr_noinv_BE(circuit.build_tensor(theta), device)
        torch.cuda.synchronize()

        ms_grad = timeit(lambda: autograd_gradient(theta, circuit, ham, torch.cfloat))
        ms_ev = timeit(lambda: ev_exact(circuit.build_tensor(theta), ham, device=device))
        ms_argm = timeit(lambda: argmax_tr_noinv_BE(circuit.build_tensor(theta), device))
        total = ms_grad + ms_ev + ms_argm
        pct = ms_grad / total * 100

        label = f"N={N:>2} chi={chi:>2} L={L}"
        print(f"{label:<25} | {ms_grad:>8.1f}ms {ms_ev:>8.1f}ms {ms_argm:>8.1f}ms | {total:>8.1f}ms {pct:>5.1f}%")

    # =========================================================
    # Part 2: VQE with cached exp_value (autograd skips rebuild)
    # =========================================================
    print()
    print("=" * 70)
    print("Part 2: Full VQE — autograd (with cache) vs param-shift")
    print("=" * 70)

    iters = 20
    print(f"\n{'config':<25} | {'autograd':>12} {'param-shift':>12} {'speedup':>8}")
    print("-" * 65)

    for N, chi, L in configs:
        torch.cuda.empty_cache()
        ham = build_maxcut_hamiltonian(N)
        circuit, theta = build_circuit(N, chi, L, device=device)
        opt = Optimizer(torch.optim.Adam, {'lr': 0.05})

        # Autograd
        grad_ad = AutogradGradient()
        grad_ad._verbose = False
        t0 = time.time()
        minimize(circuit, theta.clone(), ham, opt, grad_ad,
                 iteration=iters, best_value_method='contraction')
        t_ad = (time.time() - t0) / iters * 1000

        # Param-shift
        grad_ps = BatchParameterShiftGradient(
            shift=np.pi / 2, batch_size=None, shots=0,
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION, depth=1,
        )
        grad_ps._verbose = False
        t0 = time.time()
        minimize(circuit, theta.clone(), ham, opt, grad_ps,
                 iteration=iters, best_value_method='contraction')
        t_ps = (time.time() - t0) / iters * 1000

        sp = t_ps / t_ad
        label = f"N={N:>2} chi={chi:>2} L={L}"
        print(f"{label:<25} | {t_ad:>10.1f}ms {t_ps:>10.1f}ms {sp:>7.2f}x")

    print()


if __name__ == '__main__':
    main()
