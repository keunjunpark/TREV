#!/usr/bin/env python3
"""
Benchmark: Efficient Contraction vs Right Suffix Sampling
Tests TSP QAOA (ring topology) and PUCCD chemistry (X/Y Hamiltonian terms).

Usage:
    .venv/bin/python scripts/bench_ec_vs_rss.py
    .venv/bin/python scripts/bench_ec_vs_rss.py --skip-puccd   # TSP only (no qiskit-nature)
"""

import argparse
import math
import time
import gc
import itertools
import numpy as np
import torch

from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.enums import MeasureMethod
from TREV.optimization.gradients.batch_parameter_shift import (
    expectation_value_batch_efficient_contraction,
    expectation_value_batch_right_suffix,
    BatchParameterShiftGradient,
)
from TREV.optimization.optimization import minimize as trev_minimize
from TREV.optimization.optimizer import Optimizer
from TREV.measure.right_suffix_sampling import argmax_bitstring_tr_right_suffix

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASIS_GATES = ['swap', 'rzz', 'rx', 'h']
SHIFT = math.pi / 2


# ═══════════════════════════════════════════════════════════════
# TSP helpers
# ═══════════════════════════════════════════════════════════════

def make_tsp_ising(n_cities, seed):
    from qiskit_optimization.applications import Tsp
    from qiskit_optimization.converters import QuadraticProgramToQubo
    tsp_inst = Tsp.create_random_instance(n_cities, seed=seed)
    qp = tsp_inst.to_quadratic_program()
    qubo = QuadraticProgramToQubo().convert(qp)
    qubitOp, offset = qubo.to_ising()
    return qubitOp, offset, tsp_inst


def normalize_ising(qubitOp):
    max_coeff = max(abs(float(c.real)) for c in qubitOp.coeffs)
    if max_coeff > 0:
        qubitOp = qubitOp / max_coeff
    return qubitOp, max_coeff


def build_trev_hamiltonian(qubitOp):
    pauli_strings, coefficients = [], []
    for elm in qubitOp:
        pauli_strings.append(str(elm.paulis[0][::-1]))
        coefficients.append(float(elm.coeffs[0].real))
    return Hamiltonian(len(pauli_strings[0]), pauli_strings, coefficients)


def get_distance_matrix(tsp_inst):
    G = tsp_inst.graph
    n = len(G.nodes)
    dist = np.zeros((n, n))
    for i, j, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        dist[i][j] = w
        dist[j][i] = w
    return dist


def solve_tsp_brute(dist):
    n = dist.shape[0]
    best_cost, best_tour = float('inf'), None
    for perm in itertools.permutations(range(n)):
        cost = sum(dist[perm[i], perm[(i + 1) % n]] for i in range(n))
        if cost < best_cost:
            best_cost, best_tour = cost, list(perm)
    return best_cost, best_tour


def decode_tsp_bitstring(bits, n_cities, qubit_perm=None):
    if isinstance(bits, str):
        bits = [int(b) for b in bits]
    bits = list(bits)
    if qubit_perm is not None:
        inv_perm = [0] * len(qubit_perm)
        for i, p in enumerate(qubit_perm):
            inv_perm[p] = i
        bits = [bits[inv_perm[i]] for i in range(len(bits))]
    n = n_cities
    N = n * n
    if len(bits) < N:
        return False, None
    matrix = np.array(bits[:N]).reshape(n, n)
    if not (np.all(matrix.sum(axis=1) == 1) and np.all(matrix.sum(axis=0) == 1)):
        return False, None
    tour = [int(np.argmax(matrix[:, t])) for t in range(n)]
    return True, tour


def compute_tour_cost(tour, dist):
    n = len(tour)
    return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))


# ═══════════════════════════════════════════════════════════════
# VQE runner
# ═══════════════════════════════════════════════════════════════

def run_vqe(circuit, theta0, hamil, measure_method, shots, n_iters, lr, label):
    grad = BatchParameterShiftGradient(
        shift=SHIFT, batch_size=None, shots=shots,
        measure_method=measure_method, depth=1,
    )
    opt = Optimizer(torch.optim.Adam, {'lr': lr})
    theta_init = theta0.clone().detach().requires_grad_(True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    new_theta, exp_vals, best_results, iter_times = trev_minimize(
        circuit, theta_init, hamil, opt, grad, n_iters,
        best_value_method='argmax_tr_noinv_BE',
    )
    wall_time = time.time() - t0

    peak_gpu_mb = 0.0
    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    if hasattr(grad, '_gpu_pool') and grad._gpu_pool is not None:
        grad._gpu_pool.shutdown()
        grad._gpu_pool = None

    exp_floats = [float(v.real) if hasattr(v, 'real') else float(v) for v in exp_vals]
    med_iter = np.median(iter_times[2:]) if len(iter_times) > 2 else np.median(iter_times)
    print(f'  {label}: {wall_time:.1f}s total, {med_iter:.2f}s/iter, '
          f'final={exp_floats[-1]:.4f}, best={min(exp_floats):.4f}')

    return {
        'exp_values': exp_floats,
        'best_results': best_results,
        'iter_times': iter_times,
        'wall_time': wall_time,
        'peak_gpu_mb': peak_gpu_mb,
    }


# ═══════════════════════════════════════════════════════════════
# Test 1: TSP QAOA with ring topology
# ═══════════════════════════════════════════════════════════════

def run_tsp_benchmark(nc=3, reps=2, rank=8, n_iters=50, lr=5e-3, seed=0):
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit import transpile as qk_transpile
    from qiskit.transpiler import CouplingMap
    from TREV.transpile import from_qiskit

    print(f'\n{"="*60}')
    print(f'TSP QAOA: nc={nc}, reps={reps}, rank={rank}, seed={seed}')
    print(f'{"="*60}')

    qubitOp, offset, tsp_inst = make_tsp_ising(nc, seed)
    qubitOp_n, scale = normalize_ising(qubitOp)
    N = qubitOp_n.num_qubits
    hamil = build_trev_hamiltonian(qubitOp_n)

    dist = get_distance_matrix(tsp_inst)
    opt_cost, opt_tour = solve_tsp_brute(dist)
    print(f'  {N} qubits, {len(hamil.coefficients)} Hamiltonian terms')
    print(f'  Optimal tour: {opt_tour}, cost={opt_cost:.2f}')

    # Count Pauli types
    op_tensor = hamil.get_pauli_op_tensor()
    n_x = (op_tensor == 1).any(dim=1).sum().item()
    n_y = (op_tensor == 2).any(dim=1).sum().item()
    n_z = (op_tensor == 3).any(dim=1).sum().item()
    print(f'  Terms with X: {n_x}, Y: {n_y}, Z-only: {n_z}')

    # Build QAOA -> route for ring -> TREV
    qaoa = QAOAAnsatz(qubitOp_n, reps=reps)
    optimized = qk_transpile(qaoa, optimization_level=3, basis_gates=BASIS_GATES)
    cm = CouplingMap.from_ring(N)
    routed = qk_transpile(optimized, coupling_map=cm, optimization_level=1,
                          basis_gates=BASIS_GATES, seed_transpiler=seed)
    print(f'  Routed circuit: depth={routed.depth()}')

    K = 2 * reps
    gen = torch.Generator().manual_seed(seed)
    qaoa_x0 = 0.01 * torch.randn(K, generator=gen)
    sorted_params = sorted(routed.parameters, key=lambda p: p.name)
    bind_dict = {p: float(qaoa_x0[i]) for i, p in enumerate(sorted_params)}
    qc_bound = routed.assign_parameters(bind_dict)
    circuit, theta0 = from_qiskit(qc_bound, fuse_zz_swap=True, rank=rank, device=DEVICE)
    P = theta0.shape[0]
    print(f'  TREV: {P} params, rank={rank}')

    results = {}
    for method_name, measure_method, shots in [
        ('EC',     MeasureMethod.EFFICIENT_CONTRACTION, 0),
        ('RSS-1k', MeasureMethod.RIGHT_SUFFIX_SAMPLING, 1000),
    ]:
        res = run_vqe(circuit, theta0, hamil, measure_method, shots,
                      n_iters, lr, method_name)
        results[method_name] = res

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Decode accuracy
    qperm = circuit.qubit_perm
    print(f'\n  {"Method":<12s} {"Wall(s)":<10s} {"Best E":<12s} {"Best Acc":<10s}')
    print(f'  {"-"*44}')
    for method_name, res in results.items():
        best_ratio = 0.0
        for bits in res['best_results']:
            if isinstance(bits, str): bits = [int(b) for b in bits]
            elif hasattr(bits, 'tolist'): bits = bits.tolist()
            else: bits = list(bits)
            feasible, tour = decode_tsp_bitstring(bits, nc, qperm)
            if feasible and tour is not None:
                cost = compute_tour_cost(tour, dist)
                ratio = opt_cost / cost if cost > 0 else 0.0
                best_ratio = max(best_ratio, ratio)
        print(f'  {method_name:<12s} {res["wall_time"]:<10.1f} '
              f'{min(res["exp_values"]):<12.4f} {best_ratio:<10.3f}')

    return results


# ═══════════════════════════════════════════════════════════════
# Test 2: PUCCD Chemistry (H2 — has X/Y terms)
# ═══════════════════════════════════════════════════════════════

def run_puccd_benchmark(molecule='H2', rank=8, n_iters=50, lr=5e-3, seed=0):
    from qiskit import transpile as qk_transpile
    from qiskit.transpiler import CouplingMap
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper, InterleavedQubitMapper
    from qiskit_nature.second_q.circuit.library import PUCCD, HartreeFock
    from TREV.transpile import from_qiskit
    import scipy.sparse.linalg as spla

    CHEM_BASIS_GATES = ['cx', 'swap', 'u']
    mapper = InterleavedQubitMapper(JordanWignerMapper())

    MOLECULES = {
        'H2':  {'atom': 'H 0 0 0; H 0 0 0.735',   'charge': 0, 'spin': 0},
        'H4':  {'atom': 'H 0 0 0; H 0 0 0.735; H 0 0 1.47; H 0 0 2.205',
                'charge': 0, 'spin': 0},
        'LiH': {'atom': 'Li 0 0 0; H 0 0 1.6',     'charge': 0, 'spin': 0},
    }

    print(f'\n{"="*60}')
    print(f'PUCCD Chemistry: {molecule}, rank={rank}, seed={seed}')
    print(f'{"="*60}')

    mol = MOLECULES[molecule]
    driver = PySCFDriver(atom=mol['atom'], charge=mol['charge'],
                         spin=mol['spin'], basis='sto3g')
    problem = driver.run()
    num_spatial = problem.num_spatial_orbitals
    num_particles = problem.num_particles
    N = 2 * num_spatial
    nuc_rep = problem.nuclear_repulsion_energy
    second_q = problem.hamiltonian.second_q_op()
    qubitOp = mapper.map(second_q)

    mat = qubitOp.to_matrix(sparse=True)
    eigvals, _ = spla.eigsh(mat, k=1, which='SA')
    exact_energy = float(eigvals[0]) + nuc_rep

    qubitOp_n, scale = normalize_ising(qubitOp)
    hamil = build_trev_hamiltonian(qubitOp_n)

    # Count Pauli types
    op_tensor = hamil.get_pauli_op_tensor()
    n_x = (op_tensor == 1).any(dim=1).sum().item()
    n_y = (op_tensor == 2).any(dim=1).sum().item()
    n_z = (op_tensor == 3).any(dim=1).sum().item()
    n_i_only = ((op_tensor == 0).all(dim=1)).sum().item()
    print(f'  {molecule}: {N} qubits, E_exact={exact_energy:.6f} Ha')
    print(f'  Hamiltonian: {len(hamil.coefficients)} terms')
    print(f'  Terms with X: {n_x}, Y: {n_y}, Z-only: {n_z}, all-I: {n_i_only}')

    # Build PUCCD ansatz
    hf = HartreeFock(num_spatial, num_particles, mapper)
    ansatz = PUCCD(num_spatial, num_particles, mapper, initial_state=hf)
    optimized = qk_transpile(ansatz, optimization_level=1, basis_gates=CHEM_BASIS_GATES)
    K_ansatz = ansatz.num_parameters
    print(f'  PUCCD: {K_ansatz} params, {optimized.size()} gates')

    # Route for ring topology
    cm = CouplingMap.from_ring(N)
    routed = qk_transpile(optimized, coupling_map=cm, optimization_level=1,
                          basis_gates=CHEM_BASIS_GATES, seed_transpiler=seed)
    print(f'  Routed circuit: depth={routed.depth()}')

    # Bind random initial params and convert
    gen = torch.Generator().manual_seed(seed)
    qaoa_x0 = 0.01 * torch.randn(K_ansatz, generator=gen)
    sorted_params = sorted(routed.parameters, key=lambda p: p.name)
    bind_dict = {p: float(qaoa_x0[i]) for i, p in enumerate(sorted_params)}
    qc_bound = routed.assign_parameters(bind_dict)
    circuit, theta0 = from_qiskit(qc_bound, fuse_zz_swap=False, rank=rank, device=DEVICE)
    P = theta0.shape[0]
    print(f'  TREV: {P} params, rank={rank}')

    results = {}
    for method_name, measure_method, shots in [
        ('EC',     MeasureMethod.EFFICIENT_CONTRACTION, 0),
        ('RSS-1k', MeasureMethod.RIGHT_SUFFIX_SAMPLING, 1000),
    ]:
        res = run_vqe(circuit, theta0, hamil, measure_method, shots,
                      n_iters, lr, method_name)
        results[method_name] = res

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f'\n  Exact energy: {exact_energy:.6f} Ha')
    print(f'  {"Method":<12s} {"Wall(s)":<10s} {"Best E(norm)":<14s} {"Best E(Ha)":<12s}')
    print(f'  {"-"*48}')
    for method_name, res in results.items():
        best_e_norm = min(res['exp_values'])
        best_e_ha = best_e_norm * scale + nuc_rep
        print(f'  {method_name:<12s} {res["wall_time"]:<10.1f} '
              f'{best_e_norm:<14.6f} {best_e_ha:<12.6f}')

    return results, exact_energy, scale, nuc_rep


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark EC vs RSS')
    parser.add_argument('--skip-puccd', action='store_true',
                        help='Skip PUCCD test (no qiskit-nature needed)')
    parser.add_argument('--n-iters', type=int, default=50)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--nc', type=int, default=3, help='TSP cities')
    parser.add_argument('--molecule', type=str, default='H2',
                        choices=['H2', 'H4', 'LiH'])
    args = parser.parse_args()

    print(f'Device: {DEVICE}')
    if DEVICE == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name()}')

    # Test 1: TSP QAOA ring
    tsp_results = run_tsp_benchmark(
        nc=args.nc, reps=2, rank=args.rank,
        n_iters=args.n_iters, lr=5e-3, seed=0,
    )

    # Test 2: PUCCD Chemistry
    if not args.skip_puccd:
        puccd_results, exact_e, scale, nuc_rep = run_puccd_benchmark(
            molecule=args.molecule, rank=args.rank,
            n_iters=args.n_iters, lr=5e-3, seed=0,
        )

    print(f'\n{"="*60}')
    print('Done!')
