"""
Benchmark: autograd (backprop) with custom complex SVD vs parameter-shift.
Run: python bench_autograd.py
"""
import time, torch
from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.efficient_contraction import expectation_value_batch as ev_exact
from TREV.gates.contraction import _apply_single_qubit_gate
from TREV.gates.differentiable_svd import complex_svd
from TREV.optimization.gradients.batch_parameter_shift import (
    batch_gradient, batch_gradient_cached,
)
from TREV.measure.enums import MeasureMethod

device = 'cuda'
torch.manual_seed(42)


def _apply_2q_gate_diffsvd(gate_matrix, qu0, qu1):
    """2-qubit gate using custom differentiable SVD."""
    chi1 = qu0.shape[0]
    chi3 = qu1.shape[1]
    mps = torch.tensordot(qu0, qu1, ([1], [0]))
    mps = torch.moveaxis(mps, 2, 1)
    g = gate_matrix.reshape(2, 2, 2, 2)
    mps = torch.tensordot(g, mps, ([2, 3], [2, 3]))
    mps = torch.moveaxis(mps, 1, 2).reshape(chi1 * 2, chi3 * 2)

    # Custom SVD with complex-safe backward
    U, S, Vh = complex_svd(mps, full_matrices=False)

    x = U[:, :chi1]
    sc = S[:chi1].unsqueeze(0).to(mps.dtype)
    y = Vh[:chi3, :]

    qu0_new = (x * sc).reshape(2, chi1, chi1)
    qu1_new = y.reshape(chi3, 2, chi3)
    qu0_new = torch.moveaxis(qu0_new, 0, 2)
    qu1_new = torch.moveaxis(qu1_new, 1, 2)
    return qu0_new, qu1_new


def build_tensor_autograd(theta, circuit):
    """Build tensor ring with autograd support (no in-place ops, custom SVD)."""
    N = circuit.num_qubit
    chi = circuit.rank

    from TREV.gates.parameter_gates import ParameterOneQubitGate
    from TREV.gates.non_parameter_gates import NonParameterOneQubitGate, NonParameterTwoQubitsGate

    cores = []
    for i in range(N):
        c = torch.zeros(chi, chi, 2, dtype=torch.cfloat, device=device)
        c[0, 0, 0] = 1.0
        cores.append(c)

    for gate in circuit.gates:
        if isinstance(gate, ParameterOneQubitGate):
            mat = gate.matrix_fun(theta[gate.theta_index], device)
            cores[gate.qubit] = _apply_single_qubit_gate(mat, cores[gate.qubit])
        elif isinstance(gate, NonParameterOneQubitGate):
            mat = gate.matrix_fun(None, device)
            cores[gate.qubit] = _apply_single_qubit_gate(mat, cores[gate.qubit])
        elif isinstance(gate, NonParameterTwoQubitsGate):
            q0, q1 = gate.qubits
            mat = gate.matrix_fun(device=device)
            cores[q0], cores[q1] = _apply_2q_gate_diffsvd(mat, cores[q0], cores[q1])

    return torch.stack(cores, dim=0)


def autograd_gradient(theta, circuit, hamiltonian):
    """Gradient via backprop through custom SVD + contraction."""
    theta_ad = theta.detach().clone().requires_grad_(True)
    tensor = build_tensor_autograd(theta_ad, circuit)

    N = tensor.shape[0]
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=device)
    paulis = hamiltonian.get_bool_pauli_tensor().to(device)

    total = torch.zeros((), dtype=torch.cfloat, device=device)
    for t_idx in range(len(hamiltonian.paulis)):
        coef = hamiltonian.coefficients[t_idx]
        ten = None
        for i in range(N):
            curr = tensor[i].permute(0, 2, 1)
            if paulis[t_idx, i]:
                AO = torch.einsum('ldr,dk->lkr', curr, Z)
            else:
                AO = curr
            E = torch.tensordot(curr.conj(), AO, ([1], [1])).permute(0, 2, 1, 3)
            ten = E if ten is None else torch.tensordot(ten, E, dims=([2, 3], [0, 1]))
        total = total + coef * torch.einsum('ikik->', ten)

    total.real.backward()
    return theta_ad.grad.float()


# ── Benchmark ──
def bench(fn, warmup=2, repeats=5):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    import numpy as np
    return np.median(times)


print("=" * 70)
print("Autograd (custom SVD backprop) vs Parameter-Shift gradient")
print("=" * 70)

for N, chi, L in [(4, 4, 2), (8, 4, 2), (8, 10, 2), (12, 4, 2)]:
    c = Circuit(num_qubit=N, rank=chi, device=device)
    for i in range(N): c.h(i)
    for _ in range(L):
        for i in range(N): c.cx(i, (i + 1) % N)
        for i in range(N): c.ry(i); c.rz(i)
    h = Hamiltonian(num_qubits=N)
    for i in range(N):
        j = (i + 1) % N; p = ['I'] * N; p[i] = 'Z'; p[j] = 'Z'
        h.add_pauli(''.join(p), 0.5)
    h.add_pauli('I' * N, 1.0)

    P = c.params_size
    theta = torch.randn(P, device=device)

    # Correctness
    g_ps = batch_gradient(theta, c, h, 8, 0, 0.5 * 3.14159, 1, 0, False, MeasureMethod.EFFICIENT_CONTRACTION)
    try:
        g_ad = autograd_gradient(theta, c, h)
        cos = torch.nn.functional.cosine_similarity(g_ps.unsqueeze(0), g_ad.unsqueeze(0)).item()
        diff = (g_ps - g_ad).abs().max().item()
    except Exception as e:
        print(f"  N={N:>2} chi={chi:>3} L={L} P={P:>3}: FAILED — {e}")
        continue

    # Timing
    ms_ps = bench(lambda: batch_gradient_cached(theta, c, h, 8, 0, 0.5*3.14159, MeasureMethod.EFFICIENT_CONTRACTION))
    ms_ad = bench(lambda: autograd_gradient(theta, c, h))

    speedup = ms_ps / ms_ad
    ok = cos > 0.95
    print(f"  N={N:>2} chi={chi:>3} L={L} P={P:>3}: "
          f"ps={ms_ps:>6.0f}ms  ad={ms_ad:>6.0f}ms  "
          f"speedup={speedup:>5.1f}x  cos={cos:.4f}  diff={diff:.2e}  {'PASS' if ok else 'FAIL'}")
