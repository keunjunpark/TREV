import torch
from torch import Tensor

from ...circuit import Circuit
from ...hamiltonian.hamiltonian import Hamiltonian
from ...measure.enums import MeasureMethod
from ...optimization.gradients.gradient import Gradient
from ...gates.contraction import (
    _apply_single_qubit_gate,
    _apply_single_qubit_gate_batch,
    _apply_double_qubit_gate,
    _apply_double_qubit_gate_batch,
)
from .batch_parameter_shift import _kron_contract_right, _make_eye4


def _build_tensor_ring_differentiable(theta: Tensor, circuit: Circuit) -> Tensor:
    """Build tensor ring (N, chi, chi, 2) without in-place ops, preserving gradients.

    Instead of allocating a big tensor and writing into it with
    ``tensor[qubit] = ...``, we maintain a Python list of per-site tensors
    and ``torch.stack`` at the end.
    """
    N = circuit.num_qubit
    chi = circuit.rank
    device = circuit.device

    # Initialize each site as |0><0| ⊗ I_chi  (same as tensor[:, 0, 0, 0] = 1)
    sites = []
    for _ in range(N):
        t = torch.zeros((chi, chi, 2), dtype=torch.cfloat, device=device)
        t[0, 0, 0] = 1.0
        sites.append(t)

    for gate in circuit.gates:
        if gate.has_parameter():
            # ParameterOneQubitGate
            qubit = gate.qubit
            matrix = gate.matrix_fun(theta[gate.theta_index], gate.device)
            sites[qubit] = _apply_single_qubit_gate(matrix, sites[qubit])
        else:
            if hasattr(gate, 'qubits'):
                # NonParameterTwoQubitsGate
                q0, q1 = gate.qubits
                matrix = gate.matrix_fun(None, gate.device)
                sites[q0], sites[q1] = _apply_double_qubit_gate(
                    matrix, (sites[q0], sites[q1])
                )
            else:
                # NonParameterOneQubitGate
                qubit = gate.qubit
                matrix = gate.matrix_fun(None, gate.device)
                sites[qubit] = _apply_single_qubit_gate(matrix, sites[qubit])

    return torch.stack(sites, dim=0)  # (N, chi, chi, 2)


def _expectation_differentiable(theta: Tensor, circuit: Circuit, hamiltonian: Hamiltonian) -> Tensor:
    """Differentiable exact expectation value <psi(theta)|H|psi(theta)>.

    Port of ``expectation_value_batch_efficient_contraction`` for B=1,
    without ``@torch.no_grad()`` and with no in-place accumulation.
    Uses identity-chain factoring + Kronecker contraction = O(chi^5).
    """
    device = circuit.device
    ctype = torch.complex64

    ring = _build_tensor_ring_differentiable(theta, circuit)  # (N, chi, chi, 2)
    N, chi = ring.shape[0], ring.shape[1]

    # Unsqueeze to add batch dim B=1: (1, N, chi, chi, 2)
    ring = ring.unsqueeze(0)
    B = 1

    # Cache per-site A0/A1
    sites = []
    for i in range(N):
        Ab = ring[:, i].to(ctype)  # (1, chi, chi, 2)
        sites.append((Ab[:, :, :, 0].contiguous(), Ab[:, :, :, 1].contiguous()))

    eye4 = _make_eye4(B, chi, ctype, device)

    # Precompute left prefix under all-identity
    L_pre = [None] * (N + 1)
    acc = eye4
    for i in range(N):
        L_pre[i] = acc
        A0_i, A1_i = sites[i]
        acc = _kron_contract_right(acc, A0_i, A1_i)
    L_pre[N] = acc

    # Precompute TRANSPOSED right suffix under all-identity
    R_suf_T = [None] * (N + 1)
    acc = eye4
    for i in range(N - 1, -1, -1):
        A0_i, A1_i = sites[i]
        acc = _kron_contract_right(acc, A0_i.mT, A1_i.mT)
        R_suf_T[i] = acc
    R_suf_T[N] = eye4

    # Hamiltonian
    paulis = hamiltonian.get_bool_pauli_tensor().to(device)  # (T, N)
    coeffs = torch.as_tensor(hamiltonian.coefficients, dtype=ctype, device=device)
    T_terms = paulis.shape[0]

    # Per-term contraction — use ``totals = totals + ...`` (not ``+=``)
    totals = torch.zeros(B, dtype=ctype, device=device)

    for t in range(T_terms):
        z_sites = torch.where(paulis[t])[0].tolist()

        if len(z_sites) == 0:
            val = coeffs[t] * (L_pre[N] * R_suf_T[N]).sum(dim=(1, 2, 3, 4))
            totals = totals + val
            continue

        s_first = z_sites[0]
        s_last = z_sites[-1]

        run = L_pre[s_first]  # no .clone() needed — we never mutate run in-place

        for i in range(s_first, s_last + 1):
            A0_i, A1_i = sites[i]
            if paulis[t, i]:
                run = _kron_contract_right(run, A0_i, A1_i, sign=-1)
            else:
                run = _kron_contract_right(run, A0_i, A1_i)

        val = coeffs[t] * (run * R_suf_T[s_last + 1]).sum(dim=(1, 2, 3, 4))
        totals = totals + val

    # Return scalar
    return totals.real.float().squeeze(0)


class AutogradGradient(Gradient):
    """Compute gradients via PyTorch autograd (1 forward + 1 backward)."""

    def __init__(self):
        super().__init__(MeasureMethod.EFFICIENT_CONTRACTION)

    def run(self, theta: Tensor, circuit: Circuit, hamiltonian: Hamiltonian) -> Tensor:
        with torch.enable_grad():
            theta_diff = theta.detach().clone().to(circuit.device).requires_grad_(True)
            exp_val = _expectation_differentiable(theta_diff, circuit, hamiltonian)
            exp_val.backward()
            return theta_diff.grad.detach()
