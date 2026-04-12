"""
Adjoint differentiation for tensor ring VQE.

Hybrid approach:
- 1q gates with NO subsequent 2q gate on same qubit: analytical derivative
  + environment contraction (exact, no SVD backward, O(N*T*chi^4) per param)
- All other gates: parameter-shift fallback (exact, O(N*T*chi^5) per param)

Uses build_tensor (standard SVD) for ALL forward passes — no diff_svd needed.
"""
import torch

from ...circuit import Circuit
from ...hamiltonian.hamiltonian import Hamiltonian
from ...measure.enums import MeasureMethod
from ...gates.contraction import _apply_single_qubit_gate
from ...gates.parameter_gates import (
    ParameterOneQubitGate,
    ParameterMultiOneQubitGate,
    ParameterTwoQubitGate,
)
from ...gates.non_parameter_gates import (
    NonParameterOneQubitGate,
    NonParameterTwoQubitsGate,
)
from .gradient import Gradient


# Pauli matrices for analytical gate derivatives
_SIGMA = {
    'rx': lambda dev: torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat, device=dev),
    'ry': lambda dev: torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat, device=dev),
    'rz': lambda dev: torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=dev),
}


def _get_gate_generator(gate, device):
    """Return the Pauli generator for a parametric 1q gate, or None."""
    name = gate.matrix_fun.__name__.lower() if hasattr(gate.matrix_fun, '__name__') else ''
    for key, sigma_fn in _SIGMA.items():
        if key in name:
            return sigma_fn(device)
    return None


class AdjointGradient(Gradient):
    """
    Adjoint gradient: analytical derivatives + environment contraction.

    For 1q gates (last on qubit before any 2q gate): uses dG/dθ = -(i/2)σG
    and contracts the derivative core with the Hamiltonian environment.

    For all other gates: falls back to parameter-shift (exact).
    """

    def __init__(self, measure_method: MeasureMethod = MeasureMethod.EFFICIENT_CONTRACTION):
        super().__init__(measure_method)
        self._verbose = True
        self._printed = False
        self.last_exp_value = None
        self.last_tensor = None

    def run(self, theta: torch.Tensor, circuit: Circuit, hamiltonian: Hamiltonian):
        device = circuit.device
        N = circuit.num_qubit
        P = theta.numel()

        if circuit.qubit_perm is not None:
            hamiltonian = hamiltonian.permuted(circuit.qubit_perm)

        # Build tensor (standard, no autograd)
        tensor = circuit.build_tensor(theta)
        self.last_tensor = tensor

        from ...measure.efficient_contraction import expectation_value_batch as ev_exact
        ev = ev_exact(tensor, hamiltonian, device=device)
        self.last_exp_value = float(ev.real) if hasattr(ev, 'real') else float(ev)

        if self._verbose and not self._printed:
            print(f"[TREV] AdjointGradient: params={P}, device={device}\n", flush=True)
            self._printed = True

        grad = torch.zeros(P, device=device)

        theta_f = theta.detach().to(device)
        gates = circuit.gates
        shift = torch.pi / 2

        # Classify gates: adjoint (1q with no subsequent 2q) or param-shift
        for gi, gate in enumerate(gates):
            if not isinstance(gate, (ParameterOneQubitGate, ParameterMultiOneQubitGate, ParameterTwoQubitGate)):
                continue

            if isinstance(gate, ParameterTwoQubitGate):
                # Always param-shift for 2q gates
                tidx = gate.theta_index
                tp = theta_f.clone(); tp[tidx] += shift
                tm = theta_f.clone(); tm[tidx] -= shift
                ev_p = ev_exact(circuit.build_tensor(tp), hamiltonian, device=device)
                ev_m = ev_exact(circuit.build_tensor(tm), hamiltonian, device=device)
                grad[tidx] = float((ev_p - ev_m).real) / 2
                continue

            if isinstance(gate, ParameterMultiOneQubitGate):
                # Param-shift for multi-param gates
                for tidx in gate.theta_indices:
                    tp = theta_f.clone(); tp[tidx] += shift
                    tm = theta_f.clone(); tm[tidx] -= shift
                    ev_p = ev_exact(circuit.build_tensor(tp), hamiltonian, device=device)
                    ev_m = ev_exact(circuit.build_tensor(tm), hamiltonian, device=device)
                    grad[tidx] = float((ev_p - ev_m).real) / 2
                continue

            # 1-qubit gate: check if we can use adjoint
            q = gate.qubit
            tidx = gate.theta_index
            gen = _get_gate_generator(gate, device)

            # Collect subsequent 1q gates on same qubit (stop at 2q gate)
            subsequent_mats = []
            can_adjoint = gen is not None
            for lg in gates[gi + 1:]:
                if isinstance(lg, (ParameterOneQubitGate, ParameterMultiOneQubitGate)):
                    if lg.qubit == q:
                        if isinstance(lg, ParameterOneQubitGate):
                            subsequent_mats.append(lg.matrix_fun(theta_f[lg.theta_index], device))
                        else:
                            params = torch.stack([theta_f[i] for i in lg.theta_indices])
                            subsequent_mats.append(lg.matrix_fun(params, device))
                elif isinstance(lg, NonParameterOneQubitGate):
                    if lg.qubit == q:
                        subsequent_mats.append(lg.matrix_fun(None, device))
                elif hasattr(lg, 'qubits'):
                    if q in lg.qubits:
                        can_adjoint = False
                        break

            if not can_adjoint:
                # Fallback: param-shift
                tp = theta_f.clone(); tp[tidx] += shift
                tm = theta_f.clone(); tm[tidx] -= shift
                ev_p = ev_exact(circuit.build_tensor(tp), hamiltonian, device=device)
                ev_m = ev_exact(circuit.build_tensor(tm), hamiltonian, device=device)
                grad[tidx] = float((ev_p - ev_m).real) / 2
                continue

            # Adjoint: compute derivative core from build_tensor's FINAL tensor
            # by un-applying subsequent gates, inserting the derivative, re-applying
            #
            # derivative_core = G_last @ ... @ G_{k+1} @ dG_k/dθ @ G_k^{-1} @ ... @ G_last^{-1} @ tensor[q]
            #
            # Simplified: let M = G_last @ ... @ G_{k+1} (subsequent gates matrix)
            # then: derivative_core = M @ (-i/2) σ @ G_k @ G_k^H @ M^H @ tensor[q]
            #                       = M @ (-i/2) σ @ M^H @ tensor[q]
            #                       = (-i/2) (M σ M^H) @ tensor[q]
            gate_mat = gate.matrix_fun(theta_f[tidx], device)

            # Build M = product of subsequent gate matrices (in order)
            M = torch.eye(2, dtype=torch.cfloat, device=device)
            for smat in subsequent_mats:
                M = smat @ M

            # derivative_core = (-i/2) * M @ σ @ M^H @ tensor[q]
            effective_gen = M @ gen @ M.mH  # rotated generator
            dmat_effective = (-1j / 2) * effective_gen
            core_deriv = _apply_single_qubit_gate(dmat_effective, tensor[q])

            # Contract with environment
            deriv_ev = _contract_with_derivative_core(
                tensor, core_deriv, q, hamiltonian, device)
            grad[tidx] = 2 * deriv_ev.real

        return grad


def _apply_gate_to_cores(gate, cores, theta, device, N):
    """Apply a single gate to cores list (in-place)."""
    from ...gates.contraction import _apply_double_qubit_gate
    from ...gates.non_parameter_gates import _swap_gate_matrix

    if isinstance(gate, ParameterOneQubitGate):
        mat = gate.matrix_fun(theta[gate.theta_index], device)
        cores[gate.qubit] = _apply_single_qubit_gate(mat, cores[gate.qubit])
    elif isinstance(gate, ParameterMultiOneQubitGate):
        params = torch.stack([theta[i] for i in gate.theta_indices])
        mat = gate.matrix_fun(params, device)
        cores[gate.qubit] = _apply_single_qubit_gate(mat, cores[gate.qubit])
    elif isinstance(gate, NonParameterOneQubitGate):
        mat = gate.matrix_fun(None, device)
        cores[gate.qubit] = _apply_single_qubit_gate(mat, cores[gate.qubit])
    elif isinstance(gate, (ParameterTwoQubitGate, NonParameterTwoQubitsGate)):
        q0, q1 = gate.qubits
        if isinstance(gate, ParameterTwoQubitGate):
            mat = gate.matrix_fun(theta[gate.theta_index], device)
        else:
            mat = gate.matrix_fun(device=device)
        lo, hi = min(q0, q1), max(q0, q1)
        is_wrap = (lo == 0 and hi == N - 1)
        if is_wrap:
            if q0 == N - 1 and q1 == 0:
                cores[N-1], cores[0] = _apply_double_qubit_gate(mat, (cores[N-1], cores[0]))
            else:
                cores[N-1], cores[0] = _apply_double_qubit_gate(_swap_gate_matrix(mat), (cores[N-1], cores[0]))
        elif q0 < q1:
            cores[q0], cores[q1] = _apply_double_qubit_gate(mat, (cores[q0], cores[q1]))
        else:
            cores[q1], cores[q0] = _apply_double_qubit_gate(_swap_gate_matrix(mat), (cores[q1], cores[q0]))


def _contract_with_derivative_core(tensor, core_deriv, q, hamiltonian, device):
    """
    Contract ⟨ψ'|H|ψ⟩ where ψ' has core_deriv at site q, ψ is original.

    Uses the transfer matrix approach: at site q, the bra uses core_deriv
    and the ket uses tensor[q]. All other sites use tensor[i] for both.
    """
    N = tensor.shape[0]
    dtype = tensor.dtype

    op_tensor = hamiltonian.get_pauli_op_tensor().to(device)
    coeffs = hamiltonian.coefficients
    T = op_tensor.shape[0]

    Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    Y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    pauli_mats = [None, X, Y, Z]  # indexed by op: 0=I, 1=X, 2=Y, 3=Z

    total = torch.zeros((), dtype=dtype, device=device)

    for t in range(T):
        coef = coeffs[t]
        ten = None
        for i in range(N):
            if i == q:
                curr_bra = core_deriv.permute(0, 2, 1)
                curr_ket = tensor[i].permute(0, 2, 1)
            else:
                curr_bra = tensor[i].permute(0, 2, 1)
                curr_ket = curr_bra

            op = op_tensor[t, i].item()
            if op == 0:
                AO_ket = curr_ket
            else:
                AO_ket = torch.einsum('ldr,dk->lkr', curr_ket, pauli_mats[op])

            E = torch.tensordot(curr_bra.conj(), AO_ket, ([1], [1])).permute(0, 2, 1, 3)

            if ten is None:
                ten = E
            else:
                ten = torch.tensordot(ten, E, dims=([2, 3], [0, 1]))

        total = total + coef * torch.einsum('ikik->', ten)

    return total
