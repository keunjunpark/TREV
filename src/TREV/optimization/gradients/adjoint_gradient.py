"""
Adjoint differentiation for tensor ring VQE.

For each parameter θ_k in gate G_k on qubit q:
  dE/dθ_k = 2 Re(Tr(... E_{q-1} · E'_q · E_{q+1} ...))

where E'_q is the transfer matrix built from the core with
dG_k/dθ_k applied instead of G_k, and the trace uses the
standard left/right environment contraction.

Key advantage: NO SVD backward. Uses the same forward contraction
as parameter-shift but replaces f(θ±shift) evaluations with
analytical gate derivatives, requiring only 1 forward pass +
P derivative gate evaluations (vs 2P full forward passes).
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
from .gradient import Gradient


class AdjointGradient(Gradient):
    """
    Adjoint differentiation: analytical gate derivatives + environment contraction.

    For each parameter θ_k, computes the gradient by:
    1. Building the full tensor ring (standard build_tensor, no autograd)
    2. Computing left/right environments via efficient contraction
    3. Inserting the derivative gate dG/dθ_k and contracting

    No SVD backward needed — all operations are forward contractions.
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
        chi = circuit.rank
        P = theta.numel()

        if circuit.qubit_perm is not None:
            hamiltonian = hamiltonian.permuted(circuit.qubit_perm)

        if self._verbose and not self._printed:
            print(f"[TREV] AdjointGradient: params={P}, device={device}\n", flush=True)
            self._printed = True

        # Step 1: Build the tensor ring (standard forward, no autograd)
        tensor = circuit.build_tensor(theta)  # (N, chi, chi, 2)
        self.last_tensor = tensor

        # Compute expectation value (cache it)
        from ...measure.efficient_contraction import expectation_value_batch as ev_exact
        ev = ev_exact(tensor, hamiltonian, device=device)
        self.last_exp_value = float(ev.real) if hasattr(ev, 'real') else float(ev)

        # Step 2: For each Hamiltonian term, precompute transfer matrices
        # E_i = conj(A0_i) ⊗ A0_i + conj(A1_i) ⊗ A1_i  for identity
        # For Z: E_i = conj(A0_i) ⊗ A0_i - conj(A1_i) ⊗ A1_i
        # We need the full transfer matrix product for the environment.

        # Step 3: For each parameter, compute gradient analytically
        grad = torch.zeros(P, device=device)

        # Build intermediate core states and collect subsequent 1q gates per qubit
        theta_f = theta.detach().to(device)
        cores = [torch.zeros(chi, chi, 2, dtype=torch.cfloat, device=device) for _ in range(N)]
        for i in range(N):
            cores[i][0, 0, 0] = 1.0

        from ...gates.non_parameter_gates import NonParameterOneQubitGate

        # First pass: record all gates with pre-gate snapshots
        gate_info = []
        for gi, gate in enumerate(circuit.gates):
            if isinstance(gate, (ParameterOneQubitGate, ParameterMultiOneQubitGate)):
                snap = cores[gate.qubit].clone()
                gate_info.append((gi, gate, snap))
            elif isinstance(gate, ParameterTwoQubitGate):
                gate_info.append((gi, gate, None))  # use param-shift

            _apply_gate_to_cores(gate, cores, theta_f, device, N)

        # For each 1q parametric gate, collect subsequent 1q gate matrices
        # on the same qubit to propagate the derivative core forward
        all_gates = circuit.gates
        for info_idx, (gi, gate, snap) in enumerate(gate_info):
            if not isinstance(gate, (ParameterOneQubitGate, ParameterMultiOneQubitGate)):
                continue

            q = gate.qubit

            # Collect subsequent 1q gates on same qubit (until next 2q gate on q)
            subsequent_mats = []
            for later_gi in range(gi + 1, len(all_gates)):
                later_gate = all_gates[later_gi]
                if isinstance(later_gate, (ParameterOneQubitGate, ParameterMultiOneQubitGate)):
                    if later_gate.qubit == q:
                        if isinstance(later_gate, ParameterOneQubitGate):
                            subsequent_mats.append(later_gate.matrix_fun(theta_f[later_gate.theta_index], device))
                        else:
                            params = torch.stack([theta_f[i] for i in later_gate.theta_indices])
                            subsequent_mats.append(later_gate.matrix_fun(params, device))
                elif isinstance(later_gate, NonParameterOneQubitGate):
                    if later_gate.qubit == q:
                        subsequent_mats.append(later_gate.matrix_fun(None, device))
                elif hasattr(later_gate, 'qubits'):
                    # 2q gate: if it touches qubit q, stop (SVD changes the core)
                    if q in later_gate.qubits:
                        break

            gate_info[info_idx] = (gi, gate, snap, subsequent_mats)

        # Now compute gradient for each parametric gate
        for entry in gate_info:
            if len(entry) == 3:
                gi, gate, _ = entry
            else:
                gi, gate, snap, subsequent_mats = entry

            if isinstance(gate, ParameterOneQubitGate):
                q = gate.qubit
                tidx = gate.theta_index

                # dG/dθ via FD on gate matrix
                eps = 1e-5
                mat_p = gate.matrix_fun(theta_f[tidx] + eps, device)
                mat_m = gate.matrix_fun(theta_f[tidx] - eps, device)
                dmat = (mat_p - mat_m) / (2 * eps)

                # Apply derivative gate, then subsequent gates
                core_deriv = _apply_single_qubit_gate(dmat, snap)
                for smat in subsequent_mats:
                    core_deriv = _apply_single_qubit_gate(smat, core_deriv)

                # Contract with environment
                deriv_ev = _contract_with_derivative_core(
                    tensor, core_deriv, q, hamiltonian, device)
                grad[tidx] += 2 * deriv_ev.real

            elif isinstance(gate, ParameterMultiOneQubitGate):
                q = gate.qubit
                for pi, tidx in enumerate(gate.theta_indices):
                    eps = 1e-5
                    params_p = torch.stack([theta_f[i] for i in gate.theta_indices])
                    params_m = params_p.clone()
                    params_p[pi] += eps; params_m[pi] -= eps
                    dmat = (gate.matrix_fun(params_p, device) - gate.matrix_fun(params_m, device)) / (2*eps)
                    core_deriv = _apply_single_qubit_gate(dmat, snap)
                    for smat in subsequent_mats:
                        core_deriv = _apply_single_qubit_gate(smat, core_deriv)
                    deriv_ev = _contract_with_derivative_core(
                        tensor, core_deriv, q, hamiltonian, device)
                    grad[tidx] += 2 * deriv_ev.real

            elif isinstance(gate, ParameterTwoQubitGate):
                # Use parameter-shift for 2q gates (no SVD backward)
                tidx = gate.theta_index
                shift = torch.pi / 2
                theta_p = theta_f.clone(); theta_p[tidx] += shift
                theta_m = theta_f.clone(); theta_m[tidx] -= shift
                t_p = circuit.build_tensor(theta_p)
                t_m = circuit.build_tensor(theta_m)
                ev_p = ev_exact(t_p, hamiltonian, device=device)
                ev_m = ev_exact(t_m, hamiltonian, device=device)
                grad[tidx] = float((ev_p - ev_m).real) / 2

        return grad


def _apply_gate_to_cores(gate, cores, theta, device, N):
    """Apply a single gate to the cores list (in-place modification)."""
    from ...gates.non_parameter_gates import NonParameterOneQubitGate, NonParameterTwoQubitsGate
    from ...gates.contraction import _apply_single_qubit_gate, _apply_double_qubit_gate
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
    Contract the tensor ring with core_deriv substituted at site q.

    Computes: ⟨ψ'|H|ψ⟩ where ψ' has core_deriv at site q and
    ψ has the original tensor at all sites.

    This is the "environment contraction": contract all transfer matrices
    E_i = conj(tensor[i]) ⊗ tensor[i] for i ≠ q, and at site q use
    E'_q = conj(core_deriv) ⊗ tensor[q] (cross term).
    """
    N = tensor.shape[0]
    chi = tensor.shape[1]
    dtype = tensor.dtype

    paulis = hamiltonian.get_bool_pauli_tensor().to(device)  # (T, N) bool
    coeffs = hamiltonian.coefficients
    Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)

    total = torch.zeros((), dtype=dtype, device=device)

    for t in range(len(hamiltonian.paulis)):
        coef = coeffs[t]
        ten = None
        for i in range(N):
            if i == q:
                # Cross term: conj(core_deriv) ⊗ O ⊗ tensor[q]
                curr_bra = core_deriv.permute(0, 2, 1)  # bra (derivative)
                curr_ket = tensor[i].permute(0, 2, 1)    # ket (original)
            else:
                curr_bra = tensor[i].permute(0, 2, 1)
                curr_ket = tensor[i].permute(0, 2, 1)

            if paulis[t, i]:
                AO_ket = torch.einsum('ldr,dk->lkr', curr_ket, Z)
            else:
                AO_ket = curr_ket

            E = torch.tensordot(curr_bra.conj(), AO_ket, ([1], [1])).permute(0, 2, 1, 3)

            if ten is None:
                ten = E
            else:
                ten = torch.tensordot(ten, E, dims=([2, 3], [0, 1]))

        total = total + coef * torch.einsum('ikik->', ten)

    return total
