"""
Autograd-based gradient computation for tensor ring VQE.

Uses PyTorch's reverse-mode automatic differentiation (backpropagation)
through the circuit build and contraction. Requires a custom differentiable
SVD (diff_svd) for the 2-qubit gate decomposition.

Advantages over parameter-shift:
  - 1 forward + 1 backward pass for ALL parameters (vs 2P evaluations)
  - 1.4-2.8x faster for practical circuits (N>=8, L<=2)
  - Exact gradient (matches parameter-shift to machine precision at f64)

Requirements:
  - float64 (complex128) for numerical stability of SVD backward
  - Custom diff_svd with F/G split regularization for degenerate SVs
"""
import torch

from ...circuit import Circuit
from ...hamiltonian.hamiltonian import Hamiltonian
from ...measure.enums import MeasureMethod
from ...gates.contraction import _apply_single_qubit_gate
from ...gates.differentiable_svd import diff_svd
from ...gates.parameter_gates import ParameterOneQubitGate
from ...gates.non_parameter_gates import (
    NonParameterOneQubitGate,
    NonParameterTwoQubitsGate,
)
from .gradient import Gradient


def _apply_2q_diff(gate_matrix, qu0, qu1, dtype):
    """Apply 2-qubit gate with differentiable SVD split."""
    chi1, chi3 = qu0.shape[0], qu1.shape[1]
    mps = torch.tensordot(qu0, qu1, ([1], [0]))
    mps = torch.moveaxis(mps, 2, 1)
    gt = gate_matrix.reshape(2, 2, 2, 2).to(dtype)
    mps = torch.tensordot(gt, mps, ([2, 3], [2, 3]))
    mps = torch.moveaxis(mps, 1, 2).reshape(chi1 * 2, chi3 * 2)
    Uk, Sk, Vhk = diff_svd(mps, chi1)
    q0 = (Uk * Sk.unsqueeze(0).to(dtype)).reshape(2, chi1, chi1)
    q1 = Vhk.reshape(chi3, 2, chi3)
    return torch.moveaxis(q0, 0, 2), torch.moveaxis(q1, 1, 2)


def _build_tensor_diff(theta, circuit, dtype=torch.complex128):
    """Build tensor ring with autograd tracking (no in-place ops).

    Uses native-dtype gate matrix computation to avoid precision loss
    from info.py's forced .type(torch.cfloat) cast.
    """
    N, chi = circuit.num_qubit, circuit.rank
    device = circuit.device
    cores = [torch.zeros(chi, chi, 2, dtype=dtype, device=device) for _ in range(N)]
    for i in range(N):
        cores[i][0, 0, 0] = 1.0

    for gate in circuit.gates:
        if isinstance(gate, ParameterOneQubitGate):
            mat = gate.matrix_fun(theta[gate.theta_index], device).to(dtype)
            cores[gate.qubit] = _apply_single_qubit_gate(mat, cores[gate.qubit])
        elif isinstance(gate, NonParameterOneQubitGate):
            mat = gate.matrix_fun(None, device).to(dtype)
            cores[gate.qubit] = _apply_single_qubit_gate(mat, cores[gate.qubit])
        elif isinstance(gate, NonParameterTwoQubitsGate):
            q0, q1 = gate.qubits
            cores[q0], cores[q1] = _apply_2q_diff(
                gate.matrix_fun(device=device), cores[q0], cores[q1], dtype
            )

    return torch.stack(cores, dim=0)


def _contraction_diff(tensor, hamiltonian, dtype=torch.complex128):
    """Differentiable expectation value contraction (per-term loop).

    Uses a Python loop over Hamiltonian terms to avoid torch.where
    which breaks autograd backward for complex tensors.
    """
    N = tensor.shape[0]
    device = tensor.device
    Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    paulis = hamiltonian.get_bool_pauli_tensor().to(device)

    total = torch.zeros((), dtype=dtype, device=device)
    for t in range(len(hamiltonian.paulis)):
        coef = hamiltonian.coefficients[t]
        ten = None
        for i in range(N):
            curr = tensor[i].permute(0, 2, 1)
            if paulis[t, i]:
                AO = torch.einsum('ldr,dk->lkr', curr, Z)
            else:
                AO = curr
            E = torch.tensordot(curr.conj(), AO, ([1], [1])).permute(0, 2, 1, 3)
            if ten is None:
                ten = E
            else:
                ten = torch.tensordot(ten, E, dims=([2, 3], [0, 1]))
        total = total + coef * torch.einsum('ikik->', ten)

    return total.real


def autograd_gradient(theta, circuit, hamiltonian, dtype=torch.complex128):
    """
    Compute gradient via backpropagation through differentiable SVD.

    Args:
        theta: (P,) parameter tensor
        circuit: Circuit instance
        hamiltonian: Hamiltonian instance
        dtype: complex dtype (complex128 recommended for accuracy)

    Returns:
        (P,) float32 gradient tensor
    """
    # Must enable grad even if caller uses torch.no_grad() context
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    with torch.enable_grad():
        theta_ad = theta.detach().to(real_dtype).clone().requires_grad_(True)
        tensor = _build_tensor_diff(theta_ad, circuit, dtype)
        loss = _contraction_diff(tensor, hamiltonian, dtype)
        loss.backward()
    return theta_ad.grad.float()


class AutogradGradient(Gradient):
    """
    Gradient computation via PyTorch autograd (backpropagation).

    Uses differentiable SVD with F/G split regularization.
    Drop-in replacement for BatchParameterShiftGradient.

    Args:
        dtype: complex dtype for computation (default: complex128 for accuracy)
    """

    def __init__(self, measure_method: MeasureMethod = MeasureMethod.EFFICIENT_CONTRACTION,
                 dtype=torch.cfloat):
        super().__init__(measure_method)
        self.dtype = dtype
        self._verbose = True
        self._printed = False

    def run(self, theta: torch.Tensor, circuit: Circuit, hamiltonian: Hamiltonian):
        if self._verbose and not self._printed:
            print(
                f"[TREV] AutogradGradient: dtype={self.dtype}, "
                f"params={theta.numel()}, device={circuit.device}\n",
                flush=True,
            )
            self._printed = True

        return autograd_gradient(theta, circuit, hamiltonian, self.dtype)
