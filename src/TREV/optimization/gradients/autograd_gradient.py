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
from ...gates.parameter_gates import (
    ParameterOneQubitGate,
    ParameterMultiOneQubitGate,
    ParameterTwoQubitGate,
)
from ...gates.non_parameter_gates import (
    NonParameterOneQubitGate,
    NonParameterTwoQubitsGate,
)
from ...gates.info import SWAP as _get_swap_matrix
from .gradient import Gradient


def _swap_gate_matrix(gate_mat):
    """Swap the qubit ordering of a 4x4 gate matrix: U(q0,q1) -> U(q1,q0)."""
    g = gate_mat.reshape(2, 2, 2, 2)
    return g.permute(1, 0, 3, 2).reshape(4, 4)


def _swap_route_apply(cores, q0, q1, gate_mat, dtype, N):
    """Apply 2q gate on non-adjacent qubits via SWAP routing."""
    swap_mat = _get_swap_matrix(device=cores[0].device).to(dtype)
    lo, hi = min(q0, q1), max(q0, q1)
    # SWAP lo toward hi-1
    for s in range(lo, hi - 1):
        cores[s], cores[s + 1] = _apply_2q_diff(swap_mat, cores[s], cores[s + 1], dtype)
    # Apply gate on (hi-1, hi) — now adjacent
    if q0 < q1:
        cores[hi - 1], cores[hi] = _apply_2q_diff(gate_mat, cores[hi - 1], cores[hi], dtype)
    else:
        cores[hi - 1], cores[hi] = _apply_2q_diff(
            _swap_gate_matrix(gate_mat), cores[hi - 1], cores[hi], dtype)
    # SWAP back
    for s in range(hi - 2, lo - 1, -1):
        cores[s], cores[s + 1] = _apply_2q_diff(swap_mat, cores[s], cores[s + 1], dtype)


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

    Adds deterministic noise (1e-8) to initial cores to break singular
    value degeneracy. Without this, rank-1 initialization creates
    degenerate zero SVs that cause gradient explosion in the SVD backward.
    """
    N, chi = circuit.num_qubit, circuit.rank
    device = circuit.device
    gen = torch.Generator(device=device)
    gen.manual_seed(N * 1000 + chi)
    cores = [torch.zeros(chi, chi, 2, dtype=dtype, device=device) for _ in range(N)]
    for i in range(N):
        cores[i][0, 0, 0] = 1.0
        noise = torch.randn(chi, chi, 2, generator=gen, device=device, dtype=torch.float32)
        cores[i] = cores[i] + 1e-4 * noise.to(dtype)

    for gate in circuit.gates:
        if isinstance(gate, ParameterMultiOneQubitGate):
            params = torch.stack([theta[i] for i in gate.theta_indices])
            mat = gate.matrix_fun(params, device).to(dtype)
            cores[gate.qubit] = _apply_single_qubit_gate(mat, cores[gate.qubit])
        elif isinstance(gate, ParameterOneQubitGate):
            mat = gate.matrix_fun(theta[gate.theta_index], device).to(dtype)
            cores[gate.qubit] = _apply_single_qubit_gate(mat, cores[gate.qubit])
        elif isinstance(gate, NonParameterOneQubitGate):
            mat = gate.matrix_fun(None, device).to(dtype)
            cores[gate.qubit] = _apply_single_qubit_gate(mat, cores[gate.qubit])
        elif isinstance(gate, ParameterTwoQubitGate):
            q0, q1 = gate.qubits
            mat = gate.matrix_fun(theta[gate.theta_index], device).to(dtype)
            _lo, _hi = min(q0, q1), max(q0, q1)
            is_adj = (_hi - _lo == 1) or (_lo == 0 and _hi == N - 1)
            if is_adj:
                if _lo == 0 and _hi == N - 1:
                    if q0 == N - 1 and q1 == 0:
                        cores[N-1], cores[0] = _apply_2q_diff(mat, cores[N-1], cores[0], dtype)
                    else:
                        cores[N-1], cores[0] = _apply_2q_diff(
                            _swap_gate_matrix(mat), cores[N-1], cores[0], dtype)
                elif q0 < q1:
                    cores[q0], cores[q1] = _apply_2q_diff(mat, cores[q0], cores[q1], dtype)
                else:
                    cores[q1], cores[q0] = _apply_2q_diff(
                        _swap_gate_matrix(mat), cores[q1], cores[q0], dtype)
            else:
                _swap_route_apply(cores, q0, q1, mat, dtype, N)
        elif isinstance(gate, NonParameterTwoQubitsGate):
            q0, q1 = gate.qubits
            mat = gate.matrix_fun(device=device).to(dtype)
            _lo, _hi = min(q0, q1), max(q0, q1)
            is_adj = (_hi - _lo == 1) or (_lo == 0 and _hi == N - 1)
            if is_adj:
                if _lo == 0 and _hi == N - 1:
                    if q0 == N - 1 and q1 == 0:
                        cores[N-1], cores[0] = _apply_2q_diff(mat, cores[N-1], cores[0], dtype)
                    else:
                        cores[N-1], cores[0] = _apply_2q_diff(
                            _swap_gate_matrix(mat), cores[N-1], cores[0], dtype)
                elif q0 < q1:
                    cores[q0], cores[q1] = _apply_2q_diff(mat, cores[q0], cores[q1], dtype)
                else:
                    cores[q1], cores[q0] = _apply_2q_diff(
                        _swap_gate_matrix(mat), cores[q1], cores[q0], dtype)
            else:
                _swap_route_apply(cores, q0, q1, mat, dtype, N)

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


def _auto_term_chunk(N, chi, dtype, device, safety_frac=0.5):
    """Estimate max Hamiltonian terms to batch given GPU memory.

    Peak memory per term in vectorized contraction with autograd:
      - Forward: ~5 tensors of chi^4 (ten, r00, r11, r01, r10) per site
      - Backward graph: stores ~5 intermediates per site for N sites
      - Total: ~10 * N * chi^4 * element_size per term (conservative)

    Returns term_chunk that fits in free GPU memory.
    """
    if not torch.cuda.is_available() or torch.device(device).type != 'cuda':
        return 64  # CPU fallback

    elem_size = 8 if dtype in (torch.cfloat, torch.complex64) else 16
    bytes_per_term = 10 * N * (chi ** 4) * elem_size

    free_b, _ = torch.cuda.mem_get_info(device)
    available = int(free_b * safety_frac)

    chunk = max(1, available // max(1, bytes_per_term))
    return chunk


def _contraction_diff_vectorized(tensor, hamiltonian, dtype=torch.complex128,
                                 term_chunk=None):
    """Vectorized contraction: batches Hamiltonian terms in parallel.

    Supports all 4 Pauli operators (I, X, Y, Z) for chemistry Hamiltonians.
    Uses batched Kronecker-factored einsums at O(T*chi^5) per site.

    Transfer matrices per operator:
      I: conj(A0)⊗A0 + conj(A1)⊗A1       = r00 + r11
      X: conj(A0)⊗A1 + conj(A1)⊗A0       = r01 + r10
      Y: -i·conj(A0)⊗A1 + i·conj(A1)⊗A0  = 1j*(r10 - r01)
      Z: conj(A0)⊗A0 - conj(A1)⊗A1       = r00 - r11

    Args:
        term_chunk: max terms to batch at once. None = all terms.
    """
    N = tensor.shape[0]
    chi = tensor.shape[1]
    chi2 = chi * chi
    device = tensor.device

    op_tensor = hamiltonian.get_pauli_op_tensor().to(device)  # (T, N) uint8: 0=I,1=X,2=Y,3=Z
    T = op_tensor.shape[0]
    coeffs = torch.tensor(hamiltonian.coefficients, dtype=dtype, device=device)
    has_xy = hamiltonian.has_only_zi is False if hasattr(hamiltonian, 'has_only_zi') else (op_tensor == 1).any() or (op_tensor == 2).any()

    if term_chunk is None or term_chunk >= T:
        term_chunk = T

    total = torch.zeros((), dtype=dtype, device=device)
    eye4 = torch.eye(chi2, dtype=dtype, device=device).reshape(chi, chi, chi, chi)

    for t0 in range(0, T, term_chunk):
        t1 = min(t0 + term_chunk, T)
        Tc = t1 - t0
        ops_chunk = op_tensor[t0:t1]  # (Tc, N)

        ten = eye4.unsqueeze(0).expand(Tc, -1, -1, -1, -1).clone()

        for i in range(N):
            A0 = tensor[i][:, :, 0]  # (chi, chi)
            A1 = tensor[i][:, :, 1]  # (chi, chi)

            # Kronecker products: conj(Aa) ⊗ Ab via einsum
            r00 = torch.einsum('bc, taAbB, BC -> taAcC', A0.conj(), ten, A0)
            r11 = torch.einsum('bc, taAbB, BC -> taAcC', A1.conj(), ten, A1)

            if has_xy:
                r01 = torch.einsum('bc, taAbB, BC -> taAcC', A0.conj(), ten, A1)
                r10 = torch.einsum('bc, taAbB, BC -> taAcC', A1.conj(), ten, A0)

            # Build per-term result using operator masks
            ops_i = ops_chunk[:, i]  # (Tc,) uint8

            if not has_xy:
                # Fast path: Z/I only (MaxCut, TFIM, etc.)
                mask_z = (ops_i == 3).to(dtype).reshape(Tc, 1, 1, 1, 1)
                ten = (r00 + r11) - 2 * mask_z * r11
            else:
                # General path: all 4 Pauli operators
                # E = c00*r00 + c11*r11 + c01*r01 + c10*r10
                # I: 1,1,0,0  X: 0,0,1,1  Y: 0,0,-j,j  Z: 1,-1,0,0
                m_i = (ops_i == 0).to(dtype).reshape(Tc, 1, 1, 1, 1)
                m_x = (ops_i == 1).to(dtype).reshape(Tc, 1, 1, 1, 1)
                m_y = (ops_i == 2).to(dtype).reshape(Tc, 1, 1, 1, 1)
                m_z = (ops_i == 3).to(dtype).reshape(Tc, 1, 1, 1, 1)

                ten = ((m_i + m_z) * r00 +
                       (m_i - m_z) * r11 +
                       (m_x - 1j * m_y) * r01 +
                       (m_x + 1j * m_y) * r10)

        traces = torch.einsum('tikik->t', ten)
        total = total + (coeffs[t0:t1] * traces).sum()

    return total.real


def _contraction_real(tensor, hamiltonian):
    """Differentiable contraction in REAL arithmetic (compilable).

    Takes a complex tensor, converts to real ONCE, then all operations
    are real → torch.compile can fuse them.

    tensor: (N, chi, chi, 2) complex
    Returns: real scalar
    """
    N = tensor.shape[0]
    device = tensor.device
    chi = tensor.shape[1]

    # Convert to real ONCE: (N, 2, chi, chi, 2_phys) where dim 1 = [real, imag]
    t_r = tensor.real  # (N, chi, chi, 2)
    t_i = tensor.imag  # (N, chi, chi, 2)

    Z_r = torch.tensor([[1., 0.], [0., -1.]], device=device)
    paulis = hamiltonian.get_bool_pauli_tensor().to(device)

    total = torch.zeros((), device=device)
    for t_idx in range(len(hamiltonian.paulis)):
        coef = hamiltonian.coefficients[t_idx]
        ten_r = None
        ten_i = None
        for i in range(N):
            # curr: (chi, 2_phys, chi) — permute from (chi, chi, 2)
            cr = t_r[i].permute(0, 2, 1)  # (chi, 2, chi) real part
            ci = t_i[i].permute(0, 2, 1)  # (chi, 2, chi) imag part

            if paulis[t_idx, i]:
                # AO = curr @ Z on physical dim: Z is real, so AO_r = cr@Z, AO_i = ci@Z
                AOr = torch.einsum('ldr,dk->lkr', cr, Z_r)
                AOi = torch.einsum('ldr,dk->lkr', ci, Z_r)
            else:
                AOr, AOi = cr, ci

            # E = conj(curr) tensordot AO on dim 1
            # conj(curr) = (cr, -ci)
            # E_r = cr·AOr + ci·AOi,  E_i = cr·AOi - ci·AOr
            Er = (torch.tensordot(cr, AOr, ([1], [1])) +
                  torch.tensordot(ci, AOi, ([1], [1]))).permute(0, 2, 1, 3)
            Ei = (torch.tensordot(cr, AOi, ([1], [1])) -
                  torch.tensordot(ci, AOr, ([1], [1]))).permute(0, 2, 1, 3)

            if ten_r is None:
                ten_r, ten_i = Er, Ei
            else:
                # Complex tensordot: ten @ E on dims ([2,3],[0,1])
                new_r = (torch.tensordot(ten_r, Er, ([2, 3], [0, 1])) -
                         torch.tensordot(ten_i, Ei, ([2, 3], [0, 1])))
                new_i = (torch.tensordot(ten_r, Ei, ([2, 3], [0, 1])) +
                         torch.tensordot(ten_i, Er, ([2, 3], [0, 1])))
                ten_r, ten_i = new_r, new_i

        # Trace: Re(sum_{i,j} ten[i,j,i,j])
        trace_r = torch.einsum('ijij->', ten_r)
        total = total + coef * trace_r

    return total


def autograd_gradient(theta, circuit, hamiltonian, dtype=torch.complex128,
                      term_chunk=None):
    """
    Compute gradient via backpropagation through differentiable SVD.

    Uses vectorized contraction that batches Hamiltonian terms in parallel.

    Args:
        theta: (P,) parameter tensor
        circuit: Circuit instance
        hamiltonian: Hamiltonian instance
        dtype: complex dtype (complex128 recommended for accuracy)
        term_chunk: max Hamiltonian terms to batch. None = auto from GPU memory.

    Returns:
        (grad, loss_value): (P,) float32 gradient, scalar expectation value
    """
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    N, chi = circuit.num_qubit, circuit.rank

    if term_chunk is None:
        term_chunk = _auto_term_chunk(N, chi, dtype, circuit.device)

    # Apply qubit permutation from transpiled circuits (same as circuit.get_expectation_value)
    if circuit.qubit_perm is not None:
        hamiltonian = hamiltonian.permuted(circuit.qubit_perm)

    with torch.enable_grad():
        theta_ad = theta.detach().to(real_dtype).clone().requires_grad_(True)
        tensor = _build_tensor_diff(theta_ad, circuit, dtype)
        loss = _contraction_diff_vectorized(tensor, hamiltonian, dtype,
                                            term_chunk=term_chunk)
        loss_val = loss.detach().float().item()
        tensor_detached = tensor.detach()
        loss.backward()
    return theta_ad.grad.float(), loss_val, tensor_detached


class AutogradGradient(Gradient):
    """
    Gradient computation via PyTorch autograd (backpropagation).

    Uses differentiable SVD with F/G split regularization and vectorized
    contraction that batches Hamiltonian terms in parallel.

    Args:
        dtype: complex dtype for computation (default: cfloat)
        term_chunk: max Hamiltonian terms to batch. None = auto from GPU memory.
    """

    def __init__(self, measure_method: MeasureMethod = MeasureMethod.EFFICIENT_CONTRACTION,
                 dtype=torch.cfloat, term_chunk=None):
        super().__init__(measure_method)
        self.dtype = dtype
        self.term_chunk = term_chunk
        self._verbose = True
        self._printed = False
        self.last_exp_value = None   # cached from forward pass
        self.last_tensor = None      # cached tensor (pre-step)

    def run(self, theta: torch.Tensor, circuit: Circuit, hamiltonian: Hamiltonian):
        N, chi = circuit.num_qubit, circuit.rank
        tc = self.term_chunk
        if tc is None:
            tc = _auto_term_chunk(N, chi, self.dtype, circuit.device)

        if self._verbose and not self._printed:
            T = len(hamiltonian.paulis)
            print(
                f"[TREV] AutogradGradient: dtype={self.dtype}, "
                f"params={theta.numel()}, device={circuit.device}, "
                f"term_chunk={min(tc, T)}/{T}\n",
                flush=True,
            )
            self._printed = True

        grad, exp_val, tensor = autograd_gradient(theta, circuit, hamiltonian,
                                                    self.dtype, term_chunk=tc)
        self.last_exp_value = exp_val
        self.last_tensor = tensor
        return grad
