from typing import Tuple
import warnings
import torch
from torch import Tensor

warnings.filterwarnings("ignore", message="torch.linalg.svd")

# Relative SVD threshold: singular values below s_max * SVD_THRESHOLD
# are zeroed out to prevent noise accumulation across many gates.
# Set to 0.0 to disable thresholding entirely.
SVD_THRESHOLD: float = 0.0

# When True, upcast tensors to cdouble before SVD and cast back after.
# Reduces numerical noise accumulation across many 2-qubit gates.
UPCAST_SVD: bool = False

# When True, use eigendecomposition of M^H M for truncated SVD.
# ~1.6x faster for batched (B>=24) operations on GPU.
# Slower for small batches (B<10) due to matmul overhead.
# Test on your hardware before enabling.
USE_EIGH_SVD: bool = False

def _truncated_svd(matrix: Tensor, rank: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute truncated SVD, returning (U_k, S_k, Vh_k).

    For batched inputs with USE_EIGH_SVD=True, uses eigendecomposition
    of M^H M which is ~1.6x faster for the gradient computation path.
    Single (non-batched) inputs always use torch.linalg.svd.
    """
    if matrix.ndim == 2 or not USE_EIGH_SVD:
        return torch.linalg.svd(matrix, full_matrices=False)

    # Batched eigh path: M^H M is Hermitian positive semi-definite
    MhM = torch.bmm(matrix.conj().transpose(-2, -1), matrix)
    eigenvalues, V = torch.linalg.eigh(MhM)

    # eigh returns ascending order; take top-k (last k columns)
    s_sq = eigenvalues[:, -rank:].flip(-1).clamp(min=0)
    s = s_sq.sqrt()
    Vk = V[:, :, -rank:].flip(-1)

    # U_k = M V_k S_k^{-1}
    s_safe = s.clamp(min=1e-30)
    Uk = torch.bmm(matrix, Vk) / s_safe.unsqueeze(-2)

    Vhk = Vk.conj().transpose(-2, -1)

    return Uk, s, Vhk

def _apply_single_qubit_gate_batch(gate_matrix_batch: Tensor, qu_state_tensor_batch:Tensor):
    qu_state_tensor_batch = torch.einsum('bij,bklj->bikl', gate_matrix_batch, qu_state_tensor_batch)  # (B, 2, χ1, χ2)
    qu_state_tensor_batch = qu_state_tensor_batch.permute(0, 2, 3, 1)  # (B, χ1, χ2, 2)
    return qu_state_tensor_batch

def _apply_single_qubit_gate(gate_matrix: Tensor, qu_state_tensor: Tensor) -> Tensor:
    """ Apply the specified 1-qubit gate matrix on the specified ring-tensor """
    # gate_matrix: 2 × 2
    # qu_state_tensor: χ1 × χ2 × 2
    qu_state_tensor = torch.tensordot(gate_matrix, qu_state_tensor, ([1], [2]))
    # qu_state_tensor: (2 × [2]) . (χ1 × χ2 × [2]) = 2 × χ1 × χ2
    qu_state_tensor = torch.moveaxis(qu_state_tensor, 0, 2)

    # qu_state_tensor: χ1 × χ2 × 2
    return qu_state_tensor


def _apply_double_qubit_gate(gate_matrix: Tensor, qu_state_tensors: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    """ Apply the specified 2-qubit gate matrix on the specified ring-tensors """
    # gate_matrix: 4 × 4
    qu0, qu1 = qu_state_tensors
    orig_dtype = qu0.dtype
    # qu0: χ1 × χ2 × 2
    # qu1: χ2 × χ3 × 2

    if UPCAST_SVD and orig_dtype != torch.cdouble:
        qu0 = qu0.to(torch.cdouble)
        qu1 = qu1.to(torch.cdouble)
        gate_matrix = gate_matrix.to(torch.cdouble)

    chi_1 = qu0.shape[0]
    chi_3 = qu1.shape[1]
    # chi_1 = χ1
    # chi_3 = χ3

    mps = torch.tensordot(qu0, qu1, ([1], [0]))
    # mps: (χ1 × [χ2] × 2) . ([χ2] × χ3 × 2) = χ1 × 2 × χ3 × 2
    mps = torch.moveaxis(mps, 2, 1)
    # mps: χ1 × χ3 × 2 × 2

    gate_tensor = torch.reshape(gate_matrix, (2, 2, 2, 2))
    # gate_tensor: 2 × 2 × 2 × 2

    mps = torch.tensordot(gate_tensor, mps, ([2, 3], [2, 3]))
    # mps: (2 × 2 × [2] × [2]) . (χ1 × χ3 × [2] × [2]) = 2 × 2 × χ1 × χ3
    mps = torch.moveaxis(mps, 1, 2).reshape((chi_1 * 2, chi_3 * 2))
    # mps: 2 × χ1 × 2 × χ3 --> (2 * χ1) × (2 * χ3)

    trunc_rank = max(chi_1, chi_3)
    u, s, vh = _truncated_svd(mps, trunc_rank)
    # Zero out noise singular values to prevent accumulation at high rank
    if SVD_THRESHOLD > 0:
        s = torch.where(s > s[0] * SVD_THRESHOLD, s, torch.zeros_like(s))
    x, sx, y = u[:, :chi_1], torch.diag(s[:chi_1]).to(dtype=mps.dtype), vh[:chi_3, :]
    qu0 = torch.mm(x, sx).reshape((2, chi_1, chi_1))
    qu1 = y.reshape((chi_3, 2, chi_3))

    qu0 = torch.moveaxis(qu0, 0, 2)
    # qu0: χ1 × χ1 × 2
    qu1 = torch.moveaxis(qu1, 1, 2)
    # qu1: χ3 × χ3 × 2

    if UPCAST_SVD and orig_dtype != torch.cdouble:
        qu0 = qu0.to(orig_dtype)
        qu1 = qu1.to(orig_dtype)

    return qu0, qu1


def  _apply_double_qubit_gate_batch(
    gate_matrix: torch.Tensor,                              # (4, 4)   or (B, 4, 4)
    qu_state_tensors: Tuple[torch.Tensor, torch.Tensor],    # qu0: (B, χ1, χ2, 2), qu1: (B, χ2, χ3, 2)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched version of _apply_double_qubit_gate.

    Returns
    -------
    qu0' : (B, χ1, χ1, 2)
    qu1' : (B, χ3, χ3, 2)
    """
    qu0, qu1 = qu_state_tensors
    orig_dtype = qu0.dtype
    B, chi1, chi2, _ = qu0.shape
    _, chi2_, chi3, _ = qu1.shape
    assert chi2 == chi2_, "Bond mismatch between the two site tensors"

    if UPCAST_SVD and orig_dtype != torch.cdouble:
        qu0 = qu0.to(torch.cdouble)
        qu1 = qu1.to(torch.cdouble)
        gate_matrix = gate_matrix.to(torch.cdouble)

    mps = torch.einsum('bikp,bkjq->bijpq', qu0, qu1)

    g = gate_matrix
    if g.ndim == 2:                      # same gate for every batch element
        g = g.expand(B, -1, -1)          # (B, 4, 4)
    else:
        assert g.shape[0] == B, "gate_matrix batch dimension must match qu_state_tensors"

    mps = mps.reshape(B, chi1 * chi3, 4)               # flatten physical legs
    mps = torch.bmm(mps, g.transpose(1, 2))            # apply gate
    mps = mps.view(B, chi1, chi3, 2, 2)

    mps = mps.permute(0, 3, 1, 4, 2).reshape(B, 2 * chi1, 2 * chi3)

    trunc_rank = max(chi1, chi3)
    u, s, vh = _truncated_svd(mps, trunc_rank)
    # Zero out noise singular values to prevent accumulation at high rank
    if SVD_THRESHOLD > 0:
        threshold = s[:, 0:1] * SVD_THRESHOLD
        s = torch.where(s > threshold, s, torch.zeros_like(s))
    x  = u[:, :, :chi1]
    sx = torch.diag_embed(s[:, :chi1]).to(dtype=mps.dtype)
    y  = vh[:, :chi3, :]
    qu0_new = torch.bmm(x, sx).reshape(B, 2, chi1, chi1).permute(0, 2, 3, 1)
    qu1_new = y.reshape(B, chi3, 2, chi3).permute(0, 1, 3, 2)

    if UPCAST_SVD and orig_dtype != torch.cdouble:
        qu0_new = qu0_new.to(orig_dtype)
        qu1_new = qu1_new.to(orig_dtype)

    return qu0_new, qu1_new
