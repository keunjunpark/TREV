from typing import Tuple
import torch
from torch import Tensor

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
    # qu0: χ1 × χ2 × 2
    # qu1: χ2 × χ3 × 2

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

    u, s, v = torch.linalg.svd(mps)
    # u: (2 * χ1) × (2 * χ1)
    # s: 2 * min(χ1,χ3)
    # y: (2 * χ3) × (2 * χ3)

    # TODO apply rescaling to sx (below) b/c dim = min(χ1,χ3) ?= χ1
    #  -- not technically necessary, unless chi values start off different
    #  -- not necessary right now, but a future-proofing good-to-have

    # noinspection PyTypeChecker
    x, sx, y = u[:, :chi_1], torch.diag(s[:chi_1]).type(torch.cfloat), v[:chi_3, :]
    # x: (2 * χ1) × χ1
    # sx: χ1 × χ1
    # y: χ3 × (2 * χ3)

    qu0 = torch.mm(x, sx).reshape((2, chi_1, chi_1))
    # qu0: ((2 * χ1) × [χ1]) . ([χ1] × χ1) = (2 * χ1) × χ1 --> 2 × χ1 × χ1
    qu1 = y.reshape((chi_3, 2, chi_3))
    # qu1: χ3 × 2 × χ3

    qu0 = torch.moveaxis(qu0, 0, 2)
    # qu0: χ1 × χ1 × 2
    qu1 = torch.moveaxis(qu1, 1, 2)
    # qu1: χ3 × χ3 × 2

    return qu0, qu1


def  _apply_double_qubit_gate_batch(
    gate_matrix: torch.Tensor,                              # (4, 4)   or (B, 4, 4)
    qu_state_tensors: Tuple[torch.Tensor, torch.Tensor]     # qu0: (B, χ1, χ2, 2), qu1: (B, χ2, χ3, 2)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched version of _apply_double_qubit_gate.

    Returns
    -------
    qu0' : (B, χ1, χ1, 2)
    qu1' : (B, χ3, χ3, 2)
    """
    qu0, qu1 = qu_state_tensors
    B, chi1, chi2, _ = qu0.shape
    _, chi2_, chi3, _ = qu1.shape
    assert chi2 == chi2_, "Bond mismatch between the two site tensors"

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

    u, s, vh = torch.linalg.svd(mps, full_matrices=False)   # u: (B, 2χ1, r),  vh: (B, r, 2χ3)

    # Keep the same number of singular vectors you did in the scalar path
    x  = u[:, :, :chi1]                                     # (B, 2χ1, χ1)
    sx = torch.diag_embed(s[:, :chi1]).to(torch.cfloat)      # (B, χ1,  χ1)
    y  = vh[:, :chi3, :]                                     # (B, χ3, 2χ3)

    qu0_new = torch.bmm(x, sx).reshape(B, 2, chi1, chi1).permute(0, 2, 3, 1)  # (B, χ1, χ1, 2)
    qu1_new = y.reshape(B, chi3, 2, chi3).permute(0, 1, 3, 2)  # (B, χ3, χ3, 2)

    return qu0_new, qu1_new
