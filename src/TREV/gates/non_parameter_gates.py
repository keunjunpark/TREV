from abc import ABC, abstractmethod
from typing import Callable

import torch
from .contraction import _apply_single_qubit_gate, _apply_single_qubit_gate_batch, _apply_double_qubit_gate, \
    _apply_double_qubit_gate_batch
from TREV.gates.gates import Gate

# 4x4 SWAP matrix for permuting qubit labels in 2-qubit gate matrices.
# Used to fix bond contraction order when q0 > q1 in a tensor ring.
_SWAP_4x4 = torch.tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])


def _swap_gate_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """Permute qubit labels in a 4x4 gate matrix: SWAP @ matrix @ SWAP."""
    swap = _SWAP_4x4.to(dtype=matrix.dtype, device=matrix.device)
    return swap @ matrix @ swap


class NonParameterGate(ABC):
    def __init__(self, matrix_fun: Callable, device:str):
        self.matrix_fun = matrix_fun
        self.device = device
    @abstractmethod
    def apply(self,tensor):
        pass
    @abstractmethod
    def apply_batch(self,batch_size, batch_tensor):
        pass
    def has_parameter(self):
        return False
class NonParameterOneQubitGate(NonParameterGate):
    def __init__(self, qubit:int, matrix_fun:Callable, device:str):
        super().__init__(matrix_fun, device)
        self.qubit = qubit

    def apply(self,tensor):
        matrix = self.matrix_fun(None, self.device)
        tensor[self.qubit]= _apply_single_qubit_gate(matrix, tensor[self.qubit])

    def apply_batch(self, batch_size, batch_tensor):
        matrix = self.matrix_fun(batch_size, self.device)
        batch_tensor[:, self.qubit] = _apply_single_qubit_gate_batch(matrix, batch_tensor[:, self.qubit])

class NonParameterTwoQubitsGate(NonParameterGate):
    def __init__(self, qubits:[int,int], matrix_fun:Callable, device:str):
        super().__init__(matrix_fun, device)
        self.qubits = qubits

    def apply(self,tensor):
        q0, q1 = self.qubits
        matrix = self.matrix_fun(None, self.device)
        if q0 < q1:
            tensor[q0], tensor[q1] = _apply_double_qubit_gate(matrix, (tensor[q0], tensor[q1]))
        else:
            # q0 > q1: shared bond is tensor[q0] axis 0 <-> tensor[q1] axis 1.
            # _apply_double_qubit_gate assumes axis 1 of first <-> axis 0 of second,
            # so swap tensor order and permute gate qubit labels.
            matrix_swapped = _swap_gate_matrix(matrix)
            tensor[q1], tensor[q0] = _apply_double_qubit_gate(matrix_swapped, (tensor[q1], tensor[q0]))

    def apply_batch(self, batch_size, batch_tensor):
        q0, q1 = self.qubits
        matrix = self.matrix_fun(batch_size, self.device)
        if q0 < q1:
            batch_tensor[:, q0], batch_tensor[:, q1] = _apply_double_qubit_gate_batch(matrix, (batch_tensor[:, q0], batch_tensor[:, q1]))
        else:
            # Same fix for batched path: swap tensor order + permute gate.
            if matrix.ndim == 2:
                matrix_swapped = _swap_gate_matrix(matrix)
            else:
                # Batched gate matrices: (B, 4, 4)
                swap = _SWAP_4x4.to(dtype=matrix.dtype, device=matrix.device)
                matrix_swapped = swap @ matrix @ swap
            batch_tensor[:, q1], batch_tensor[:, q0] = _apply_double_qubit_gate_batch(matrix_swapped, (batch_tensor[:, q1], batch_tensor[:, q0]))