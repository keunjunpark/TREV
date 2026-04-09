from abc import ABC, abstractmethod
from typing import Callable

from .contraction import _apply_single_qubit_gate, _apply_single_qubit_gate_batch, _apply_double_qubit_gate, \
    _apply_double_qubit_gate_batch
from TREV.gates.gates import Gate


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
        matrix = self.matrix_fun(None,self.device)
        tensor[self.qubits[0]] ,tensor[self.qubits[1]]= _apply_double_qubit_gate(matrix, (tensor[self.qubits[0]], tensor[self.qubits[1]]))

    def apply_batch(self, batch_size, batch_tensor):
        matrix = self.matrix_fun(batch_size, self.device)
        batch_tensor[:, self.qubits[0]], batch_tensor[:, self.qubits[1]] = _apply_double_qubit_gate_batch(matrix, (batch_tensor[:, self.qubits[0]], batch_tensor[:, self.qubits[1]]))