from abc import ABC, abstractmethod
from typing import Callable

from .contraction import _apply_single_qubit_gate, _apply_single_qubit_gate_batch,_apply_double_qubit_gate
from TRVQA.gates.gates import Gate
class ParameterGate(ABC):
    def __init__(self, theta_index:int, device:str):
        self.theta_index = theta_index
        self.device = device
    @abstractmethod
    def apply(self, theta, tensor):
        pass
    @abstractmethod
    def apply_batch(self, batch_theta, batch_size, batch_tensor):
        pass
    def has_parameter(self):
        return True
class ParameterOneQubitGate(ParameterGate):
    def __init__(self, qubit:int, theta_index:int, matrix_fun:Callable, device:str):
        super().__init__(theta_index, device)
        self.qubit = qubit
        self.matrix_fun = matrix_fun
    def apply(self, theta, tensor):
        matrix = self.matrix_fun(theta[self.theta_index], self.device)
        tensor[self.qubit]= _apply_single_qubit_gate(matrix, tensor[self.qubit])

    def apply_batch(self, batch_theta, batch_size, batch_tensor):
        matrix = self.matrix_fun(batch_theta[:, self.theta_index],self.device)
        batch_tensor[:, self.qubit] = _apply_single_qubit_gate_batch(matrix, batch_tensor[:, self.qubit] )

class ParameterTwoQubitGate(ParameterGate):
    def apply(self, theta, tensor):
        raise NotImplementedError()
    def apply_batch(self, batch_theta, batch_size, batch_tensor):
        raise NotImplementedError()