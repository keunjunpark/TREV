from abc import ABC, abstractmethod
from typing import Callable

from .contraction import _apply_single_qubit_gate, _apply_single_qubit_gate_batch, _apply_double_qubit_gate, _apply_double_qubit_gate_batch
from .non_parameter_gates import _swap_gate_matrix, _SWAP_4x4
from TREV.gates.gates import Gate
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

class ParameterMultiOneQubitGate(ParameterGate):
    """Single-qubit gate with multiple parameters (e.g. U3 with theta, phi, lambda)."""
    def __init__(self, qubit:int, theta_indices:list, matrix_fun:Callable, device:str):
        super().__init__(theta_indices[0], device)
        self.qubit = qubit
        self.theta_indices = theta_indices
        self.matrix_fun = matrix_fun
        self.num_params = len(theta_indices)

    def apply(self, theta, tensor):
        params = torch.stack([theta[i] for i in self.theta_indices])  # (num_params,)
        matrix = self.matrix_fun(params, self.device)
        tensor[self.qubit] = _apply_single_qubit_gate(matrix, tensor[self.qubit])

    def apply_batch(self, batch_theta, batch_size, batch_tensor):
        params = torch.stack([batch_theta[:, i] for i in self.theta_indices], dim=-1)  # (batch, num_params)
        matrix = self.matrix_fun(params, self.device)
        batch_tensor[:, self.qubit] = _apply_single_qubit_gate_batch(matrix, batch_tensor[:, self.qubit])

class ParameterTwoQubitGate(ParameterGate):
    def __init__(self, qubits, theta_index: int, matrix_fun, device: str):
        super().__init__(theta_index, device)
        self.qubits = qubits
        self.matrix_fun = matrix_fun

    def apply(self, theta, tensor):
        q0, q1 = self.qubits
        N = tensor.shape[0]
        matrix = self.matrix_fun(theta[self.theta_index], self.device)

        is_wrap_fwd = (q0 == N - 1 and q1 == 0)
        is_wrap_bwd = (q0 == 0 and q1 == N - 1)

        if is_wrap_fwd:
            tensor[N-1], tensor[0] = _apply_double_qubit_gate(matrix, (tensor[N-1], tensor[0]))
        elif is_wrap_bwd:
            matrix_swapped = _swap_gate_matrix(matrix)
            tensor[N-1], tensor[0] = _apply_double_qubit_gate(matrix_swapped, (tensor[N-1], tensor[0]))
        elif q0 < q1:
            tensor[q0], tensor[q1] = _apply_double_qubit_gate(matrix, (tensor[q0], tensor[q1]))
        else:
            matrix_swapped = _swap_gate_matrix(matrix)
            tensor[q1], tensor[q0] = _apply_double_qubit_gate(matrix_swapped, (tensor[q1], tensor[q0]))

    def apply_batch(self, batch_theta, batch_size, batch_tensor):
        q0, q1 = self.qubits
        N = batch_tensor.shape[1]
        matrix = self.matrix_fun(batch_theta[:, self.theta_index], self.device)

        def _swap_matrix(m):
            if m.ndim == 2:
                return _swap_gate_matrix(m)
            swap = _SWAP_4x4.to(dtype=m.dtype, device=m.device)
            return swap @ m @ swap

        is_wrap_fwd = (q0 == N - 1 and q1 == 0)
        is_wrap_bwd = (q0 == 0 and q1 == N - 1)

        if is_wrap_fwd:
            batch_tensor[:, N-1], batch_tensor[:, 0] = _apply_double_qubit_gate_batch(matrix, (batch_tensor[:, N-1], batch_tensor[:, 0]))
        elif is_wrap_bwd:
            matrix_swapped = _swap_matrix(matrix)
            batch_tensor[:, N-1], batch_tensor[:, 0] = _apply_double_qubit_gate_batch(matrix_swapped, (batch_tensor[:, N-1], batch_tensor[:, 0]))
        elif q0 < q1:
            batch_tensor[:, q0], batch_tensor[:, q1] = _apply_double_qubit_gate_batch(matrix, (batch_tensor[:, q0], batch_tensor[:, q1]))
        else:
            matrix_swapped = _swap_matrix(matrix)
            batch_tensor[:, q1], batch_tensor[:, q0] = _apply_double_qubit_gate_batch(matrix_swapped, (batch_tensor[:, q1], batch_tensor[:, q0]))