from abc import ABC, abstractmethod

import torch

from ...circuit import Circuit
from ...hamiltonian.hamiltonian import Hamiltonian
from ...measure.enums import MeasureMethod


class Gradient(ABC):
    def __init__(self, measure_method: MeasureMethod):
        self.measure_method = measure_method
    @abstractmethod
    def run(self, theta:torch.Tensor, circuit:Circuit, hamiltonian:Hamiltonian):
        pass



class VanillaParameterShift(Gradient):
    def __init__(self, shift, shots, measure_method: MeasureMethod):
        super().__init__(measure_method)
        self.shift = shift
        self.shots = shots
    def run(self, theta:torch.Tensor, circuit:Circuit, hamiltonian:Hamiltonian):
        gradients = torch.zeros_like(theta).to(circuit.device)
        shift = torch.tensor(torch.pi / 2).to(circuit.device)
        for i in range(len(theta)):
            params_forward = theta.clone().to(circuit.device)
            params_backward = theta.clone().to(circuit.device)
            params_forward[i] += shift
            params_backward[i] -= shift
            gradients[i] = (circuit.get_expectation_value(params_forward, hamiltonian, self.measure_method,
                                                          int(self.shots)) -
                            circuit.get_expectation_value(params_backward, hamiltonian, self.measure_method,
                                                          int(self.shots))) / 2
        return gradients