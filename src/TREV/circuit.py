"""
Core Class that can handle all.
Similar as much as possible from qiskit.
"""
from typing import List, Literal

import torch
from torch import Tensor
from .gates.non_parameter_gates import NonParameterOneQubitGate, NonParameterTwoQubitsGate, NonParameterGate
from .gates.parameter_gates import ParameterOneQubitGate, ParameterGate
from .gates.info import I, H,X,Y,Z, RX, RY, RZ, CNOT, SWAP
from .hamiltonian.hamiltonian import Hamiltonian
from .measure.enums import MeasureMethod
from .measure import contraction, perfect_sampling, efficient_contraction, right_suffix_sampling
class Circuit(torch.nn.Module):
    def __init__(self, num_qubit:int, rank:int=10, device:str='cpu'):
        super().__init__()
        self.rank:int = rank
        self.gates:List[ParameterGate|NonParameterGate] = []
        self.params_size:int = 0
        self.num_qubit = num_qubit
        self.device = device
    def id(self, qubit: int):
        self.gates.append(NonParameterOneQubitGate(qubit, I,self.device))

    def h(self, qubit: int):
        self.gates.append(NonParameterOneQubitGate(qubit,H,self.device))

    def x(self, qubit: int):
        self.gates.append(NonParameterOneQubitGate(qubit, X,self.device))

    def y(self, qubit: int):
        self.gates.append(NonParameterOneQubitGate(qubit, Y,self.device))

    def z(self, qubit: int):
        self.gates.append(NonParameterOneQubitGate(qubit, Z,self.device))

    def rx(self, qubit: int):
        self.gates.append(ParameterOneQubitGate(qubit, self.params_size, RX,self.device))
        self.params_size += 1

    def ry(self, qubit: int):
        self.gates.append(ParameterOneQubitGate(qubit, self.params_size, RY,self.device))
        self.params_size += 1

    def rz(self, qubit: int):
        self.gates.append(ParameterOneQubitGate(qubit, self.params_size, RZ,self.device))
        self.params_size += 1

    def cx(self, control:int, target:int):
        self.gates.append(NonParameterTwoQubitsGate([control,target], CNOT,self.device))

    def swap(self, control:int, target:int):
        self.gates.append(NonParameterTwoQubitsGate([control,target], SWAP,self.device))

    def build_tensor(self, theta: Tensor):
        tensor:Tensor = torch.zeros((self.num_qubit, self.rank, self.rank , 2), dtype=torch.cfloat, device=self.device)
        tensor[:, 0, 0, 0] = 1.0
        for gate in self.gates:
            if gate.has_parameter():
                p_gate:ParameterGate = gate
                p_gate.apply(theta, tensor)
            else:
                np_gate: NonParameterGate = gate
                np_gate.apply(tensor)
        return tensor
    
    def build_tensor_batch(self, theta: Tensor, batch_size:int):
        tensor: Tensor = torch.zeros((self.num_qubit, self.rank, self.rank, 2), dtype=torch.cfloat, device=self.device)
        tensor[:, 0, 0, 0] = 1.0
        tensor = tensor.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).clone()
        for gate in self.gates:
            if gate.has_parameter():
                p_gate:ParameterGate = gate
                p_gate.apply_batch(theta, batch_size, tensor)
            else:
                np_gate: NonParameterGate = gate
                np_gate.apply_batch(batch_size, tensor)
        return tensor

    def measure(self, theta: Tensor, method:MeasureMethod=MeasureMethod.PERFECT_SAMPLING, shots:int= int(1e4)):
        tensor = self.build_tensor(theta)
        if method == MeasureMethod.FULL_CONTRACTION:
            return contraction.measure(tensor)
        elif method == MeasureMethod.PERFECT_SAMPLING:
            return perfect_sampling.measure(tensor,shots,device=self.device)
        else:
            raise NotImplementedError()

    def get_expectation_value(self, theta: Tensor, hamiltonian:Hamiltonian, method: MeasureMethod, shots:int= int(1e4)):
        tensor = self.build_tensor(theta)
        if method == MeasureMethod.FULL_CONTRACTION:
            return contraction.expectation_value(tensor,hamiltonian, device=self.device).real
        elif method == MeasureMethod.PERFECT_SAMPLING:
            return perfect_sampling.expectation_value(tensor,hamiltonian,device=self.device, shot=shots)
        elif method == MeasureMethod.EFFICIENT_CONTRACTION:
            return efficient_contraction.expectation_value_batch(tensor,hamiltonian,device=self.device, chunk_size=shots)
        elif method == MeasureMethod.RIGHT_SUFFIX_SAMPLING:
            return right_suffix_sampling.expectation_value(tensor,hamiltonian,shots=shots, chunk_size=shots)
        else:
            raise NotImplementedError()
