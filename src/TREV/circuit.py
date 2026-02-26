"""
Core Class that can handle all.
Similar as much as possible from qiskit.
"""
import copy
from typing import List, Literal

import torch
from torch import Tensor
from .gates.non_parameter_gates import NonParameterOneQubitGate, NonParameterTwoQubitsGate, NonParameterGate
from .gates.parameter_gates import ParameterOneQubitGate, ParameterTwoQubitGate, ParameterGate
from .gates.info import I, H,X,Y,Z, RX, RY, RZ, ZZ, ZZ_SWAP, CNOT, SWAP
from .gates.contraction import _apply_single_qubit_gate, _apply_single_qubit_gate_batch
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

    def to_device(self, device: str) -> 'Circuit':
        """Create a lightweight clone targeting a different device.

        Shallow-copies the Circuit and rebuilds the gate list with updated
        device attributes.  No tensor copying — gate matrices are created
        on-the-fly in apply/apply_batch.
        """
        clone = copy.copy(self)
        clone.device = device
        clone.gates = []
        for gate in self.gates:
            g = copy.copy(gate)
            g.device = device
            clone.gates.append(g)
        return clone

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

    def zz(self, qubit0: int, qubit1: int):
        """ZZ(θ) gate: equivalent to CX-RZ-CX but uses a single SVD."""
        self.gates.append(ParameterTwoQubitGate([qubit0, qubit1], self.params_size, ZZ, self.device))
        self.params_size += 1

    def zz_swap(self, qubit0: int, qubit1: int):
        """Fused ZZ(θ)·SWAP gate: applies ZZ interaction and swaps in a single SVD."""
        self.gates.append(ParameterTwoQubitGate([qubit0, qubit1], self.params_size, ZZ_SWAP, self.device))
        self.params_size += 1

    def _compile_fused_ops(self):
        """Group consecutive single-qubit gates into fusible blocks, separated by 2-qubit gates.

        Returns a list of (op_type, payload):
          - ('block1q', {qubit: [gate, ...]})  -- fusible single-qubit block
          - ('2q', gate)                       -- non-parameter two-qubit gate
          - ('p2q', gate)                      -- parameter two-qubit gate (e.g. ZZ)
        """
        ops = []
        current_block = {}

        for gate in self.gates:
            if isinstance(gate, (ParameterOneQubitGate, NonParameterOneQubitGate)):
                q = gate.qubit
                if q not in current_block:
                    current_block[q] = []
                current_block[q].append(gate)
            elif isinstance(gate, ParameterTwoQubitGate):
                if current_block:
                    ops.append(('block1q', current_block))
                    current_block = {}
                ops.append(('p2q', gate))
            else:
                if current_block:
                    ops.append(('block1q', current_block))
                    current_block = {}
                ops.append(('2q', gate))

        if current_block:
            ops.append(('block1q', current_block))

        return ops

    def build_tensor(self, theta: Tensor):
        tensor:Tensor = torch.zeros((self.num_qubit, self.rank, self.rank , 2), dtype=torch.cfloat, device=self.device)
        tensor[:, 0, 0, 0] = 1.0

        ops = self._compile_fused_ops()
        for op_type, payload in ops:
            if op_type == 'block1q':
                for qubit, gates in payload.items():
                    fused = None
                    for gate in gates:
                        if gate.has_parameter():
                            mat = gate.matrix_fun(theta[gate.theta_index], self.device)
                        else:
                            mat = gate.matrix_fun(None, self.device)
                        fused = mat if fused is None else torch.mm(mat, fused)
                    tensor[qubit] = _apply_single_qubit_gate(fused, tensor[qubit])
            elif op_type == 'p2q':
                payload.apply(theta, tensor)
            else:  # '2q'
                payload.apply(tensor)
        return tensor

    def build_tensor_batch(self, theta: Tensor, batch_size:int):
        tensor: Tensor = torch.zeros((self.num_qubit, self.rank, self.rank, 2), dtype=torch.cfloat, device=self.device)
        tensor[:, 0, 0, 0] = 1.0
        tensor = tensor.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).clone()

        ops = self._compile_fused_ops()
        for op_type, payload in ops:
            if op_type == 'block1q':
                for qubit, gates in payload.items():
                    fused = None
                    for gate in gates:
                        if gate.has_parameter():
                            mat = gate.matrix_fun(theta[:, gate.theta_index], self.device)
                        else:
                            mat = gate.matrix_fun(batch_size, self.device)
                        fused = mat if fused is None else torch.bmm(mat, fused)
                    tensor[:, qubit] = _apply_single_qubit_gate_batch(fused, tensor[:, qubit])
            elif op_type == 'p2q':
                payload.apply_batch(theta, batch_size, tensor)
            else:  # '2q'
                payload.apply_batch(batch_size, tensor)
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
        from .optimization.gradients.batch_parameter_shift import (
            expectation_value_batch as _perfect_sampling_batch,
            expectation_value_batch_efficient_contraction,
            expectation_value_batch_right_suffix,
        )

        single = theta.dim() == 1
        theta_batch = theta.unsqueeze(0) if single else theta
        if method == MeasureMethod.PERFECT_SAMPLING:
            result = _perfect_sampling_batch(theta_batch, self, hamiltonian, shots)
        elif method == MeasureMethod.EFFICIENT_CONTRACTION:
            result = expectation_value_batch_efficient_contraction(theta_batch, self, hamiltonian, shots)
        elif method == MeasureMethod.RIGHT_SUFFIX_SAMPLING:
            result = expectation_value_batch_right_suffix(theta_batch, self, hamiltonian, shots)
        else:
            raise NotImplementedError()
        return result.squeeze(0) if single else result
