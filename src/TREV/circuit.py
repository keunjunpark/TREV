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

    # ── Prefix caching for parameter-shift gradient ──

    def get_prefix_checkpoints(self):
        """Identify gate indices where we can cache tensor state.

        Returns list of (gate_index, param_range) where:
          - gate_index: first gate index in this segment (start replaying from here)
          - param_range: (first_theta_idx, last_theta_idx+1) of params in this segment

        The prefix up to gate_index is parameter-free or uses earlier params,
        so it's identical across shifts of params in param_range.
        """
        checkpoints = []
        seg_start = 0
        seg_param_lo = None
        seg_param_hi = None

        for i, gate in enumerate(self.gates):
            if isinstance(gate, NonParameterTwoQubitsGate):
                # 2-qubit gate: flush current segment if it has params
                if seg_param_lo is not None:
                    checkpoints.append((seg_start, (seg_param_lo, seg_param_hi + 1)))
                # Next segment starts after this 2-qubit gate
                seg_start = i  # include the 2q gate in replay
                seg_param_lo = None
                seg_param_hi = None
            elif gate.has_parameter():
                tidx = gate.theta_index
                if seg_param_lo is None:
                    seg_param_lo = tidx
                    seg_param_hi = tidx
                    seg_start = i  # segment starts at first param gate
                else:
                    seg_param_hi = max(seg_param_hi, tidx)

        # Flush last segment
        if seg_param_lo is not None:
            checkpoints.append((seg_start, (seg_param_lo, seg_param_hi + 1)))

        return checkpoints

    def build_prefix_batch(self, theta: Tensor, batch_size: int, up_to_gate: int):
        """Build tensor applying only gates [0, up_to_gate).

        theta can be a single (P,) tensor since the prefix is param-independent
        or shares params with the base circuit.
        """
        tensor = torch.zeros((self.num_qubit, self.rank, self.rank, 2),
                             dtype=torch.cfloat, device=self.device)
        tensor[:, 0, 0, 0] = 1.0
        # For prefix, use batch_size=1 since all are identical
        for gate in self.gates[:up_to_gate]:
            if gate.has_parameter():
                gate.apply(theta, tensor)
            else:
                gate.apply(tensor)
        # Expand to batch
        return tensor.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).clone()

    def build_from_prefix_batch(self, prefix: Tensor, theta: Tensor,
                                batch_size: int, from_gate: int):
        """Continue building from a cached prefix tensor, applying gates [from_gate, end).

        prefix: (B, N, chi, chi, 2) — cached state
        theta: (B, P) — batched parameters
        """
        tensor = prefix.clone()
        for gate in self.gates[from_gate:]:
            if gate.has_parameter():
                gate.apply_batch(theta, batch_size, tensor)
            else:
                gate.apply_batch(batch_size, tensor)
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
