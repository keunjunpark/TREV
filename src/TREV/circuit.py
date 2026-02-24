"""
Core Class that can handle all.
Similar as much as possible from qiskit.
"""
import copy
from typing import List, Literal

import torch
from torch import Tensor
from .gates.non_parameter_gates import NonParameterOneQubitGate, NonParameterTwoQubitsGate, NonParameterGate
from .gates.parameter_gates import ParameterOneQubitGate, ParameterGate
from .gates.contraction import _apply_single_qubit_gate, _apply_single_qubit_gate_batch
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

    def _compile_fused_ops(self):
        """Group consecutive single-qubit gates into blocks, fuse per-qubit.

        Returns a list of operations:
          ('block1q', {qubit: [gate, ...]})  — fused single-qubit block
          ('2q', gate)                       — two-qubit gate (unchanged)
        """
        ops = []
        i = 0
        n = len(self.gates)
        while i < n:
            gate = self.gates[i]
            if isinstance(gate, (ParameterOneQubitGate, NonParameterOneQubitGate)):
                # Collect all consecutive single-qubit gates into one block
                block = {}  # qubit -> [gates in application order]
                j = i
                while j < n:
                    g = self.gates[j]
                    if isinstance(g, (ParameterOneQubitGate, NonParameterOneQubitGate)):
                        q = g.qubit
                        if q not in block:
                            block[q] = []
                        block[q].append(g)
                        j += 1
                    else:
                        break
                ops.append(('block1q', block))
                i = j
            else:
                ops.append(('2q', gate))
                i += 1
        return ops

    def build_tensor(self, theta: Tensor):
        tensor:Tensor = torch.zeros((self.num_qubit, self.rank, self.rank , 2), dtype=torch.cfloat, device=self.device)
        tensor[:, 0, 0, 0] = 1.0
        for op_type, payload in self._compile_fused_ops():
            if op_type == 'block1q':
                for qubit in sorted(payload.keys()):
                    fused = None
                    for g in payload[qubit]:
                        if g.has_parameter():
                            mat = g.matrix_fun(theta[g.theta_index], self.device)
                        else:
                            mat = g.matrix_fun(None, self.device)
                        fused = mat if fused is None else torch.mm(mat, fused)
                    tensor[qubit] = _apply_single_qubit_gate(fused, tensor[qubit])
            else:  # '2q'
                gate = payload
                if gate.has_parameter():
                    gate.apply(theta, tensor)
                else:
                    gate.apply(tensor)
        return tensor

    def build_tensor_batch(self, theta: Tensor, batch_size:int):
        tensor: Tensor = torch.zeros((self.num_qubit, self.rank, self.rank, 2), dtype=torch.cfloat, device=self.device)
        tensor[:, 0, 0, 0] = 1.0
        tensor = tensor.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).clone()
        for op_type, payload in self._compile_fused_ops():
            if op_type == 'block1q':
                for qubit in sorted(payload.keys()):
                    fused = None
                    for g in payload[qubit]:
                        if g.has_parameter():
                            mat = g.matrix_fun(theta[:, g.theta_index], self.device)
                        else:
                            mat = g.matrix_fun(batch_size, self.device)
                        if mat.dim() == 2:
                            mat = mat.unsqueeze(0).expand(batch_size, -1, -1)
                        fused = mat if fused is None else torch.bmm(mat, fused)
                    tensor[:, qubit] = _apply_single_qubit_gate_batch(
                        fused, tensor[:, qubit])
            else:  # '2q'
                gate = payload
                if gate.has_parameter():
                    gate.apply_batch(theta, batch_size, tensor)
                else:
                    gate.apply_batch(batch_size, tensor)
        return tensor

    def _build_prefix_checkpoints(self, theta_base: Tensor):
        """Build base tensor (B=1) saving checkpoints before each compiled op.

        Args:
            theta_base: (P,) base parameters

        Returns:
            checkpoints: list of tensors, checkpoints[i] = state before op i,
                         checkpoints[len(ops)] = final state. Shape: (N, chi, chi, 2)
            param_to_op: dict mapping theta_index to the op index where it first appears
        """
        ops = self._compile_fused_ops()
        tensor = torch.zeros((self.num_qubit, self.rank, self.rank, 2),
                             dtype=torch.cfloat, device=self.device)
        tensor[:, 0, 0, 0] = 1.0

        checkpoints = []
        param_to_op = {}

        for op_idx, (op_type, payload) in enumerate(ops):
            checkpoints.append(tensor.clone())

            if op_type == 'block1q':
                # Record param_to_op mapping
                for qubit in sorted(payload.keys()):
                    for g in payload[qubit]:
                        if g.has_parameter() and g.theta_index not in param_to_op:
                            param_to_op[g.theta_index] = op_idx

                # Apply the block (same as build_tensor)
                for qubit in sorted(payload.keys()):
                    fused = None
                    for g in payload[qubit]:
                        if g.has_parameter():
                            mat = g.matrix_fun(theta_base[g.theta_index], self.device)
                        else:
                            mat = g.matrix_fun(None, self.device)
                        fused = mat if fused is None else torch.mm(mat, fused)
                    tensor[qubit] = _apply_single_qubit_gate(fused, tensor[qubit])
            else:  # '2q'
                gate = payload
                if gate.has_parameter():
                    gate.apply(theta_base, tensor)
                else:
                    gate.apply(tensor)

        checkpoints.append(tensor.clone())  # Final state after all ops
        return checkpoints, param_to_op

    def build_tensor_paramshift(self, theta_base: Tensor, shift_indices: Tensor,
                                shift_val: float, checkpoints=None, param_to_op=None):
        """Build shifted parameter tensors using prefix caching.

        For parameter shift gradient, each shifted parameter only differs from
        the base at one gate. This method reuses precomputed prefix checkpoints
        to avoid redundant gate applications.

        Args:
            theta_base: (P,) base parameters
            shift_indices: (C,) parameter indices to shift
            shift_val: shift amount (e.g. pi/2)
            checkpoints: from _build_prefix_checkpoints (computed if None)
            param_to_op: from _build_prefix_checkpoints (computed if None)

        Returns:
            (2*C, N, chi, chi, 2) tensor — first C are +shift, last C are -shift
        """
        if checkpoints is None or param_to_op is None:
            checkpoints, param_to_op = self._build_prefix_checkpoints(theta_base)

        ops = self._compile_fused_ops()
        C = shift_indices.numel()

        if C == 0:
            return torch.empty((0, self.num_qubit, self.rank, self.rank, 2),
                               dtype=torch.cfloat, device=self.device)

        out = torch.empty((2 * C, self.num_qubit, self.rank, self.rank, 2),
                          dtype=torch.cfloat, device=self.device)

        # Group shift_indices by divergence op
        groups = {}  # op_idx -> [(position_in_C, param_idx)]
        unmatched = []  # positions where param isn't in any gate

        for pos in range(C):
            pidx = shift_indices[pos].item()
            if pidx in param_to_op:
                op_idx = param_to_op[pidx]
                if op_idx not in groups:
                    groups[op_idx] = []
                groups[op_idx].append((pos, pidx))
            else:
                unmatched.append(pos)

        # Handle unmatched params (not in any gate -> shifting has no effect)
        if unmatched:
            final = checkpoints[-1]
            for pos in unmatched:
                out[pos] = final
                out[C + pos] = final

        # Process each group
        for op_k, members in groups.items():
            G = len(members)
            B_group = 2 * G

            # Build theta_batch for this group
            theta_batch = theta_base.unsqueeze(0).expand(B_group, -1).clone()
            for local_idx, (pos, pidx) in enumerate(members):
                theta_batch[local_idx, pidx] += shift_val
                theta_batch[G + local_idx, pidx] -= shift_val

            # Clone checkpoint, expand to batch
            tensor = checkpoints[op_k].unsqueeze(0).expand(B_group, -1, -1, -1, -1).clone()

            # Apply suffix ops (from op_k to end)
            for suffix_idx in range(op_k, len(ops)):
                op_type, payload = ops[suffix_idx]
                if op_type == 'block1q':
                    for qubit in sorted(payload.keys()):
                        fused = None
                        for g in payload[qubit]:
                            if g.has_parameter():
                                mat = g.matrix_fun(theta_batch[:, g.theta_index], self.device)
                            else:
                                mat = g.matrix_fun(B_group, self.device)
                            if mat.dim() == 2:
                                mat = mat.unsqueeze(0).expand(B_group, -1, -1)
                            fused = mat if fused is None else torch.bmm(mat, fused)
                        tensor[:, qubit] = _apply_single_qubit_gate_batch(
                            fused, tensor[:, qubit])
                else:  # '2q'
                    gate = payload
                    if gate.has_parameter():
                        gate.apply_batch(theta_batch, B_group, tensor)
                    else:
                        gate.apply_batch(B_group, tensor)

            # Store results into output at correct positions
            for local_idx, (pos, pidx) in enumerate(members):
                out[pos] = tensor[local_idx]
                out[C + pos] = tensor[G + local_idx]

        return out

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
