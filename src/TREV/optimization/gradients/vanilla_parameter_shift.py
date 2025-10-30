import torch
from torch import Tensor

from ...circuit import Circuit
from ...hamiltonian.hamiltonian import Hamiltonian
from ...measure.enums import MeasureMethod


def vanilla_parameter_shift(
        params:     torch.Tensor,           # (P,)
        circuit : Circuit,
        hamiltonian: Hamiltonian,
        chunk_size: int,
        shots: int,):
    gradients = torch.zeros_like(params).to(circuit.device)
    shift = torch.tensor(torch.pi / 2).to(circuit.device)
    for i in range(len(params)):
        params_forward = params.clone().to(circuit.device)
        params_backward = params.clone().to(circuit.device)
        params_forward[i] += shift
        params_backward[i] -= shift
        gradients[i] = (circuit.get_expectation_value(params_forward, hamiltonian, MeasureMethod.SAMPLING, int(shots)) -
                        circuit.get_expectation_value(params_backward, hamiltonian, MeasureMethod.SAMPLING, int(shots))) / 2
    return gradients