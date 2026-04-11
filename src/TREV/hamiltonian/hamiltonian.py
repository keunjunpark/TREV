from typing import List

import torch

from ..gates.info import Z, I, X, Y

OPS = {
    'I': I(None),
    'X': X(None),
    'Y': Y(None),
    'Z': Z(None),
}

_PAULI_TO_OP = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}

class Hamiltonian():
    def __init__(self, num_qubits:int, paulis:List[str]= None, coefficients:List[complex]=None):
        if paulis is None:
            paulis = []
        if coefficients is None:
            coefficients = []
        self.paulis:List[str] = paulis
        self.coefficients:List[complex] = coefficients
        self.num_qubits = num_qubits

    def add_pauli(self, pauli, coefficient):
        if len(pauli) != self.num_qubits:
            raise ValueError()
        self.paulis.append(pauli)
        self.coefficients.append(coefficient)
    def get_bool_pauli_tensor(self, basis='Z'):
        if basis == 'Z':
            return  torch.tensor([[1 if p[i] == 'Z' else 0  for i in range(self.num_qubits) ] for p in self.paulis],
        dtype=torch.bool)
        else:
            raise NotImplementedError()

    def get_pauli_op_tensor(self):
        """Returns (T, N) uint8 tensor: 0=I, 1=X, 2=Y, 3=Z."""
        return torch.tensor(
            [[_PAULI_TO_OP[p[i]] for i in range(self.num_qubits)] for p in self.paulis],
            dtype=torch.uint8,
        )

    @property
    def has_only_zi(self):
        """True if Hamiltonian only uses Z and I operators."""
        return all(c in 'IZ' for p in self.paulis for c in p)

    def pauli_string_to_matrix_torch(self,pauli: str) -> torch.Tensor:
        result = OPS[pauli[0]]
        for p in pauli[1:]:
            result = torch.kron(result, OPS[p])
        return result

    def get_density_matrix(self):

        dim = 2 ** self.num_qubits
        rho = torch.zeros((dim, dim), dtype=torch.cfloat)

        for pauli_str, coeff in zip(self.paulis, self.coefficients):
            rho += coeff * self.pauli_string_to_matrix_torch(pauli_str)

        return rho

