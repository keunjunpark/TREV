from typing import List
import math

import torch

from ..gates.info import Z, I, X, Y

OPS = {
    'I': I(None),
    'X': X(None),
    'Y': Y(None),
    'Z': Z(None),
}

VALID_OPS = frozenset('IXYZ')

# Encoding for get_pauli_op_tensor: I=0, X=1, Y=2, Z=3
_OP_TO_UINT8 = {'I': 0, 'X': 1, 'Y': 2, 'Z': 3}

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
            raise ValueError(f"Pauli string length {len(pauli)} != num_qubits {self.num_qubits}")
        for ch in pauli:
            if ch not in VALID_OPS:
                raise ValueError(f"Invalid Pauli operator '{ch}', must be one of {VALID_OPS}")
        self.paulis.append(pauli)
        self.coefficients.append(coefficient)
    def get_bool_pauli_tensor(self, basis='Z'):
        if basis == 'Z':
            return  torch.tensor([[1 if p[i] == 'Z' else 0  for i in range(self.num_qubits) ] for p in self.paulis],
        dtype=torch.bool)
        else:
            raise NotImplementedError()

    def get_pauli_op_tensor(self) -> torch.Tensor:
        """Return (T, N) uint8 tensor encoding Pauli operators: I=0, X=1, Y=2, Z=3."""
        return torch.tensor(
            [[_OP_TO_UINT8[ch] for ch in p] for p in self.paulis],
            dtype=torch.uint8,
        )

    @property
    def has_only_zi(self) -> bool:
        """True if all Pauli strings contain only Z and I operators."""
        return all(ch in ('Z', 'I') for p in self.paulis for ch in p)

    def pauli_string_to_matrix_torch(self,pauli: str) -> torch.Tensor:
        result = OPS[pauli[0]]
        for p in pauli[1:]:
            result = torch.kron(result, OPS[p])
        return result

    def get_qwc_groups(self):
        """Group Hamiltonian terms by qubit-wise commuting measurement bases.

        Returns list of dicts with keys:
          'term_indices': list of int (indices into self.paulis)
          'basis': str of length num_qubits ('I','X','Y','Z' per qubit)
        """
        groups = []  # list of (indices_list, basis_list)

        for t, pauli in enumerate(self.paulis):
            placed = False
            for g_indices, g_basis in groups:
                compatible = True
                for q in range(self.num_qubits):
                    if pauli[q] == 'I' or g_basis[q] == 'I':
                        continue
                    if pauli[q] != g_basis[q]:
                        compatible = False
                        break
                if compatible:
                    g_indices.append(t)
                    for q in range(self.num_qubits):
                        if g_basis[q] == 'I' and pauli[q] != 'I':
                            g_basis[q] = pauli[q]
                    placed = True
                    break

            if not placed:
                groups.append(([t], list(pauli)))

        return [{'term_indices': idx, 'basis': ''.join(b)} for idx, b in groups]

    def get_density_matrix(self):

        dim = 2 ** self.num_qubits
        rho = torch.zeros((dim, dim), dtype=torch.cfloat)

        for pauli_str, coeff in zip(self.paulis, self.coefficients):
            rho += coeff * self.pauli_string_to_matrix_torch(pauli_str)

        return rho


# --- Measurement basis rotation utilities ---

_S = 2 ** -0.5  # 1/sqrt(2)

# U such that measuring Z after applying U gives the Pauli expectation
_MEAS_ROTATIONS = {
    'X': torch.tensor([[_S, _S], [_S, -_S]], dtype=torch.cfloat),           # Hadamard
    'Y': torch.tensor([[_S, -1j * _S], [-1j * _S, _S]], dtype=torch.cfloat),  # Rx(pi/2)
}


def rotate_tensor_for_measurement(tensor, basis):
    """Apply measurement basis rotations to tensor ring cores.

    tensor: (N, chi, chi, 2) or (B, N, chi, chi, 2)
    basis: string of length N with chars from 'IXYZ'

    Returns rotated tensor (cloned). For I/Z sites, no rotation is applied.
    For X sites, Hadamard is applied. For Y sites, Rx(pi/2) is applied.
    """
    rotated = tensor.clone()
    is_batched = tensor.dim() == 5

    for i, b in enumerate(basis):
        U = _MEAS_ROTATIONS.get(b)
        if U is None:  # I or Z: no rotation needed
            continue
        U = U.to(device=tensor.device, dtype=tensor.dtype)
        # A'[..., p] = sum_d U[p, d] * A[..., d]  =>  A' = A @ U.mT
        if is_batched:
            rotated[:, i] = rotated[:, i] @ U.mT
        else:
            rotated[i] = rotated[i] @ U.mT

    return rotated

