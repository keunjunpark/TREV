"""Unit tests for Hamiltonian class."""

import pytest
import torch
import numpy as np
from TREV.hamiltonian.hamiltonian import Hamiltonian


class TestHamiltonianInitialization:
    """Test Hamiltonian initialization."""
    
    def test_empty_hamiltonian(self):
        """Test creating an empty Hamiltonian."""
        hamiltonian = Hamiltonian(num_qubits=4)
        assert hamiltonian.num_qubits == 4
        assert len(hamiltonian.paulis) == 0
        assert len(hamiltonian.coefficients) == 0
    
    def test_hamiltonian_with_paulis(self):
        """Test creating Hamiltonian with initial Pauli strings."""
        paulis = ['ZZ', 'XI']
        coefficients = [1.0, 0.5]
        hamiltonian = Hamiltonian(num_qubits=2, paulis=paulis, coefficients=coefficients)
        
        assert hamiltonian.num_qubits == 2
        assert len(hamiltonian.paulis) == 2
        assert len(hamiltonian.coefficients) == 2


class TestPauliStringOperations:
    """Test Pauli string operations."""
    
    def test_add_single_pauli(self):
        """Test adding a single Pauli string."""
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZI', 1.0)
        
        assert len(hamiltonian.paulis) == 1
        assert hamiltonian.paulis[0] == 'ZI'
        assert hamiltonian.coefficients[0] == 1.0
    
    def test_add_multiple_paulis(self):
        """Test adding multiple Pauli strings."""
        hamiltonian = Hamiltonian(num_qubits=3)
        hamiltonian.add_pauli('ZZI', 1.0)
        hamiltonian.add_pauli('IZZ', 0.5)
        hamiltonian.add_pauli('ZIZ', 0.3)
        
        assert len(hamiltonian.paulis) == 3
        assert len(hamiltonian.coefficients) == 3
    
    def test_add_pauli_wrong_length(self):
        """Test that adding Pauli string with wrong length raises error."""
        hamiltonian = Hamiltonian(num_qubits=3)
        
        with pytest.raises(ValueError):
            hamiltonian.add_pauli('ZZ', 1.0)  # Too short
        
        with pytest.raises(ValueError):
            hamiltonian.add_pauli('ZZII', 1.0)  # Too long
    
    def test_add_pauli_complex_coefficient(self):
        """Test adding Pauli string with complex coefficient."""
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZZ', 1.0 + 2.0j)
        
        assert hamiltonian.coefficients[0] == 1.0 + 2.0j
    
    def test_identity_pauli(self):
        """Test adding identity Pauli string."""
        hamiltonian = Hamiltonian(num_qubits=3)
        hamiltonian.add_pauli('III', 1.0)
        
        assert hamiltonian.paulis[0] == 'III'


class TestBoolPauliTensor:
    """Test boolean Pauli tensor representation."""
    
    def test_get_bool_pauli_tensor_z_basis(self):
        """Test getting boolean tensor for Z basis."""
        hamiltonian = Hamiltonian(num_qubits=3)
        hamiltonian.add_pauli('ZZI', 1.0)
        hamiltonian.add_pauli('IZZ', 1.0)
        hamiltonian.add_pauli('ZIZ', 1.0)
        
        bool_tensor = hamiltonian.get_bool_pauli_tensor(basis='Z')
        
        assert bool_tensor.shape == (3, 3)
        assert bool_tensor.dtype == torch.bool
        
        # Check first Pauli string: ZZI
        assert bool_tensor[0, 0] == True   # Z at position 0
        assert bool_tensor[0, 1] == True   # Z at position 1
        assert bool_tensor[0, 2] == False  # I at position 2
    
    def test_bool_tensor_all_identity(self):
        """Test boolean tensor for all-identity Pauli string."""
        hamiltonian = Hamiltonian(num_qubits=4)
        hamiltonian.add_pauli('IIII', 1.0)
        
        bool_tensor = hamiltonian.get_bool_pauli_tensor(basis='Z')
        
        assert torch.all(bool_tensor == False)
    
    def test_bool_tensor_all_z(self):
        """Test boolean tensor for all-Z Pauli string."""
        hamiltonian = Hamiltonian(num_qubits=3)
        hamiltonian.add_pauli('ZZZ', 1.0)
        
        bool_tensor = hamiltonian.get_bool_pauli_tensor(basis='Z')
        
        assert torch.all(bool_tensor == True)
    
    def test_bool_tensor_multiple_terms(self):
        """Test boolean tensor with multiple terms."""
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZI', 1.0)
        hamiltonian.add_pauli('IZ', 1.0)
        hamiltonian.add_pauli('ZZ', 1.0)
        
        bool_tensor = hamiltonian.get_bool_pauli_tensor(basis='Z')
        
        assert bool_tensor.shape == (3, 2)
        assert bool_tensor[0, 0] == True and bool_tensor[0, 1] == False  # ZI
        assert bool_tensor[1, 0] == False and bool_tensor[1, 1] == True  # IZ
        assert bool_tensor[2, 0] == True and bool_tensor[2, 1] == True   # ZZ


class TestPauliToMatrix:
    """Test Pauli string to matrix conversion."""
    
    def test_single_qubit_z(self):
        """Test single qubit Z operator."""
        hamiltonian = Hamiltonian(num_qubits=1)
        matrix = hamiltonian.pauli_string_to_matrix_torch('Z')
        
        expected = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.cfloat)
        assert torch.allclose(matrix, expected)
    
    def test_single_qubit_identity(self):
        """Test single qubit identity operator."""
        hamiltonian = Hamiltonian(num_qubits=1)
        matrix = hamiltonian.pauli_string_to_matrix_torch('I')
        
        expected = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.cfloat)
        assert torch.allclose(matrix, expected)
    
    def test_two_qubit_zi(self):
        """Test two-qubit ZI operator."""
        hamiltonian = Hamiltonian(num_qubits=2)
        matrix = hamiltonian.pauli_string_to_matrix_torch('ZI')
        
        assert matrix.shape == (4, 4)
        # ZI should have eigenvalues [1, 1, -1, -1]
        eigenvalues = torch.linalg.eigvalsh(matrix.real)
        expected_eigenvalues = torch.tensor([-1., -1., 1., 1.])
        assert torch.allclose(torch.sort(eigenvalues)[0], torch.sort(expected_eigenvalues)[0])
    
    def test_two_qubit_zz(self):
        """Test two-qubit ZZ operator."""
        hamiltonian = Hamiltonian(num_qubits=2)
        matrix = hamiltonian.pauli_string_to_matrix_torch('ZZ')
        
        assert matrix.shape == (4, 4)
        # ZZ should be diagonal
        assert torch.allclose(matrix, torch.diag(torch.diag(matrix)))
    
    def test_multi_qubit_identity(self):
        """Test multi-qubit identity operator."""
        hamiltonian = Hamiltonian(num_qubits=3)
        matrix = hamiltonian.pauli_string_to_matrix_torch('III')
        
        assert matrix.shape == (8, 8)
        expected = torch.eye(8, dtype=torch.cfloat)
        assert torch.allclose(matrix, expected)


class TestDensityMatrix:
    """Test density matrix generation."""
    
    def test_density_matrix_single_term(self):
        """Test density matrix with single Pauli term."""
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZI', 1.0)
        
        rho = hamiltonian.get_density_matrix()
        
        assert rho.shape == (4, 4)
        assert rho.dtype == torch.cfloat
        # Check Hermiticity
        assert torch.allclose(rho, rho.conj().T)
    
    def test_density_matrix_multiple_terms(self):
        """Test density matrix with multiple Pauli terms."""
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZI', 1.0)
        hamiltonian.add_pauli('IZ', 0.5)
        hamiltonian.add_pauli('ZZ', 0.3)
        
        rho = hamiltonian.get_density_matrix()
        
        assert rho.shape == (4, 4)
        # Check Hermiticity
        assert torch.allclose(rho, rho.conj().T)
    
    def test_density_matrix_identity(self):
        """Test density matrix for identity Hamiltonian."""
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('II', 1.0)
        
        rho = hamiltonian.get_density_matrix()
        
        expected = torch.eye(4, dtype=torch.cfloat)
        assert torch.allclose(rho, expected)
    
    def test_density_matrix_complex_coefficients(self):
        """Test density matrix with complex coefficients."""
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZI', 1.0 + 1.0j)
        
        rho = hamiltonian.get_density_matrix()
        
        assert rho.dtype == torch.cfloat
        assert not torch.all(rho.imag == 0)  # Should have imaginary parts


class TestHamiltonianProperties:
    """Test Hamiltonian properties."""
    
    def test_hermiticity_real_coefficients(self):
        """Test that Hamiltonian with real coefficients is Hermitian."""
        hamiltonian = Hamiltonian(num_qubits=3)
        hamiltonian.add_pauli('ZZI', 1.0)
        hamiltonian.add_pauli('IZZ', 0.5)
        
        rho = hamiltonian.get_density_matrix()
        assert torch.allclose(rho, rho.conj().T)
    
    def test_trace(self):
        """Test trace of Hamiltonian identity component."""
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('II', 1.0)
        
        rho = hamiltonian.get_density_matrix()
        trace = torch.trace(rho)
        
        assert torch.allclose(trace, torch.tensor(4.0 + 0.0j))


class TestMaxCutHamiltonian:
    """Test Hamiltonian for MaxCut problem."""
    
    def test_maxcut_2_qubits(self):
        """Test MaxCut Hamiltonian for 2 qubits."""
        hamiltonian = Hamiltonian(num_qubits=2)
        # MaxCut: 0.5 * (1 - Z_i Z_j) for each edge
        hamiltonian.add_pauli('II', 0.5)
        hamiltonian.add_pauli('ZZ', -0.5)
        
        assert len(hamiltonian.paulis) == 2
        rho = hamiltonian.get_density_matrix()
        assert rho.shape == (4, 4)
    
    def test_maxcut_3_qubits_ring(self):
        """Test MaxCut Hamiltonian for 3-qubit ring."""
        hamiltonian = Hamiltonian(num_qubits=3)
        # Ring: edges (0,1), (1,2), (2,0)
        edges = [(0, 1), (1, 2), (2, 0)]
        
        hamiltonian.add_pauli('III', 0.5 * len(edges))
        hamiltonian.add_pauli('ZZI', -0.5)  # edge (0,1)
        hamiltonian.add_pauli('IZZ', -0.5)  # edge (1,2)
        hamiltonian.add_pauli('ZIZ', -0.5)  # edge (0,2)
        
        assert len(hamiltonian.paulis) == 4
        rho = hamiltonian.get_density_matrix()
        assert torch.allclose(rho, rho.conj().T)


class TestHamiltonianEdgeCases:
    """Test edge cases for Hamiltonian."""
    
    def test_zero_coefficient(self):
        """Test adding Pauli string with zero coefficient."""
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZZ', 0.0)
        
        assert len(hamiltonian.paulis) == 1
        assert hamiltonian.coefficients[0] == 0.0
    
    def test_negative_coefficient(self):
        """Test adding Pauli string with negative coefficient."""
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZZ', -1.0)
        
        assert hamiltonian.coefficients[0] == -1.0
    
    def test_large_hamiltonian(self):
        """Test Hamiltonian with many terms."""
        num_qubits = 4
        hamiltonian = Hamiltonian(num_qubits=num_qubits)
        
        # Add many terms
        for i in range(num_qubits - 1):
            pauli = ['I'] * num_qubits
            pauli[i] = 'Z'
            pauli[i+1] = 'Z'
            hamiltonian.add_pauli(''.join(pauli), 1.0)
        
        assert len(hamiltonian.paulis) == num_qubits - 1
        bool_tensor = hamiltonian.get_bool_pauli_tensor(basis='Z')
        assert bool_tensor.shape == (num_qubits - 1, num_qubits)
    
    def test_single_qubit_hamiltonian(self):
        """Test single qubit Hamiltonian."""
        hamiltonian = Hamiltonian(num_qubits=1)
        hamiltonian.add_pauli('Z', 1.0)
        
        rho = hamiltonian.get_density_matrix()
        assert rho.shape == (2, 2)
        
        expected = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.cfloat)
        assert torch.allclose(rho, expected)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
