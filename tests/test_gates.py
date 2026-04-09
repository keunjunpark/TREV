"""Unit tests for gate implementations."""

import pytest
import torch
import numpy as np
from TREV.gates.info import I, X, Y, Z, H, RX, RY, RZ, CNOT, SWAP
from TREV.gates.non_parameter_gates import NonParameterOneQubitGate, NonParameterTwoQubitsGate
from TREV.gates.parameter_gates import ParameterOneQubitGate


class TestPauliGates:
    """Test Pauli gate matrices."""
    
    def test_identity_gate(self):
        """Test identity gate matrix."""
        matrix = I(device='cpu')
        expected = torch.eye(2, dtype=torch.cfloat)
        assert torch.allclose(matrix, expected)
        assert matrix.shape == (2, 2)
    
    def test_x_gate(self):
        """Test Pauli-X gate matrix."""
        matrix = X(device='cpu')
        expected = torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat)
        assert torch.allclose(matrix, expected)
    
    def test_y_gate(self):
        """Test Pauli-Y gate matrix."""
        matrix = Y(device='cpu')
        expected = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat)
        assert torch.allclose(matrix, expected)
    
    def test_z_gate(self):
        """Test Pauli-Z gate matrix."""
        matrix = Z(device='cpu')
        expected = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat)
        assert torch.allclose(matrix, expected)
    
    def test_pauli_gates_unitarity(self):
        """Test that Pauli gates are unitary."""
        for gate in [I, X, Y, Z]:
            matrix = gate(device='cpu')
            product = matrix @ matrix.conj().T
            expected = torch.eye(2, dtype=torch.cfloat)
            assert torch.allclose(product, expected, atol=1e-6)


class TestHadamardGate:
    """Test Hadamard gate."""
    
    def test_hadamard_matrix(self):
        """Test Hadamard gate matrix."""
        matrix = H(device='cpu')
        expected = (1 / np.sqrt(2)) * torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat)
        assert torch.allclose(matrix, expected)
    
    def test_hadamard_unitarity(self):
        """Test that Hadamard gate is unitary."""
        matrix = H(device='cpu')
        product = matrix @ matrix.conj().T
        expected = torch.eye(2, dtype=torch.cfloat)
        assert torch.allclose(product, expected, atol=1e-6)
    
    def test_hadamard_self_inverse(self):
        """Test that H^2 = I."""
        matrix = H(device='cpu')
        product = matrix @ matrix
        expected = torch.eye(2, dtype=torch.cfloat)
        assert torch.allclose(product, expected, atol=1e-6)


class TestRotationGates:
    """Test rotation gates."""
    
    def test_rx_gate_zero(self):
        """Test RX gate with zero angle."""
        matrix = RX(torch.tensor(0.0), device='cpu')
        expected = torch.eye(2, dtype=torch.cfloat)
        assert torch.allclose(matrix, expected, atol=1e-6)
    
    def test_rx_gate_pi(self):
        """Test RX gate with pi angle."""
        matrix = RX(torch.tensor(np.pi), device='cpu')
        expected = -1j * X(device='cpu')
        assert torch.allclose(matrix, expected, atol=1e-6)
    
    def test_ry_gate_zero(self):
        """Test RY gate with zero angle."""
        matrix = RY(torch.tensor(0.0), device='cpu')
        expected = torch.eye(2, dtype=torch.cfloat)
        assert torch.allclose(matrix, expected, atol=1e-6)
    
    def test_ry_gate_pi(self):
        """Test RY gate with pi angle."""
        matrix = RY(torch.tensor(np.pi), device='cpu')
        # RY(pi) = [[cos(pi/2), -sin(pi/2)], [sin(pi/2), cos(pi/2)]] = [[0, -1], [1, 0]]
        expected = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.cfloat)
        assert torch.allclose(matrix, expected, atol=1e-6)
    
    def test_rz_gate_zero(self):
        """Test RZ gate with zero angle."""
        matrix = RZ(torch.tensor(0.0), device='cpu')
        expected = torch.eye(2, dtype=torch.cfloat)
        assert torch.allclose(matrix, expected, atol=1e-6)
    
    def test_rz_gate_pi(self):
        """Test RZ gate with pi angle."""
        matrix = RZ(torch.tensor(np.pi), device='cpu')
        # RZ(pi) = diag(e^(-i*pi/2), e^(i*pi/2)) = diag(-i, i)
        expected = torch.diag(torch.tensor([-1j, 1j], dtype=torch.cfloat))
        assert torch.allclose(matrix, expected, atol=1e-6)
    
    def test_rotation_gates_unitarity(self):
        """Test that rotation gates are unitary."""
        angles = [0.0, np.pi/4, np.pi/2, np.pi]
        
        for angle in angles:
            theta = torch.tensor(angle)
            for gate_func in [RX, RY, RZ]:
                matrix = gate_func(theta, device='cpu')
                product = matrix @ matrix.conj().T
                expected = torch.eye(2, dtype=torch.cfloat)
                assert torch.allclose(product, expected, atol=1e-6)
    
    def test_rx_batch(self):
        """Test RX gate with batch of angles."""
        batch_size = 3
        thetas = torch.tensor([0.0, np.pi/2, np.pi])
        matrices = RX(thetas, device='cpu')
        
        assert matrices.shape == (batch_size, 2, 2)
        
        # Check each matrix is unitary
        for i in range(batch_size):
            product = matrices[i] @ matrices[i].conj().T
            expected = torch.eye(2, dtype=torch.cfloat)
            assert torch.allclose(product, expected, atol=1e-6)
    
    def test_ry_batch(self):
        """Test RY gate with batch of angles."""
        batch_size = 3
        thetas = torch.tensor([0.0, np.pi/2, np.pi])
        matrices = RY(thetas, device='cpu')
        
        assert matrices.shape == (batch_size, 2, 2)
    
    def test_rz_batch(self):
        """Test RZ gate with batch of angles."""
        batch_size = 3
        thetas = torch.tensor([0.0, np.pi/2, np.pi])
        matrices = RZ(thetas, device='cpu')
        
        assert matrices.shape == (batch_size, 2, 2)


class TestTwoQubitGates:
    """Test two-qubit gates."""
    
    def test_cnot_matrix(self):
        """Test CNOT gate matrix."""
        matrix = CNOT(device='cpu')
        expected = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=torch.cfloat)
        assert torch.allclose(matrix, expected)
        assert matrix.shape == (4, 4)
    
    def test_cnot_unitarity(self):
        """Test that CNOT gate is unitary."""
        matrix = CNOT(device='cpu')
        product = matrix @ matrix.conj().T
        expected = torch.eye(4, dtype=torch.cfloat)
        assert torch.allclose(product, expected, atol=1e-6)
    
    def test_cnot_self_inverse(self):
        """Test that CNOT is self-inverse."""
        matrix = CNOT(device='cpu')
        product = matrix @ matrix
        expected = torch.eye(4, dtype=torch.cfloat)
        assert torch.allclose(product, expected, atol=1e-6)
    
    def test_cnot_batch(self):
        """Test CNOT gate with batch."""
        batch_size = 5
        matrices = CNOT(batch_size=batch_size, device='cpu')
        
        assert matrices.shape == (batch_size, 4, 4)
        
        # Check each matrix is the same
        for i in range(batch_size):
            expected = CNOT(device='cpu')
            assert torch.allclose(matrices[i], expected)
    
    def test_swap_matrix(self):
        """Test SWAP gate matrix."""
        matrix = SWAP(device='cpu')
        expected = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=torch.cfloat)
        assert torch.allclose(matrix, expected)
        assert matrix.shape == (4, 4)
    
    def test_swap_unitarity(self):
        """Test that SWAP gate is unitary."""
        matrix = SWAP(device='cpu')
        product = matrix @ matrix.conj().T
        expected = torch.eye(4, dtype=torch.cfloat)
        assert torch.allclose(product, expected, atol=1e-6)
    
    def test_swap_self_inverse(self):
        """Test that SWAP is self-inverse."""
        matrix = SWAP(device='cpu')
        product = matrix @ matrix
        expected = torch.eye(4, dtype=torch.cfloat)
        assert torch.allclose(product, expected, atol=1e-6)


class TestBatchGates:
    """Test batch operations for gates."""
    
    def test_identity_batch(self):
        """Test identity gate in batch mode."""
        batch_size = 10
        matrices = I(batch_size=batch_size, device='cpu')
        
        assert matrices.shape == (batch_size, 2, 2)
        
        for i in range(batch_size):
            expected = torch.eye(2, dtype=torch.cfloat)
            assert torch.allclose(matrices[i], expected)
    
    def test_pauli_x_batch(self):
        """Test Pauli-X gate in batch mode."""
        batch_size = 5
        matrices = X(batch_size=batch_size, device='cpu')
        
        assert matrices.shape == (batch_size, 2, 2)
    
    def test_hadamard_batch(self):
        """Test Hadamard gate in batch mode."""
        batch_size = 5
        matrices = H(batch_size=batch_size, device='cpu')
        
        assert matrices.shape == (batch_size, 2, 2)
        
        # Check unitarity for each matrix in batch
        for i in range(batch_size):
            product = matrices[i] @ matrices[i].conj().T
            expected = torch.eye(2, dtype=torch.cfloat)
            assert torch.allclose(product, expected, atol=1e-6)


class TestGateProperties:
    """Test mathematical properties of gates."""
    
    def test_pauli_commutation_relations(self):
        """Test Pauli commutation relations."""
        x = X(device='cpu')
        y = Y(device='cpu')
        z = Z(device='cpu')
        
        # [X, Y] = 2iZ (anti-commutator)
        comm_xy = x @ y - y @ x
        expected_xy = 2j * z
        assert torch.allclose(comm_xy, expected_xy, atol=1e-6)
        
        # [Y, Z] = 2iX
        comm_yz = y @ z - z @ y
        expected_yz = 2j * x
        assert torch.allclose(comm_yz, expected_yz, atol=1e-6)
        
        # [Z, X] = 2iY
        comm_zx = z @ x - x @ z
        expected_zx = 2j * y
        assert torch.allclose(comm_zx, expected_zx, atol=1e-6)
    
    def test_pauli_anticommutation(self):
        """Test Pauli anticommutation relations."""
        x = X(device='cpu')
        y = Y(device='cpu')
        z = Z(device='cpu')
        
        # {X, Y} = XY + YX = 0
        anticomm_xy = x @ y + y @ x
        assert torch.allclose(anticomm_xy, torch.zeros(2, 2, dtype=torch.cfloat), atol=1e-6)
        
        # {Y, Z} = 0
        anticomm_yz = y @ z + z @ y
        assert torch.allclose(anticomm_yz, torch.zeros(2, 2, dtype=torch.cfloat), atol=1e-6)
        
        # {Z, X} = 0
        anticomm_zx = z @ x + x @ z
        assert torch.allclose(anticomm_zx, torch.zeros(2, 2, dtype=torch.cfloat), atol=1e-6)
    
    def test_hadamard_pauli_relation(self):
        """Test Hadamard transformation of Pauli gates."""
        h = H(device='cpu')
        x = X(device='cpu')
        z = Z(device='cpu')
        
        # HXH = Z
        hxh = h @ x @ h
        assert torch.allclose(hxh, z, atol=1e-6)
        
        # HZH = X
        hzh = h @ z @ h
        assert torch.allclose(hzh, x, atol=1e-6)
    
    def test_rotation_commutation(self):
        """Test commutation of rotation gates."""
        theta = torch.tensor(np.pi/4)
        
        rx = RX(theta, device='cpu')
        ry = RY(theta, device='cpu')
        
        # RX and RY don't commute in general
        prod1 = rx @ ry
        prod2 = ry @ rx
        
        # They should not be equal (unless theta is special)
        assert not torch.allclose(prod1, prod2, atol=1e-6)


class TestGateEigenvalues:
    """Test eigenvalues of gates."""
    
    def test_pauli_eigenvalues(self):
        """Test that Pauli gates have eigenvalues ±1."""
        for gate_func in [X, Y, Z]:
            matrix = gate_func(device='cpu')
            eigenvalues = torch.linalg.eigvals(matrix)
            
            # Sort eigenvalues
            eigenvalues_sorted = torch.sort(eigenvalues.real)[0]
            expected = torch.tensor([-1.0, 1.0])
            
            assert torch.allclose(eigenvalues_sorted, expected, atol=1e-6)
    
    def test_hadamard_eigenvalues(self):
        """Test Hadamard gate eigenvalues."""
        matrix = H(device='cpu')
        eigenvalues = torch.linalg.eigvals(matrix)
        
        # Eigenvalues should be ±1
        eigenvalues_sorted = torch.sort(eigenvalues.real)[0]
        expected = torch.tensor([-1.0, 1.0])
        
        assert torch.allclose(eigenvalues_sorted, expected, atol=1e-6)
    
    def test_rotation_eigenvalues_magnitude(self):
        """Test that rotation gates have unit magnitude eigenvalues."""
        angle = np.pi/3
        theta = torch.tensor(angle)
        
        for gate_func in [RX, RY, RZ]:
            matrix = gate_func(theta, device='cpu')
            eigenvalues = torch.linalg.eigvals(matrix)
            
            # All eigenvalues should have magnitude 1 (unitary)
            magnitudes = torch.abs(eigenvalues)
            assert torch.allclose(magnitudes, torch.ones(2), atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAGates:
    """Test gate operations on CUDA device."""
    
    def test_pauli_gates_cuda(self):
        """Test Pauli gates on CUDA."""
        for gate_func in [I, X, Y, Z]:
            matrix = gate_func(device='cuda')
            assert matrix.device.type == 'cuda'
            assert matrix.shape == (2, 2)
    
    def test_hadamard_cuda(self):
        """Test Hadamard gate on CUDA."""
        matrix = H(device='cuda')
        assert matrix.device.type == 'cuda'
    
    def test_rotation_gates_cuda(self):
        """Test rotation gates on CUDA."""
        theta = torch.tensor(np.pi/4, device='cuda')
        
        for gate_func in [RX, RY, RZ]:
            matrix = gate_func(theta, device='cuda')
            assert matrix.device.type == 'cuda'
    
    def test_two_qubit_gates_cuda(self):
        """Test two-qubit gates on CUDA."""
        cnot = CNOT(device='cuda')
        assert cnot.device.type == 'cuda'
        
        swap = SWAP(device='cuda')
        assert swap.device.type == 'cuda'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
