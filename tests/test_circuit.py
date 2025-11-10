"""Unit tests for Circuit class."""

import pytest
import torch
import numpy as np
from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.enums import MeasureMethod


class TestCircuitInitialization:
    """Test Circuit initialization."""
    
    def test_circuit_creation(self):
        """Test basic circuit creation."""
        circuit = Circuit(num_qubit=4, rank=10, device='cpu')
        assert circuit.num_qubit == 4
        assert circuit.rank == 10
        assert circuit.device == 'cpu'
        assert len(circuit.gates) == 0
        assert circuit.params_size == 0
    
    def test_circuit_different_ranks(self):
        """Test circuit with different rank values."""
        for rank in [5, 10, 20]:
            circuit = Circuit(num_qubit=3, rank=rank, device='cpu')
            assert circuit.rank == rank


class TestNonParameterGates:
    """Test non-parameterized gate operations."""
    
    def test_identity_gate(self):
        """Test identity gate addition."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.id(0)
        assert len(circuit.gates) == 1
        assert circuit.params_size == 0
    
    def test_hadamard_gate(self):
        """Test Hadamard gate addition."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.h(0)
        circuit.h(1)
        assert len(circuit.gates) == 2
        assert circuit.params_size == 0
    
    def test_pauli_gates(self):
        """Test Pauli X, Y, Z gates."""
        circuit = Circuit(num_qubit=3, rank=10, device='cpu')
        circuit.x(0)
        circuit.y(1)
        circuit.z(2)
        assert len(circuit.gates) == 3
        assert circuit.params_size == 0
    
    def test_cnot_gate(self):
        """Test CNOT gate addition."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.cx(0, 1)
        assert len(circuit.gates) == 1
        assert circuit.params_size == 0
    
    def test_swap_gate(self):
        """Test SWAP gate addition."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.swap(0, 1)
        assert len(circuit.gates) == 1
        assert circuit.params_size == 0


class TestParameterGates:
    """Test parameterized gate operations."""
    
    def test_rx_gate(self):
        """Test RX rotation gate."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.rx(0)
        assert len(circuit.gates) == 1
        assert circuit.params_size == 1
    
    def test_ry_gate(self):
        """Test RY rotation gate."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.ry(0)
        assert len(circuit.gates) == 1
        assert circuit.params_size == 1
    
    def test_rz_gate(self):
        """Test RZ rotation gate."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.rz(0)
        assert len(circuit.gates) == 1
        assert circuit.params_size == 1
    
    def test_multiple_parameter_gates(self):
        """Test multiple parameterized gates."""
        circuit = Circuit(num_qubit=3, rank=10, device='cpu')
        circuit.rx(0)
        circuit.ry(1)
        circuit.rz(2)
        assert len(circuit.gates) == 3
        assert circuit.params_size == 3
    
    def test_mixed_gates(self):
        """Test mixture of parameterized and non-parameterized gates."""
        circuit = Circuit(num_qubit=3, rank=10, device='cpu')
        circuit.h(0)
        circuit.rx(1)
        circuit.cx(0, 1)
        circuit.ry(2)
        assert len(circuit.gates) == 4
        assert circuit.params_size == 2


class TestTensorBuilding:
    """Test tensor building operations."""
    
    def test_build_tensor_shape(self):
        """Test that build_tensor returns correct shape."""
        circuit = Circuit(num_qubit=3, rank=10, device='cpu')
        circuit.h(0)
        circuit.rx(1)
        
        theta = torch.randn(circuit.params_size)
        tensor = circuit.build_tensor(theta)
        
        assert tensor.shape == (3, 10, 10, 2)
        assert tensor.dtype == torch.cfloat
    
    def test_build_tensor_initial_state(self):
        """Test initial state of empty circuit."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        theta = torch.tensor([])
        tensor = circuit.build_tensor(theta)
        
        # Check that initial state is |00...0>
        assert tensor[0, 0, 0, 0] == 1.0
        assert tensor[1, 0, 0, 0] == 1.0
    
    def test_build_tensor_batch_shape(self):
        """Test batch tensor building."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.rx(0)
        circuit.ry(1)
        
        batch_size = 5
        theta = torch.randn(batch_size, circuit.params_size)
        tensor = circuit.build_tensor_batch(theta, batch_size)
        
        assert tensor.shape == (batch_size, 2, 10, 10, 2)
        assert tensor.dtype == torch.cfloat
    
    def test_build_tensor_with_parameters(self):
        """Test tensor building with various parameters."""
        circuit = Circuit(num_qubit=2, rank=5, device='cpu')
        circuit.rx(0)
        circuit.ry(1)
        
        theta = torch.tensor([np.pi/4, np.pi/2])
        tensor = circuit.build_tensor(theta)
        
        assert tensor.shape == (2, 5, 5, 2)
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()


class TestMeasurement:
    """Test measurement operations."""
    
    def test_measure_sampling(self):
        """Test measurement with sampling method."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.h(0)
        
        theta = torch.tensor([])
        prob_dist = circuit.measure(theta, method=MeasureMethod.PERFECT_SAMPLING, shots=1000)

        # Results should be a probability distribution
        assert isinstance(prob_dist, list)
        assert abs(sum(prob_dist) - 1.0) < 1e-9  # Allow small floating-point error

    def test_measure_contraction(self):
        """Test measurement with contraction method."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')       
        circuit.h(0)
        
        theta = torch.tensor([])
        result = circuit.measure(theta, method=MeasureMethod.FULL_CONTRACTION)
        
        assert result is not None


class TestExpectationValue:
    """Test expectation value calculations."""
    
    def test_expectation_value_simple_hamiltonian(self):
        """Test expectation value with simple Hamiltonian."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.h(0)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZI', 1.0)
        
        theta = torch.tensor([])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.FULL_CONTRACTION
        )
        
        assert isinstance(exp_val, (float, torch.Tensor))
        assert not np.isnan(float(exp_val))
    
    def test_expectation_value_with_parameters(self):
        """Test expectation value with parameterized circuit."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.rx(0)
        circuit.ry(1)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZZ', 1.0)
        
        theta = torch.randn(circuit.params_size)
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.FULL_CONTRACTION
        )
        
        assert isinstance(exp_val, (float, torch.Tensor))
        assert not np.isnan(float(exp_val))
    
    def test_expectation_value_sampling_method(self):
        """Test expectation value with sampling method."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.h(0)
        circuit.h(1)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZZ', 1.0)
        
        theta = torch.tensor([])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.PERFECT_SAMPLING, shots=10000
        )
        
        assert isinstance(exp_val, (complex, torch.Tensor))
    
    def test_expectation_value_efficient_contraction(self):
        """Test expectation value with efficient contraction method."""
        circuit = Circuit(num_qubit=3, rank=10, device='cpu')
        circuit.h(0)
        circuit.rx(1)
        
        hamiltonian = Hamiltonian(num_qubits=3)
        hamiltonian.add_pauli('ZZI', 1.0)
        
        theta = torch.randn(circuit.params_size)
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.EFFICIENT_CONTRACTION, shots=1000
        )
        
        assert isinstance(exp_val, (float, torch.Tensor))


class TestCircuitComplexity:
    """Test complex circuit configurations."""
    
    def test_qaoa_like_circuit(self):
        """Test QAOA-like circuit structure."""
        num_qubits = 4
        circuit = Circuit(num_qubit=num_qubits, rank=10, device='cpu')
        
        # Initial Hadamard layer
        for i in range(num_qubits):
            circuit.h(i)
        
        # Problem layer
        for i in range(num_qubits):
            circuit.rz(i)
        
        # Mixer layer
        for i in range(num_qubits):
            circuit.rx(i)
        
        assert circuit.params_size == 2 * num_qubits
        assert len(circuit.gates) == 3 * num_qubits
    
    def test_entangling_circuit(self):
        """Test circuit with entangling gates."""
        circuit = Circuit(num_qubit=4, rank=10, device='cpu')
        
        # Create entangled state
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        
        theta = torch.tensor([])
        tensor = circuit.build_tensor(theta)
        
        assert tensor.shape == (4, 10, 10, 2)
    
    def test_deep_circuit(self):
        """Test deep circuit with many layers."""
        circuit = Circuit(num_qubit=3, rank=10, device='cpu')
        
        for _ in range(10):  # 10 layers
            for i in range(3):
                circuit.ry(i)
            for i in range(2):
                circuit.cx(i, i+1)
        
        assert circuit.params_size == 30  # 10 layers * 3 qubits
        
        theta = torch.randn(circuit.params_size)
        tensor = circuit.build_tensor(theta)
        assert not torch.isnan(tensor).any()


class TestGradients:
    """Test gradient computation."""
    
    def test_parameter_gradient(self):
        """Test that parameters support gradient computation."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.rx(0)
        circuit.ry(1)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZZ', 1.0)
        
        theta = torch.randn(circuit.params_size, requires_grad=True)
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.FULL_CONTRACTION
        )
        
        # Check that we can compute gradients
        if isinstance(exp_val, torch.Tensor):
            exp_val.backward()
            assert theta.grad is not None
            assert theta.grad.shape == theta.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDASupport:
    """Test CUDA device support."""
    
    def test_cuda_circuit_creation(self):
        """Test circuit creation on CUDA device."""
        circuit = Circuit(num_qubit=2, rank=10, device='cuda')
        assert circuit.device == 'cuda'
    
    def test_cuda_tensor_building(self):
        """Test tensor building on CUDA."""
        circuit = Circuit(num_qubit=2, rank=10, device='cuda')
        circuit.h(0)
        circuit.rx(1)
        
        theta = torch.randn(circuit.params_size, device='cuda')
        tensor = circuit.build_tensor(theta)
        
        assert tensor.device.type == 'cuda'
    
    def test_cuda_expectation_value(self):
        """Test expectation value computation on CUDA."""
        circuit = Circuit(num_qubit=2, rank=10, device='cuda')
        circuit.rx(0)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZI', 1.0)
        
        theta = torch.randn(circuit.params_size, device='cuda')
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.FULL_CONTRACTION
        )
        
        assert not np.isnan(float(exp_val))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
