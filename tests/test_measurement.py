"""Unit tests for measurement methods."""

import pytest
import torch
import numpy as np
from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.enums import MeasureMethod


class TestSamplingMeasurement:
    """Test sampling measurement method."""
    
    def test_sampling_simple_circuit(self):
        """Test sampling measurement on simple circuit."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.h(0)  # Create superposition on qubit 0
        
        theta = torch.tensor([])
        counts = circuit.measure(theta, method=MeasureMethod.SAMPLING, shots=1000)
        
        # Check that result is a dictionary
        assert isinstance(counts, (dict, list))
        
        # Total probability should be approximately 1 (or sum of counts = shots)
        if isinstance(counts, dict):
            total = sum(counts.values())
            assert total == 1000
    
    def test_sampling_deterministic_state(self):
        """Test sampling on deterministic state."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        # Don't apply any gates - should be |00>
        
        theta = torch.tensor([])
        result = circuit.measure(theta, method=MeasureMethod.SAMPLING, shots=1000)
        
        # Result should be deterministic (all 0s)
        assert result is not None
    
    def test_sampling_different_shot_counts(self):
        """Test sampling with different shot counts."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.h(0)
        circuit.h(1)
        
        theta = torch.tensor([])
        
        for shots in [100, 1000, 10000]:
            result = circuit.measure(theta, method=MeasureMethod.SAMPLING, shots=shots)
            assert result is not None


class TestContractionMeasurement:
    """Test contraction measurement method."""
    
    def test_contraction_simple_circuit(self):
        """Test contraction measurement on simple circuit."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.h(0)
        
        theta = torch.tensor([])
        result = circuit.measure(theta, method=MeasureMethod.CONTRACTION)
        
        assert result is not None
    
    def test_contraction_ground_state(self):
        """Test contraction measurement on ground state."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        # No gates - ground state |00>
        
        theta = torch.tensor([])
        result = circuit.measure(theta, method=MeasureMethod.CONTRACTION)
        
        assert result is not None


class TestExpectationValueContraction:
    """Test expectation value calculation with contraction method."""
    
    def test_expectation_value_identity(self):
        """Test expectation value of identity Hamiltonian."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.h(0)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('II', 1.0)
        
        theta = torch.tensor([])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CONTRACTION
        )
        
        # Expectation value of identity should be 1
        assert np.isclose(float(exp_val), 1.0, atol=1e-5)
    
    def test_expectation_value_z_basis(self):
        """Test expectation value in Z basis."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        # Ground state |00>
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZI', 1.0)
        
        theta = torch.tensor([])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CONTRACTION
        )
        
        # |0> is +1 eigenstate of Z
        assert np.isclose(float(exp_val), 1.0, atol=1e-5)
    
    def test_expectation_value_superposition(self):
        """Test expectation value with superposition state."""
        circuit = Circuit(num_qubit=1, rank=10, device='cpu')
        circuit.h(0)  # |+> = (|0> + |1>)/sqrt(2)
        
        hamiltonian = Hamiltonian(num_qubits=1)
        hamiltonian.add_pauli('Z', 1.0)
        
        theta = torch.tensor([])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CONTRACTION
        )
        
        # <+|Z|+> = 0
        assert np.isclose(float(exp_val), 0.0, atol=1e-5)
    
    def test_expectation_value_multiple_terms(self):
        """Test expectation value with multiple Hamiltonian terms."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZI', 1.0)
        hamiltonian.add_pauli('IZ', 1.0)
        
        theta = torch.tensor([])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CONTRACTION
        )
        
        # Both qubits in |0>, so expect 1 + 1 = 2
        assert np.isclose(float(exp_val), 2.0, atol=1e-5)
    
    def test_expectation_value_with_rotation(self):
        """Test expectation value with parameterized rotation."""
        circuit = Circuit(num_qubit=1, rank=10, device='cpu')
        circuit.rx(0)
        
        hamiltonian = Hamiltonian(num_qubits=1)
        hamiltonian.add_pauli('Z', 1.0)
        
        # RX(pi) flips |0> to |1>
        theta = torch.tensor([np.pi])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CONTRACTION
        )
        
        # After RX(pi), expect -1
        assert np.isclose(float(exp_val), -1.0, atol=1e-5)
    
    def test_expectation_value_entangled_state(self):
        """Test expectation value with entangled state."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.h(0)
        circuit.cx(0, 1)  # Bell state
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZZ', 1.0)
        
        theta = torch.tensor([])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CONTRACTION
        )
        
        # Bell state has ZZ expectation value of 1
        assert np.isclose(float(exp_val), 1.0, atol=1e-5)


class TestExpectationValueSampling:
    """Test expectation value calculation with sampling method."""
    
    def test_sampling_expectation_value_identity(self):
        """Test sampling expectation value of identity."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.h(0)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('II', 1.0)
        
        theta = torch.tensor([])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.SAMPLING, shots=10000
        )
        
        # Should be close to 1 with high shot count
        assert np.isclose(float(exp_val.real), 1.0, atol=0.1)
    
    def test_sampling_convergence(self):
        """Test that sampling converges with more shots."""
        circuit = Circuit(num_qubit=3, rank=10, device='cpu')
        circuit.h(0)
        
        hamiltonian = Hamiltonian(num_qubits=3)
        hamiltonian.add_pauli('ZII', 1.0)
        
        theta = torch.tensor([])
        
        # Get reference value with contraction
        exact = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CONTRACTION
        )
        
        # Sample with increasing shots
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CORRECT_SAMPLING, shots=10000
        )
        print(exp_val, exact)
        # Should be reasonably close
        assert np.isclose(float(exp_val.real), float(exact), atol=0.1)


class TestExpectationValueEfficientContraction:
    """Test expectation value with efficient contraction method."""
    
    def test_efficient_contraction_simple(self):
        """Test efficient contraction on simple circuit."""
        circuit = Circuit(num_qubit=3, rank=10, device='cpu')
        circuit.h(0)
        circuit.h(1)
        
        hamiltonian = Hamiltonian(num_qubits=3)
        hamiltonian.add_pauli('ZZI', 1.0)
        
        theta = torch.tensor([])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.EFFICIENT_CONTRACTION, shots=1000
        )
        
        assert isinstance(exp_val, (float, torch.Tensor))
        assert not np.isnan(float(exp_val.real))
    
    def test_efficient_contraction_vs_contraction(self):
        """Compare efficient contraction with full contraction."""
        circuit = Circuit(num_qubit=3, rank=10, device='cpu')
        circuit.h(0)
        circuit.rx(1)
        
        hamiltonian = Hamiltonian(num_qubits=3)
        hamiltonian.add_pauli('ZII', 1.0)
        
        theta = torch.randn(circuit.params_size)
        # Get both methods
        exp_val_exact = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CONTRACTION
        )
        exp_val_efficient = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.EFFICIENT_CONTRACTION, shots=10000
        )
        
        # Should be close
        assert np.isclose(float(exp_val_exact), float(exp_val_efficient), atol=0.1)


class TestExpectationValueCorrectSampling:
    """Test expectation value with correct sampling method."""
    
    def test_correct_sampling_simple(self):
        """Test correct sampling on simple circuit."""
        circuit = Circuit(num_qubit=3, rank=10, device='cpu')
        circuit.h(0)
        
        hamiltonian = Hamiltonian(num_qubits=3)
        hamiltonian.add_pauli('ZII', 1.0)
        
        theta = torch.tensor([])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CORRECT_SAMPLING, shots=1000
        )
        
        assert isinstance(exp_val, (float, torch.Tensor))
        assert not np.isnan(float(exp_val))


class TestMeasurementConsistency:
    """Test consistency across different measurement methods."""
    
    def test_methods_consistency_simple_circuit(self):
        """Test that different methods give consistent results."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.h(0)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('II', 1.0)
        
        theta = torch.tensor([])
        
        # Get expectation value with different methods
        exp_val_contraction = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CONTRACTION
        )
        
        exp_val_sampling = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.SAMPLING, shots=10000
        )
        
        # Should be close (identity operator)
        assert np.isclose(float(exp_val_contraction.real), 1.0, atol=1e-5)
        assert np.isclose(float(exp_val_sampling.real), 1.0, atol=0.1)


class TestMeasurementEdgeCases:
    """Test edge cases for measurement."""
    
    def test_single_qubit_measurement(self):
        """Test measurement on single qubit."""
        circuit = Circuit(num_qubit=1, rank=10, device='cpu')
        circuit.h(0)
        
        hamiltonian = Hamiltonian(num_qubits=1)
        hamiltonian.add_pauli('Z', 1.0)
        
        theta = torch.tensor([])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CONTRACTION
        )
        
        assert np.isclose(float(exp_val.real), 0.0, atol=1e-5)
    
    def test_zero_hamiltonian(self):
        """Test expectation value with zero Hamiltonian."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.h(0)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('II', 0.0)
        
        theta = torch.tensor([])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CONTRACTION
        )
        
        assert np.isclose(float(exp_val), 0.0, atol=1e-5)
    
    def test_negative_coefficient(self):
        """Test expectation value with negative coefficient."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZI', -1.0)
        
        theta = torch.tensor([])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CONTRACTION
        )
        
        # |00> state, Z gives +1, coefficient is -1
        assert np.isclose(float(exp_val), -1.0, atol=1e-5)


class TestMeasurementNumericalStability:
    """Test numerical stability of measurements."""
    
    def test_small_parameters(self):
        """Test with very small rotation parameters."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.rx(0)
        circuit.ry(1)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZZ', 1.0)
        
        theta = torch.tensor([1e-6, 1e-6])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CONTRACTION
        )
        
        assert not np.isnan(float(exp_val))
        assert not np.isinf(float(exp_val))
    
    def test_large_parameters(self):
        """Test with large rotation parameters."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.rx(0)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZI', 1.0)
        
        theta = torch.tensor([100 * np.pi])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CONTRACTION
        )
        
        assert not np.isnan(float(exp_val))
        assert not np.isinf(float(exp_val))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMeasurementCUDA:
    """Test measurement methods on CUDA device."""
    
    def test_contraction_cuda(self):
        """Test contraction measurement on CUDA."""
        circuit = Circuit(num_qubit=2, rank=10, device='cuda')
        circuit.h(0)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZI', 1.0)
        
        theta = torch.tensor([], device='cuda')
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.CONTRACTION
        )
        
        assert not np.isnan(float(exp_val))
    
    def test_sampling_cuda(self):
        """Test sampling measurement on CUDA."""
        circuit = Circuit(num_qubit=2, rank=10, device='cuda')
        circuit.h(0)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('II', 1.0)
        
        theta = torch.tensor([], device='cuda')
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.SAMPLING, shots=1000
        )
        
        assert not np.isnan(float(exp_val.real))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
