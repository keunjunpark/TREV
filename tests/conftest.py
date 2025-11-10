"""Pytest configuration and fixtures for TREV tests."""

import pytest
import torch
import numpy as np


@pytest.fixture
def set_random_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # Cleanup after test
    torch.manual_seed(torch.initial_seed())


@pytest.fixture
def cpu_device():
    """Fixture for CPU device."""
    return 'cpu'


@pytest.fixture
def cuda_device():
    """Fixture for CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return 'cuda'


@pytest.fixture
def small_circuit():
    """Fixture providing a small test circuit."""
    from TREV.circuit import Circuit
    circuit = Circuit(num_qubit=2, rank=10, device='cpu')
    return circuit


@pytest.fixture
def simple_hamiltonian():
    """Fixture providing a simple test Hamiltonian."""
    from TREV.hamiltonian.hamiltonian import Hamiltonian
    hamiltonian = Hamiltonian(num_qubits=2)
    hamiltonian.add_pauli('ZZ', 1.0)
    return hamiltonian


@pytest.fixture
def small_theta():
    """Fixture providing small random parameters."""
    return torch.randn(2)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "cuda: marks tests that require CUDA"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add slow marker to tests with many iterations
        if "large" in item.nodeid or "deep" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add cuda marker to CUDA tests
        if "cuda" in item.nodeid.lower() or "gpu" in item.nodeid.lower():
            item.add_marker(pytest.mark.cuda)
