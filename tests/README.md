# TREV Unit Tests

This directory contains comprehensive unit tests for the TREV quantum computing library.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest configuration and fixtures
├── requirements.txt         # Test dependencies
├── test_circuit.py          # Tests for Circuit class
├── test_hamiltonian.py      # Tests for Hamiltonian class
├── test_gates.py            # Tests for quantum gates
└── test_measurement.py      # Tests for measurement methods
```

## Test Coverage

### test_circuit.py
- Circuit initialization and configuration
- Non-parameterized gates (H, X, Y, Z, CNOT, SWAP)
- Parameterized gates (RX, RY, RZ)
- Tensor building (single and batch)
- Measurement operations
- Expectation value calculations
- Complex circuit structures (QAOA-like, deep circuits)
- Gradient computation
- CUDA support

### test_hamiltonian.py
- Hamiltonian initialization
- Pauli string operations
- Boolean tensor representations
- Pauli string to matrix conversion
- Density matrix generation
- MaxCut problem Hamiltonians
- Edge cases and numerical stability

### test_gates.py
- Pauli gates (I, X, Y, Z)
- Hadamard gate
- Rotation gates (RX, RY, RZ)
- Two-qubit gates (CNOT, SWAP)
- Batch operations
- Gate properties (unitarity, commutation relations)
- Gate eigenvalues
- CUDA support

### test_measurement.py
- Perfect Sampling measurement
- Full contraction measurement
- Efficient contraction method
- Right suffix sampling method
- Expectation value calculations
- Consistency across methods
- Edge cases and numerical stability
- CUDA support

## Running Tests

### Install Test Dependencies

```bash
pip install -r tests/requirements.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test File

```bash
pytest tests/test_circuit.py
```

### Run Specific Test Class

```bash
pytest tests/test_circuit.py::TestCircuitInitialization
```

### Run Specific Test

```bash
pytest tests/test_circuit.py::TestCircuitInitialization::test_circuit_creation
```

### Run with Verbose Output

```bash
pytest -v
```

### Run with Coverage Report

```bash
pytest --cov=src/TREV --cov-report=html
```

### Run Tests in Parallel

```bash
pytest -n auto
```

### Skip Slow Tests

```bash
pytest -m "not slow"
```

### Skip CUDA Tests

```bash
pytest -m "not cuda"
```

### Run Only CUDA Tests

```bash
pytest -m cuda
```

## Test Markers

Tests are organized with the following markers:

- `@pytest.mark.slow` - Tests that take longer to run
- `@pytest.mark.cuda` - Tests that require CUDA/GPU
- `@pytest.mark.integration` - Integration tests

## Fixtures

Common fixtures available in `conftest.py`:

- `set_random_seed` - Sets random seed for reproducibility
- `cpu_device` - Provides CPU device string
- `cuda_device` - Provides CUDA device string (skips if CUDA unavailable)
- `small_circuit` - Provides a small test circuit
- `simple_hamiltonian` - Provides a simple test Hamiltonian
- `small_theta` - Provides small random parameters

## Continuous Integration

These tests are designed to be run in CI/CD pipelines. Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.8'
      - name: Install dependencies
        run: |
          pip install -r tests/requirements.txt
          pip install -e .
      - name: Run tests
        run: pytest --cov=src/TREV --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Writing New Tests

When adding new features to TREV, please add corresponding tests:

1. Create test methods in the appropriate test file
2. Use descriptive test names starting with `test_`
3. Add docstrings explaining what the test does
4. Use appropriate assertions
5. Add markers for slow or CUDA-dependent tests
6. Test edge cases and error conditions

Example test structure:

```python
class TestNewFeature:
    """Test new feature functionality."""
    
    def test_basic_functionality(self):
        """Test basic usage of new feature."""
        # Arrange
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        
        # Act
        result = circuit.new_feature()
        
        # Assert
        assert result is not None
    
    @pytest.mark.slow
    def test_large_scale(self):
        """Test new feature at large scale."""
        # Test with large parameters
        pass
```

## Troubleshooting

### Import Errors

If you get import errors, make sure TREV is installed:

```bash
pip install -e .
```

### CUDA Tests Failing

If CUDA tests fail, they may be skipped automatically if CUDA is not available. To check CUDA availability:

```python
import torch
print(torch.cuda.is_available())
```

### Slow Tests

Some tests may be slow. Use the `-m "not slow"` flag to skip them during development.

## Contributing

When contributing tests:

1. Ensure all tests pass before submitting PR
2. Add tests for new features
3. Maintain test coverage above 80%
4. Follow existing test patterns and naming conventions
5. Document complex test logic with comments
