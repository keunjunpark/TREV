# TREV: Python Library for Efficient Implementations of Variational Quantum Algorithms for Optimization using Tensor Networks

TREV is a high-performance quantum computing simulation framework implemented in PyTorch, designed for variational quantum algorithms using tensor ring (periodic Matrix Product State) representations. The framework provides multiple measurement strategies and optimization methods for quantum circuit simulation.

## Features

- **Tensor Ring Architecture**: Efficient quantum state representation using periodic Matrix Product States
- **Multiple Measurement Methods**: 
  - Direct tensor contraction
  - Monte Carlo sampling
  - Efficient batched contraction
  - Corrected sampling algorithms
- **Quantum Gates**: Support for common quantum gates (Pauli gates, rotation gates, CNOT, SWAP)
- **Hamiltonian Operations**: Full support for Pauli string Hamiltonians
- **Optimization**: Built-in gradient computation and optimization algorithms
- **GPU Acceleration**: Full CUDA support for high-performance computation
- **Qiskit Integration**: Utilities for working with Qiskit quantum circuits

## Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- NumPy
- Qiskit (for utility functions)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TREV
```

2. Install dependencies:
```bash
pip install torch numpy qiskit
```

3. Install TREV:
```bash
pip install -e .
```

## Quick Start

### Basic Circuit Creation

```python
import torch
from TREV import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.enums import MeasureMethod

# Create a 4-qubit circuit
circuit = Circuit(num_qubit=4, rank=10, device='cuda')

# Add quantum gates
circuit.h(0)  # Hadamard gate on qubit 0
circuit.rx(1) # Parameterized X-rotation on qubit 1
circuit.ry(2) # Parameterized Y-rotation on qubit 2
circuit.cx(0, 3)  # CNOT gate

# Define parameters
theta = torch.randn(circuit.params_size, device='cuda', requires_grad=True)
```

### Hamiltonian Definition

```python
# Create a Hamiltonian for MaxCut problem
hamiltonian = Hamiltonian(num_qubits=4)
hamiltonian.add_pauli('ZZII', 1.0)  # Z⊗Z⊗I⊗I with coefficient 1.0
hamiltonian.add_pauli('IZZI', 0.5)  # I⊗Z⊗Z⊗I with coefficient 0.5
```

### Expectation Value Calculation

```python
# Calculate expectation value using different methods
exp_val_contraction = circuit.get_expectation_value(
    theta, hamiltonian, MeasureMethod.CONTRACTION
)

exp_val_sampling = circuit.get_expectation_value(
    theta, hamiltonian, MeasureMethod.SAMPLING, shots=10000
)

exp_val_efficient = circuit.get_expectation_value(
    theta, hamiltonian, MeasureMethod.EFFICIENT_CONTRACTION
)
```

### Optimization

```python
from TREV.optimization.optimization import minimize
from TREV.optimization.optimizer import Optimizer
from TREV.optimization.gradients.vanilla_parameter_shift import VanillaParameterShift

# Set up optimization
optimizer = Optimizer('Adam', lr=0.01)
gradient = VanillaParameterShift(MeasureMethod.EFFICIENT_CONTRACTION, shots=1000)

# Run optimization
result = minimize(
    circuit=circuit,
    theta=theta,
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    gradient=gradient,
    iteration=100,
    best_value_method='full_contraction'
)
```

## Architecture

### Core Components

#### `Circuit` Class
The main class for building and simulating quantum circuits:
- **Gate Operations**: `h()`, `x()`, `y()`, `z()`, `rx()`, `ry()`, `rz()`, `cx()`, `swap()`
- **Tensor Building**: `build_tensor()`, `build_tensor_batch()`
- **Measurement**: `measure()`, `get_expectation_value()`

#### Measurement Methods

1. **Contraction** (`MeasureMethod.CONTRACTION`)
   - Exact tensor contraction for small systems
   - Provides exact results but scales exponentially

2. **Sampling** (`MeasureMethod.SAMPLING`)
   - Monte Carlo sampling from quantum state
   - Scalable but provides statistical estimates

3. **Efficient Contraction** (`MeasureMethod.EFFICIENT_CONTRACTION`)
   - Optimized batched contraction method
   - Balance between accuracy and efficiency

4. **Correct Sampling** (`MeasureMethod.CORRECT_SAMPLING`)
   - Advanced sampling with error correction
   - High accuracy with controlled statistical errors

#### Hamiltonian Support
- **Pauli Strings**: Support for arbitrary Pauli string operators
- **Coefficients**: Complex coefficient support
- **Efficient Representation**: Boolean tensor representation for fast computation

### Module Structure

```
TREV/
├── circuit.py              # Main Circuit class
├── gates/                  # Quantum gate implementations
│   ├── gates.py           # Base gate classes
│   ├── parameter_gates.py # Parameterized gates
│   ├── non_parameter_gates.py # Fixed gates
│   └── info.py           # Gate definitions
├── hamiltonian/           # Hamiltonian operations
│   └── hamiltonian.py    # Hamiltonian class
├── measure/               # Measurement methods
│   ├── contraction.py    # Tensor contraction
│   ├── sampling.py       # Monte Carlo sampling
│   ├── efficient_contraction.py # Efficient methods
│   ├── correct_sampling.py # Advanced sampling
│   └── enums.py         # Method enumerations
├── optimization/         # Optimization algorithms
│   ├── optimization.py  # Main optimization routines
│   ├── optimizer.py     # Optimizer wrappers
│   └── gradients/       # Gradient computation
└── utils/               # Utility functions
    └── maxcut.py       # MaxCut problem utilities
```

## Advanced Usage

### Custom Hamiltonians

```python
from TREV.utils.maxcut import gengraph, create_hamiltonian, make_hamiltonian

# Generate random MaxCut instance
edges = gengraph(n=6)  # 6-node graph
ham_terms = create_hamiltonian(6, edges)

# Convert to TREV Hamiltonian
hamiltonian = Hamiltonian(num_qubits=6)
for coeff, pauli_tuple in ham_terms:
    pauli_string = ''.join('Z' if b else 'I' for b in pauli_tuple)
    hamiltonian.add_pauli(pauli_string, coeff)
```

### Batch Processing

```python
# Process multiple parameter sets simultaneously
batch_size = 32
theta_batch = torch.randn(batch_size, circuit.params_size, device='cuda')

# Build tensors for entire batch
tensor_batch = circuit.build_tensor_batch(theta_batch, batch_size)

# Compute expectation values
from TREV.measure.efficient_contraction import expectation_value_batch
exp_vals = expectation_value_batch(tensor_batch, hamiltonian, device='cuda')
```

### Gradient Computation

```python
from TREV.optimization.gradients.batch_parameter_shift import BatchParameterShift

# Use batch parameter shift for efficient gradients
gradient_method = BatchParameterShift(
    MeasureMethod.EFFICIENT_CONTRACTION,
    shots=1000,
    batch_size=16
)

# Compute gradients
grad = gradient_method.run(theta, circuit, hamiltonian)
```

## Examples

### Quantum Approximate Optimization Algorithm (QAOA)

```python
# QAOA for MaxCut
def create_qaoa_circuit(num_qubits, p_layers):
    circuit = Circuit(num_qubits, rank=10)
    
    # Initial state preparation
    for i in range(num_qubits):
        circuit.h(i)
    
    # QAOA layers
    for p in range(p_layers):
        # Problem Hamiltonian evolution
        for i in range(num_qubits):
            circuit.rz(i)  # γ parameter
        
        # Mixer Hamiltonian evolution  
        for i in range(num_qubits):
            circuit.rx(i)  # β parameter
    
    return circuit

# Create and optimize QAOA circuit
qaoa_circuit = create_qaoa_circuit(6, p_layers=3)
theta = torch.randn(qaoa_circuit.params_size, requires_grad=True)

# Optimize
result = minimize(qaoa_circuit, theta, hamiltonian, optimizer, gradient, 200)
```

## Performance

TREV is optimized for high-performance quantum simulation:

- **GPU Acceleration**: Full CUDA support for tensor operations
- **Memory Efficiency**: Tensor ring representation reduces memory footprint
- **Parallel Processing**: Batch operations for multiple parameter sets
- **Scalable Algorithms**: Multiple measurement methods for different system sizes

### Recommended Usage:
- **Small systems (≤12 qubits)**: Use `CONTRACTION` for exact results
- **Medium systems (12-20 qubits)**: Use `EFFICIENT_CONTRACTION` 
- **Large systems (>20 qubits)**: Use `SAMPLING` or `CORRECT_SAMPLING`

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest features.

## Acknowledgments

This framework builds upon advances in tensor network methods for quantum simulation and variational quantum algorithms.
