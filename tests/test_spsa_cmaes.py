"""
Tests for SPSA and CMA-ES derivative-free optimizers.

Verifies both optimizers can minimize a simple VQE problem:
4-qubit HEA circuit with ZZ Hamiltonian.
"""

import math
import torch
import pytest

from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.enums import MeasureMethod
from TREV.optimization.optimizer import Optimizer
from TREV.optimization.optimization import minimize
from TREV.optimization.gradients.spsa import SPSAGradient
from TREV.optimization.cma_es import CMAES, minimize_cma_es

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_circuit_and_hamiltonian(n_qubits=4, rank=4):
    """Build a simple HEA circuit and ZZ-chain Hamiltonian."""
    circuit = Circuit(n_qubits, rank=rank, device=DEVICE)
    for q in range(n_qubits):
        circuit.ry(q)
    for q in range(n_qubits - 1):
        circuit.cx(q, q + 1)
    for q in range(n_qubits):
        circuit.ry(q)

    # ZZ chain: Z_i Z_{i+1}
    terms = []
    coeffs = []
    for i in range(n_qubits - 1):
        pauli = ["I"] * n_qubits
        pauli[i] = "Z"
        pauli[i + 1] = "Z"
        terms.append("".join(pauli))
        coeffs.append(1.0)

    hamiltonian = Hamiltonian(n_qubits, terms, coeffs)
    return circuit, hamiltonian


class TestSPSA:
    def test_gradient_shape(self):
        """SPSA gradient has correct shape."""
        circuit, hamiltonian = _make_circuit_and_hamiltonian()
        theta = torch.randn(circuit.params_size, device=DEVICE)

        spsa = SPSAGradient(
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            shots=0,
            c=0.1,
        )
        grad = spsa.run(theta, circuit, hamiltonian)

        assert grad.shape == theta.shape

    def test_gradient_nonzero(self):
        """SPSA gradient is not identically zero (with high probability)."""
        circuit, hamiltonian = _make_circuit_and_hamiltonian()
        theta = torch.randn(circuit.params_size, device=DEVICE)

        spsa = SPSAGradient(
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            shots=0,
            c=0.1,
        )
        grad = spsa.run(theta, circuit, hamiltonian)

        assert grad.norm().item() > 1e-10

    def test_caches_exp_value(self):
        """SPSA caches the midpoint expectation value."""
        circuit, hamiltonian = _make_circuit_and_hamiltonian()
        theta = torch.randn(circuit.params_size, device=DEVICE)

        spsa = SPSAGradient(
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            shots=0,
            c=0.1,
        )
        spsa.run(theta, circuit, hamiltonian)

        assert spsa.last_exp_value is not None

    def test_minimize_converges(self):
        """SPSA + Adam reduces the expectation value over iterations."""
        circuit, hamiltonian = _make_circuit_and_hamiltonian()
        theta = torch.zeros(circuit.params_size, device=DEVICE)

        spsa = SPSAGradient(
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            shots=0,
            c=0.1,
        )
        optimizer = Optimizer(torch.optim.Adam, {"lr": 0.05})

        theta_opt, exp_values, _, _ = minimize(
            circuit=circuit,
            theta=theta,
            hamiltonian=hamiltonian,
            optimizer=optimizer,
            gradient=spsa,
            iteration=50,
            best_value_method="highest_probability",
        )

        # Should decrease from initial value
        assert exp_values[-1] < exp_values[0]


class TestCMAES:
    def test_single_generation(self):
        """CMA-ES runs one generation without error."""
        circuit, hamiltonian = _make_circuit_and_hamiltonian()
        theta = torch.randn(circuit.params_size, device=DEVICE)

        cma = CMAES(
            sigma=0.5,
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            shots=0,
        )

        theta_opt, exp_values, best_result, times = minimize_cma_es(
            circuit=circuit,
            theta=theta,
            hamiltonian=hamiltonian,
            cma=cma,
            generations=1,
            best_value_method="highest_probability",
        )

        assert len(exp_values) == 1
        assert theta_opt.shape == theta.shape

    def test_minimize_converges(self):
        """CMA-ES reduces the expectation value over generations."""
        circuit, hamiltonian = _make_circuit_and_hamiltonian()
        theta = torch.zeros(circuit.params_size, device=DEVICE)

        cma = CMAES(
            sigma=0.5,
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            shots=0,
        )

        theta_opt, exp_values, _, _ = minimize_cma_es(
            circuit=circuit,
            theta=theta,
            hamiltonian=hamiltonian,
            cma=cma,
            generations=30,
            best_value_method="highest_probability",
        )

        # Should find a value lower than the initial
        assert min(exp_values) < exp_values[0]

    def test_wall_clock_cap(self):
        """CMA-ES respects wall-clock cap."""
        circuit, hamiltonian = _make_circuit_and_hamiltonian()
        theta = torch.randn(circuit.params_size, device=DEVICE)

        cma = CMAES(
            sigma=0.5,
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            shots=0,
        )

        _, exp_values, _, _ = minimize_cma_es(
            circuit=circuit,
            theta=theta,
            hamiltonian=hamiltonian,
            cma=cma,
            generations=10000,
            best_value_method="highest_probability",
            wall_clock_cap=2.0,
        )

        # Should have stopped well before 10000 generations
        assert len(exp_values) < 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
