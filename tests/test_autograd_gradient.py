"""Tests for AutogradGradient: correctness vs parameter-shift and analytical checks."""

import pytest
import torch
import numpy as np

from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.enums import MeasureMethod
from TREV.optimization.gradients.autograd_gradient import (
    AutogradGradient,
    _build_tensor_ring_differentiable,
    _expectation_differentiable,
)
from TREV.optimization.gradients.batch_parameter_shift import BatchParameterShiftGradient


def _make_test_circuit(N=4, rank=4, device="cpu"):
    """Build a simple parameterised circuit: RY on every qubit."""
    c = Circuit(num_qubit=N, rank=rank, device=device)
    for q in range(N):
        c.ry(q)
    return c


def _make_test_hamiltonian(N=4):
    """H = sum_i Z_i."""
    h = Hamiltonian(num_qubits=N)
    for i in range(N):
        label = "I" * i + "Z" + "I" * (N - 1 - i)
        h.add_pauli(label, 1.0)
    return h


class TestAutogradVsParameterShift:
    """Compare autograd gradient to parameter-shift gradient."""

    def test_ry_circuit_gradient_matches(self):
        """RY-only circuit: autograd ≈ parameter-shift."""
        N, rank = 4, 4
        circuit = _make_test_circuit(N, rank)
        hamiltonian = _make_test_hamiltonian(N)

        torch.manual_seed(42)
        theta = torch.randn(circuit.params_size)

        autograd = AutogradGradient()
        grad_auto = autograd.run(theta, circuit, hamiltonian)

        ps = BatchParameterShiftGradient(
            shift=torch.pi / 2, batch_size=circuit.params_size,
            shots=0, measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            depth=1, is_partial=False,
        )
        grad_ps = ps.run(theta, circuit, hamiltonian)

        np.testing.assert_allclose(
            grad_auto.cpu().numpy(), grad_ps.cpu().numpy(), atol=1e-4,
            err_msg="Autograd gradient does not match parameter-shift",
        )

    def test_multi_term_hamiltonian(self):
        """Multiple Hamiltonian terms: H = ZI + IZ + ZZ."""
        N, rank = 2, 4
        circuit = Circuit(num_qubit=N, rank=rank, device="cpu")
        circuit.ry(0)
        circuit.ry(1)
        hamiltonian = Hamiltonian(num_qubits=N)
        hamiltonian.add_pauli("ZI", 0.5)
        hamiltonian.add_pauli("IZ", -0.3)
        hamiltonian.add_pauli("ZZ", 0.8)

        theta = torch.tensor([0.7, 1.2])

        grad_auto = AutogradGradient().run(theta, circuit, hamiltonian)
        ps = BatchParameterShiftGradient(
            shift=torch.pi / 2, batch_size=2,
            shots=0, measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            depth=1, is_partial=False,
        )
        grad_ps = ps.run(theta, circuit, hamiltonian)

        np.testing.assert_allclose(
            grad_auto.cpu().numpy(), grad_ps.cpu().numpy(), atol=1e-4,
        )

    def test_mixed_gates(self):
        """Circuit with RY + RX + H gates."""
        N, rank = 3, 4
        circuit = Circuit(num_qubit=N, rank=rank, device="cpu")
        circuit.h(0)
        circuit.ry(0)
        circuit.rx(1)
        circuit.rz(2)
        hamiltonian = Hamiltonian(num_qubits=N)
        hamiltonian.add_pauli("ZII", 1.0)
        hamiltonian.add_pauli("IZI", 1.0)

        theta = torch.tensor([0.5, -0.3, 1.1])

        grad_auto = AutogradGradient().run(theta, circuit, hamiltonian)
        ps = BatchParameterShiftGradient(
            shift=torch.pi / 2, batch_size=3,
            shots=0, measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            depth=1, is_partial=False,
        )
        grad_ps = ps.run(theta, circuit, hamiltonian)

        np.testing.assert_allclose(
            grad_auto.cpu().numpy(), grad_ps.cpu().numpy(), atol=1e-4,
        )


class TestAnalyticalGradient:
    """Analytical gradient checks for simple circuits."""

    def test_ry_single_qubit_z(self):
        """RY(theta)|0> with H=Z: <psi|Z|psi> = cos(theta), grad = -sin(theta)."""
        circuit = Circuit(num_qubit=1, rank=4, device="cpu")
        circuit.ry(0)
        hamiltonian = Hamiltonian(num_qubits=1)
        hamiltonian.add_pauli("Z", 1.0)

        for angle in [0.0, 0.5, 1.0, np.pi / 4, np.pi / 2, np.pi]:
            theta = torch.tensor([angle])
            grad = AutogradGradient().run(theta, circuit, hamiltonian)
            expected = -np.sin(angle)
            np.testing.assert_allclose(
                grad.item(), expected, atol=1e-5,
                err_msg=f"Analytical gradient failed at theta={angle}",
            )

    def test_ry_two_qubit_zi(self):
        """RY(theta)|00> with H=ZI: <psi|ZI|psi> = cos(theta), grad = -sin(theta)."""
        circuit = Circuit(num_qubit=2, rank=4, device="cpu")
        circuit.ry(0)
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli("ZI", 1.0)

        theta = torch.tensor([0.7])
        grad = AutogradGradient().run(theta, circuit, hamiltonian)
        expected = -np.sin(0.7)
        np.testing.assert_allclose(grad.item(), expected, atol=1e-5)


class TestGradientProperties:
    """Test gradient shape, dtype, NaN, detached properties."""

    def test_shape(self):
        circuit = _make_test_circuit(4, 4)
        hamiltonian = _make_test_hamiltonian(4)
        theta = torch.randn(circuit.params_size)
        grad = AutogradGradient().run(theta, circuit, hamiltonian)
        assert grad.shape == theta.shape

    def test_no_nan(self):
        circuit = _make_test_circuit(4, 4)
        hamiltonian = _make_test_hamiltonian(4)
        theta = torch.randn(circuit.params_size)
        grad = AutogradGradient().run(theta, circuit, hamiltonian)
        assert not torch.any(torch.isnan(grad))

    def test_detached(self):
        circuit = _make_test_circuit(4, 4)
        hamiltonian = _make_test_hamiltonian(4)
        theta = torch.randn(circuit.params_size)
        grad = AutogradGradient().run(theta, circuit, hamiltonian)
        assert not grad.requires_grad

    def test_works_under_no_grad(self):
        """AutogradGradient must work even inside torch.no_grad() (as in minimize)."""
        circuit = _make_test_circuit(4, 4)
        hamiltonian = _make_test_hamiltonian(4)
        theta = torch.randn(circuit.params_size)

        with torch.no_grad():
            grad = AutogradGradient().run(theta, circuit, hamiltonian)

        assert grad.shape == theta.shape
        assert not torch.any(torch.isnan(grad))


class TestBuildTensorRingDifferentiable:
    """Test that _build_tensor_ring_differentiable matches circuit.build_tensor."""

    def test_matches_build_tensor(self):
        circuit = _make_test_circuit(4, 4)
        theta = torch.randn(circuit.params_size)

        ring_orig = circuit.build_tensor(theta)  # (N, chi, chi, 2)
        ring_diff = _build_tensor_ring_differentiable(theta, circuit)

        np.testing.assert_allclose(
            ring_diff.detach().cpu().numpy(),
            ring_orig.detach().cpu().numpy(),
            atol=1e-6,
        )

    def test_preserves_grad(self):
        circuit = Circuit(num_qubit=2, rank=4, device="cpu")
        circuit.ry(0)
        circuit.ry(1)

        theta = torch.tensor([0.5, 1.0], requires_grad=True)
        ring = _build_tensor_ring_differentiable(theta, circuit)
        loss = ring.real.sum()
        loss.backward()
        assert theta.grad is not None
        assert not torch.all(theta.grad == 0)
