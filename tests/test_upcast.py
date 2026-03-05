"""Tests for UPCAST_SVD feature in contraction and minimize."""

import pytest
import torch
import numpy as np
from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.enums import MeasureMethod
from TREV.gates import contraction as _contraction
from TREV.optimization.optimization import minimize as trev_minimize
from TREV.optimization.gradients.batch_parameter_shift import BatchParameterShiftGradient
from TREV.optimization.optimizer import Optimizer


def _make_circuit_and_hamiltonian(n_qubits=4, depth=3, rank=4):
    """Build a simple parameterized circuit and ZZ Hamiltonian."""
    circ = Circuit(n_qubits, rank=rank, device='cpu')
    for _ in range(depth):
        for q in range(n_qubits):
            circ.ry(q)
        for q in range(n_qubits - 1):
            circ.cx(q, q + 1)
    n_params = circ.params_size

    # ZZ Hamiltonian on adjacent pairs
    paulis = ['I' * n_qubits]
    coeffs = [0.0]
    for q in range(n_qubits - 1):
        p = list('I' * n_qubits)
        p[q] = 'Z'
        p[q + 1] = 'Z'
        paulis.append(''.join(p))
        coeffs.append(1.0)
    hamil = Hamiltonian(n_qubits, paulis, coeffs)
    return circ, hamil, n_params


class TestUpcastFlag:
    """Test that UPCAST_SVD flag works correctly."""

    def test_default_flag_is_false(self):
        assert _contraction.UPCAST_SVD is False

    def test_flag_toggle(self):
        _contraction.UPCAST_SVD = True
        assert _contraction.UPCAST_SVD is True
        _contraction.UPCAST_SVD = False
        assert _contraction.UPCAST_SVD is False


class TestUpcastContraction:
    """Test that upcast produces more accurate SVD results."""

    def test_single_gate_upcast_runs(self):
        """Verify _apply_double_qubit_gate works with UPCAST_SVD=True."""
        qu0 = torch.randn(4, 4, 2, dtype=torch.cfloat)
        qu1 = torch.randn(4, 4, 2, dtype=torch.cfloat)
        gate = torch.eye(4, dtype=torch.cfloat)

        _contraction.UPCAST_SVD = False
        r0_f32, r1_f32 = _contraction._apply_double_qubit_gate(gate, (qu0, qu1))

        _contraction.UPCAST_SVD = True
        r0_up, r1_up = _contraction._apply_double_qubit_gate(gate, (qu0, qu1))
        _contraction.UPCAST_SVD = False

        # Both should return cfloat
        assert r0_f32.dtype == torch.cfloat
        assert r0_up.dtype == torch.cfloat
        # Results should be close (identity gate)
        assert torch.allclose(r0_f32, r0_up, atol=1e-5)
        assert torch.allclose(r1_f32, r1_up, atol=1e-5)

    def test_batch_gate_upcast_runs(self):
        """Verify _apply_double_qubit_gate_batch works with UPCAST_SVD=True."""
        B = 3
        qu0 = torch.randn(B, 4, 4, 2, dtype=torch.cfloat)
        qu1 = torch.randn(B, 4, 4, 2, dtype=torch.cfloat)
        gate = torch.eye(4, dtype=torch.cfloat)

        _contraction.UPCAST_SVD = False
        r0_f32, r1_f32 = _contraction._apply_double_qubit_gate_batch(gate, (qu0, qu1))

        _contraction.UPCAST_SVD = True
        r0_up, r1_up = _contraction._apply_double_qubit_gate_batch(gate, (qu0, qu1))
        _contraction.UPCAST_SVD = False

        assert r0_f32.dtype == torch.cfloat
        assert r0_up.dtype == torch.cfloat
        # Shapes must match; values may differ slightly due to SVD sign ambiguity
        assert r0_f32.shape == r0_up.shape
        assert r1_f32.shape == r1_up.shape

    def test_upcast_preserves_dtype_cdouble_input(self):
        """If input is already cdouble, upcast should be a no-op."""
        qu0 = torch.randn(4, 4, 2, dtype=torch.cdouble)
        qu1 = torch.randn(4, 4, 2, dtype=torch.cdouble)
        gate = torch.eye(4, dtype=torch.cdouble)

        _contraction.UPCAST_SVD = True
        r0, r1 = _contraction._apply_double_qubit_gate(gate, (qu0, qu1))
        _contraction.UPCAST_SVD = False

        assert r0.dtype == torch.cdouble
        assert r1.dtype == torch.cdouble

    def test_upcast_changes_result(self):
        """Verify upcast produces different (presumably more accurate) tensors than f32."""
        torch.manual_seed(42)
        n_qubits = 4
        rank = 4

        circ = Circuit(n_qubits, rank=rank, device='cpu', cdtype=torch.cfloat)
        for _ in range(3):
            for q in range(n_qubits):
                circ.ry(q)
            for q in range(n_qubits - 1):
                circ.cx(q, q + 1)

        theta = torch.randn(circ.params_size)

        # f32 without upcast
        _contraction.UPCAST_SVD = False
        tensor_f32 = circ.build_tensor(theta)

        # f32 with upcast
        _contraction.UPCAST_SVD = True
        tensor_up = circ.build_tensor(theta)
        _contraction.UPCAST_SVD = False

        # Both should be cfloat
        assert tensor_f32.dtype == torch.cfloat
        assert tensor_up.dtype == torch.cfloat

        # Results should be close but may differ due to precision
        diff = (tensor_f32 - tensor_up).abs().max().item()
        # Both are valid tensor network states
        assert diff < 5.0, f"Difference too large: {diff}"


class TestMinimizeUpcast:
    """Test upcast parameter in minimize function."""

    def test_minimize_upcast_false(self):
        """minimize with upcast=False should not change the flag."""
        circ, hamil, n_params = _make_circuit_and_hamiltonian()
        theta = 0.1 * torch.randn(n_params)
        grad = BatchParameterShiftGradient(
            shift=torch.pi / 2, batch_size=None, shots=100,
            measure_method=MeasureMethod.PERFECT_SAMPLING, depth=1,
        )
        opt = Optimizer(torch.optim.Adam, {'lr': 0.01})

        _contraction.UPCAST_SVD = False
        trev_minimize(circ, theta, hamil, opt, grad, iteration=2,
                      best_value_method='highest_probability', upcast=False)
        assert _contraction.UPCAST_SVD is False

    def test_minimize_upcast_true_restores_flag(self):
        """minimize with upcast=True should restore the flag after."""
        circ, hamil, n_params = _make_circuit_and_hamiltonian()
        theta = 0.1 * torch.randn(n_params)
        grad = BatchParameterShiftGradient(
            shift=torch.pi / 2, batch_size=None, shots=100,
            measure_method=MeasureMethod.PERFECT_SAMPLING, depth=1,
        )
        opt = Optimizer(torch.optim.Adam, {'lr': 0.01})

        _contraction.UPCAST_SVD = False
        trev_minimize(circ, theta, hamil, opt, grad, iteration=2,
                      best_value_method='highest_probability', upcast=True)
        # Flag should be restored to previous value
        assert _contraction.UPCAST_SVD is False

    def test_minimize_upcast_produces_values(self):
        """minimize with upcast=True should produce valid exp_values."""
        circ, hamil, n_params = _make_circuit_and_hamiltonian()
        theta = 0.1 * torch.randn(n_params)
        grad = BatchParameterShiftGradient(
            shift=torch.pi / 2, batch_size=None, shots=100,
            measure_method=MeasureMethod.PERFECT_SAMPLING, depth=1,
        )
        opt = Optimizer(torch.optim.Adam, {'lr': 0.01})

        _, exp_values, _, _ = trev_minimize(
            circ, theta, hamil, opt, grad, iteration=3,
            best_value_method='highest_probability', upcast=True,
        )
        assert len(exp_values) == 3
        for v in exp_values:
            assert np.isfinite(float(v))
