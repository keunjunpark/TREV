import pytest
import torch

from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.enums import MeasureMethod
from TREV.optimization.gradients.batch_parameter_shift import (
    BatchParameterShiftGradient,
    batch_gradient,
    _get_gpu_count,
)


def _make_circuit_and_hamiltonian(device="cpu", num_qubit=4, rank=4):
    circuit = Circuit(num_qubit=num_qubit, rank=rank, device=device)
    circuit.h(0)
    circuit.rx(0)
    circuit.ry(1)
    circuit.rz(2)
    circuit.rx(3)
    circuit.cx(0, 1)
    circuit.ry(0)
    circuit.rz(1)

    h = Hamiltonian(num_qubits=num_qubit)
    h.add_pauli("ZZII", 1.0)
    h.add_pauli("IZZI", 0.5)
    h.add_pauli("IIZZ", -0.3)

    theta = torch.randn(circuit.params_size, device=device)
    return circuit, h, theta


# ---------- Circuit.to_device tests ----------

class TestToDevice:
    def test_to_device_preserves_structure(self):
        circuit = Circuit(num_qubit=4, rank=8, device="cpu")
        circuit.h(0)
        circuit.rx(1)
        circuit.ry(2)
        circuit.cx(0, 3)

        clone = circuit.to_device("cpu")

        assert clone.device == "cpu"
        assert clone.num_qubit == circuit.num_qubit
        assert clone.rank == circuit.rank
        assert clone.params_size == circuit.params_size
        assert len(clone.gates) == len(circuit.gates)
        # Gates should be different objects
        for g_orig, g_clone in zip(circuit.gates, clone.gates):
            assert g_orig is not g_clone
            assert g_clone.device == "cpu"

    def test_to_device_build_tensor_batch_same_output(self):
        circuit, _, theta = _make_circuit_and_hamiltonian(device="cpu")
        clone = circuit.to_device("cpu")

        batch_theta = theta.unsqueeze(0).expand(3, -1)
        t1 = circuit.build_tensor_batch(batch_theta, 3)
        t2 = clone.build_tensor_batch(batch_theta, 3)

        assert torch.allclose(t1, t2, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_cuda(self):
        circuit = Circuit(num_qubit=3, rank=4, device="cuda:0")
        circuit.rx(0)
        circuit.ry(1)
        circuit.cx(0, 1)

        # Clone to same device (should still work)
        clone = circuit.to_device("cuda:0")
        assert clone.device == "cuda:0"

        theta = torch.randn(circuit.params_size, device="cuda:0")
        batch_theta = theta.unsqueeze(0).expand(2, -1)

        t1 = circuit.build_tensor_batch(batch_theta, 2)
        t2 = clone.build_tensor_batch(batch_theta, 2)
        assert torch.allclose(t1, t2, atol=1e-6)

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Need 2+ GPUs"
    )
    def test_to_device_cross_gpu(self):
        circuit = Circuit(num_qubit=3, rank=4, device="cuda:0")
        circuit.rx(0)
        circuit.ry(1)

        clone = circuit.to_device("cuda:1")
        assert clone.device == "cuda:1"

        theta = torch.randn(circuit.params_size, device="cuda:0")
        batch_theta = theta.unsqueeze(0).expand(2, -1)

        t_orig = circuit.build_tensor_batch(batch_theta.to("cuda:0"), 2)
        t_clone = clone.build_tensor_batch(batch_theta.to("cuda:1"), 2)

        assert t_orig.device == torch.device("cuda:0")
        assert t_clone.device == torch.device("cuda:1")
        assert torch.allclose(t_orig.cpu(), t_clone.cpu(), atol=1e-6)


# ---------- Multi-GPU gradient tests ----------

class TestMultiGPUGradient:
    def test_single_gpu_fallback_cpu(self):
        """num_gpus=1 should use the single-device code path on CPU."""
        circuit, h, theta = _make_circuit_and_hamiltonian(device="cpu")

        grad = batch_gradient(
            theta, circuit, h,
            chunk_size=2, shots=100, shift=0.5,
            depth=1, curr_depth=0, is_partial=False,
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            num_gpus=1,
        )
        assert grad.shape == theta.shape
        assert grad.device == theta.device

    def test_num_gpus_zero_uses_single_path(self):
        """num_gpus=0 should fall through to single-device path."""
        circuit, h, theta = _make_circuit_and_hamiltonian(device="cpu")

        grad = batch_gradient(
            theta, circuit, h,
            chunk_size=2, shots=100, shift=0.5,
            depth=1, curr_depth=0, is_partial=False,
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            num_gpus=0,
        )
        assert grad.shape == theta.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_gpu_explicit(self):
        """Explicit num_gpus=1 on CUDA should work."""
        circuit, h, theta = _make_circuit_and_hamiltonian(device="cuda")

        grad = batch_gradient(
            theta, circuit, h,
            chunk_size=3, shots=100, shift=0.5,
            depth=1, curr_depth=0, is_partial=False,
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            num_gpus=1,
        )
        assert grad.shape == theta.shape

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Need 2+ GPUs"
    )
    def test_multi_gpu_matches_single_gpu(self):
        """Multi-GPU gradient should match single-GPU within float tolerance."""
        circuit, h, theta = _make_circuit_and_hamiltonian(device="cuda:0")

        grad_1gpu = batch_gradient(
            theta, circuit, h,
            chunk_size=2, shots=100, shift=0.5,
            depth=1, curr_depth=0, is_partial=False,
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            num_gpus=1,
        )

        grad_2gpu = batch_gradient(
            theta, circuit, h,
            chunk_size=2, shots=100, shift=0.5,
            depth=1, curr_depth=0, is_partial=False,
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            num_gpus=2,
        )

        assert torch.allclose(grad_1gpu, grad_2gpu, atol=1e-5), \
            f"Max diff: {(grad_1gpu - grad_2gpu).abs().max().item()}"

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Need 2+ GPUs"
    )
    def test_multi_gpu_fewer_params_than_gpus(self):
        """P < num_gpus should still work (some GPUs get no work)."""
        # Circuit with only 1 parameter
        circuit = Circuit(num_qubit=2, rank=4, device="cuda:0")
        circuit.rx(0)

        h = Hamiltonian(num_qubits=2)
        h.add_pauli("ZI", 1.0)

        theta = torch.randn(circuit.params_size, device="cuda:0")

        grad = batch_gradient(
            theta, circuit, h,
            chunk_size=1, shots=100, shift=0.5,
            depth=1, curr_depth=0, is_partial=False,
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            num_gpus=4,  # more GPUs than params
        )
        assert grad.shape == theta.shape


# ---------- BatchParameterShiftGradient integration ----------

class TestBatchParameterShiftGradientMultiGPU:
    def test_num_gpus_stored(self):
        grad = BatchParameterShiftGradient(
            shift=0.5, batch_size=4, shots=100,
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            depth=1, num_gpus=3,
        )
        assert grad._num_gpus == 3

    def test_num_gpus_auto_detect(self):
        grad = BatchParameterShiftGradient(
            shift=0.5, batch_size=4, shots=100,
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            depth=1,
        )
        assert grad._num_gpus == _get_gpu_count()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_end_to_end_single_gpu(self):
        circuit, h, theta = _make_circuit_and_hamiltonian(device="cuda")

        grad_obj = BatchParameterShiftGradient(
            shift=0.5, batch_size=3, shots=100,
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            depth=1, num_gpus=1,
        )
        g = grad_obj.run(theta, circuit, h)
        assert g.shape == theta.shape

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Need 2+ GPUs"
    )
    def test_end_to_end_multi_gpu(self):
        circuit, h, theta = _make_circuit_and_hamiltonian(device="cuda:0")

        grad_obj = BatchParameterShiftGradient(
            shift=0.5, batch_size=2, shots=100,
            measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
            depth=1, num_gpus=2,
        )
        g = grad_obj.run(theta, circuit, h)
        assert g.shape == theta.shape


# ---------- PERFECT_SAMPLING multi-GPU tests ----------

class TestPerfectSamplingMultiGPU:
    def test_single_gpu_perfect_sampling_cpu(self):
        """PERFECT_SAMPLING with num_gpus=1 on CPU should produce a valid gradient."""
        circuit, h, theta = _make_circuit_and_hamiltonian(device="cpu")

        grad = batch_gradient(
            theta, circuit, h,
            chunk_size=2, shots=100, shift=0.5,
            depth=1, curr_depth=0, is_partial=False,
            measure_method=MeasureMethod.PERFECT_SAMPLING,
            num_gpus=1,
        )
        assert grad.shape == theta.shape
        assert grad.device == theta.device

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Need 2+ GPUs"
    )
    def test_multi_gpu_perfect_sampling(self):
        """PERFECT_SAMPLING gradient across 2 GPUs should produce a valid gradient."""
        circuit, h, theta = _make_circuit_and_hamiltonian(device="cuda:0")

        grad = batch_gradient(
            theta, circuit, h,
            chunk_size=2, shots=100, shift=0.5,
            depth=1, curr_depth=0, is_partial=False,
            measure_method=MeasureMethod.PERFECT_SAMPLING,
            num_gpus=2,
        )
        assert grad.shape == theta.shape


# ---------- Cross-method accuracy tests ----------

HIGH_SHOTS = 10_000
SAMPLING_ATOL = 0.1


def _exact_gradient(circuit, h, theta):
    """Compute exact gradient via EFFICIENT_CONTRACTION as ground truth."""
    return batch_gradient(
        theta, circuit, h,
        chunk_size=2, shots=100, shift=0.5,
        depth=1, curr_depth=0, is_partial=False,
        measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
        num_gpus=1,
    )


class TestCrossMethodAccuracy:
    def test_perfect_sampling_matches_exact_cpu(self):
        """PERFECT_SAMPLING gradient should approximate EFFICIENT_CONTRACTION on CPU."""
        circuit, h, theta = _make_circuit_and_hamiltonian(device="cpu")

        grad_exact = _exact_gradient(circuit, h, theta)
        grad_ps = batch_gradient(
            theta, circuit, h,
            chunk_size=2, shots=HIGH_SHOTS, shift=0.5,
            depth=1, curr_depth=0, is_partial=False,
            measure_method=MeasureMethod.PERFECT_SAMPLING,
            num_gpus=1,
        )
        assert torch.allclose(grad_exact, grad_ps, atol=SAMPLING_ATOL), \
            f"Max diff: {(grad_exact - grad_ps).abs().max().item():.4f}"

    def test_right_suffix_sampling_matches_exact_cpu(self):
        """RIGHT_SUFFIX_SAMPLING gradient should approximate EFFICIENT_CONTRACTION on CPU."""
        circuit, h, theta = _make_circuit_and_hamiltonian(device="cpu")

        grad_exact = _exact_gradient(circuit, h, theta)
        grad_rs = batch_gradient(
            theta, circuit, h,
            chunk_size=2, shots=HIGH_SHOTS, shift=0.5,
            depth=1, curr_depth=0, is_partial=False,
            measure_method=MeasureMethod.RIGHT_SUFFIX_SAMPLING,
            num_gpus=1,
        )
        assert torch.allclose(grad_exact, grad_rs, atol=SAMPLING_ATOL), \
            f"Max diff: {(grad_exact - grad_rs).abs().max().item():.4f}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_perfect_sampling_matches_exact_cuda(self):
        """PERFECT_SAMPLING gradient should approximate EFFICIENT_CONTRACTION on CUDA."""
        circuit, h, theta = _make_circuit_and_hamiltonian(device="cuda")

        grad_exact = _exact_gradient(circuit, h, theta)
        grad_ps = batch_gradient(
            theta, circuit, h,
            chunk_size=2, shots=HIGH_SHOTS, shift=0.5,
            depth=1, curr_depth=0, is_partial=False,
            measure_method=MeasureMethod.PERFECT_SAMPLING,
            num_gpus=1,
        )
        assert torch.allclose(grad_exact, grad_ps, atol=SAMPLING_ATOL), \
            f"Max diff: {(grad_exact - grad_ps).abs().max().item():.4f}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_right_suffix_sampling_matches_exact_cuda(self):
        """RIGHT_SUFFIX_SAMPLING gradient should approximate EFFICIENT_CONTRACTION on CUDA."""
        circuit, h, theta = _make_circuit_and_hamiltonian(device="cuda")

        grad_exact = _exact_gradient(circuit, h, theta)
        grad_rs = batch_gradient(
            theta, circuit, h,
            chunk_size=2, shots=HIGH_SHOTS, shift=0.5,
            depth=1, curr_depth=0, is_partial=False,
            measure_method=MeasureMethod.RIGHT_SUFFIX_SAMPLING,
            num_gpus=1,
        )
        assert torch.allclose(grad_exact, grad_rs, atol=SAMPLING_ATOL), \
            f"Max diff: {(grad_exact - grad_rs).abs().max().item():.4f}"


# ---------- Single vs multi-GPU consistency ----------

class TestMultiGPUSamplingConsistency:
    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Need 2+ GPUs"
    )
    def test_multi_gpu_perfect_sampling_matches_exact(self):
        """Multi-GPU PERFECT_SAMPLING should approximate single-GPU exact gradient."""
        circuit, h, theta = _make_circuit_and_hamiltonian(device="cuda:0")

        grad_exact = _exact_gradient(circuit, h, theta)
        grad_ps = batch_gradient(
            theta, circuit, h,
            chunk_size=2, shots=HIGH_SHOTS, shift=0.5,
            depth=1, curr_depth=0, is_partial=False,
            measure_method=MeasureMethod.PERFECT_SAMPLING,
            num_gpus=2,
        )
        assert torch.allclose(grad_exact, grad_ps, atol=SAMPLING_ATOL), \
            f"Max diff: {(grad_exact - grad_ps).abs().max().item():.4f}"

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Need 2+ GPUs"
    )
    def test_multi_gpu_right_suffix_sampling_matches_exact(self):
        """Multi-GPU RIGHT_SUFFIX_SAMPLING should approximate single-GPU exact gradient."""
        circuit, h, theta = _make_circuit_and_hamiltonian(device="cuda:0")

        grad_exact = _exact_gradient(circuit, h, theta)
        grad_rs = batch_gradient(
            theta, circuit, h,
            chunk_size=2, shots=HIGH_SHOTS, shift=0.5,
            depth=1, curr_depth=0, is_partial=False,
            measure_method=MeasureMethod.RIGHT_SUFFIX_SAMPLING,
            num_gpus=2,
        )
        assert torch.allclose(grad_exact, grad_rs, atol=SAMPLING_ATOL), \
            f"Max diff: {(grad_exact - grad_rs).abs().max().item():.4f}"
