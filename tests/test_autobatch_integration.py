import pytest
import torch
import os
from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.enums import MeasureMethod

# ✅ CHANGE THIS import to wherever your class lives
# Example (guess): from TREV.optimization.gradients.batch_parameter_shift import BatchParameterShiftGradient
from TREV.optimization.gradients.batch_parameter_shift import BatchParameterShiftGradient


@pytest.mark.parametrize(
    "measure_method",
    [
       # MeasureMethod.EFFICIENT_CONTRACTION,
        # If this is too slow in CI, comment it out:
        MeasureMethod.RIGHT_SUFFIX_SAMPLING,
        #MeasureMethod.FULL_CONTRACTION,
    ],
)
def test_auto_batch_size_end_to_end_prints_and_runs(measure_method, capsys):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1) Build a small real TREV circuit (with parameters)
    circuit = Circuit(num_qubit=4, rank=4, device=device)

    # Make sure it has params (rx/ry/rz add params)
    circuit.h(0)
    circuit.rx(1)
    circuit.ry(2)
    circuit.rz(3)
    circuit.cx(0, 3)

    theta = torch.randn(circuit.params_size, device=device)

    # ---- 2) Build a small real Hamiltonian
    h = Hamiltonian(num_qubits=4)
    h.add_pauli("ZZII", 1.0)
    h.add_pauli("IZZI", 0.5)

    # ---- 3) Create gradient object with batch_size=None (this should trigger auto-tune)
    grad = BatchParameterShiftGradient(
        shift=0.5,
        batch_size=None,          # <-- key
        shots=128,                # keep small so test is fast
        measure_method=measure_method,
        depth=1,
        is_partial=False
    )

    # ---- 4) Run once (should auto-tune + print)
    g = grad.run(theta, circuit, h)

    # ---- 5) Assertions: output + batch size is set
    assert isinstance(g, torch.Tensor)
    assert g.shape == theta.shape
    assert hasattr(grad, "batch_size")
    assert grad.batch_size is not None
    assert int(grad.batch_size) >= 1

    # ---- 6) Check the print happened (depends on your print string)
    out = capsys.readouterr().out
    print(out)
    assert "batch" in out.lower() and "selected" in out.lower()
