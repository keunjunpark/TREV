# Autograd Gradient Status & Investigation Log

## Current State

Autograd gradient works **perfectly for HEA circuits** (H + CNOT + RY/RZ) but
has issues with **transpiled QAOA/PUCCD circuits** from `from_qiskit`.

## What Works

- **HEA circuits**: `cos(autograd, param_shift) = 1.000000` at all theta scales
- **Ring and chain topologies**: both work for HEA
- **ParameterTwoQubitGate (ZZ_SWAP)**: supported in `_build_tensor_diff`
- **ParameterMultiOneQubitGate (U3)**: supported in `_build_tensor_diff`
- **qubit_perm**: applied in both `autograd_gradient` and `BatchParameterShiftGradient`
- **Contraction backward**: verified correct (individual element FD matches autograd)
- **diff_svd backward**: verified correct in isolation (cos=1.0 for rank-2 matrices)

## What Doesn't Work

- **Transpiled QAOA circuits** via `from_qiskit` + `build_parameter_mapping`:
  `cos(autograd, param_shift) ~ 0.1-0.4` (should be ~1.0)
- **Gradient norm**: autograd gives 200-28000 while param_shift gives 80-750
  (autograd gradients are 3-100x larger)
- FD of autograd's **own** forward also disagrees with autograd backward (cos=0.1)

## Root Cause Analysis

### Verified NOT the cause:

1. **qubit_perm**: Applied consistently in both methods. Tested with and without.

2. **Float precision mismatch**: Fixed by building tensor in cfloat (matching
   `build_tensor`). Tensor diff is now 0.00 between `_build_tensor_diff` and
   `build_tensor`. EV diff is ~5e-6 (cfloat vs complex128 contraction).

3. **diff_svd backward**: Correct in isolation. For a standalone matrix, the
   backward matches FD to 1e-9 precision.

4. **Contraction backward**: `_contraction_diff_vectorized` backward is correct.
   Individual tensor element gradients match FD.

5. **Missing gate types**: `ParameterTwoQubitGate` and `ParameterMultiOneQubitGate`
   are now handled in `_build_tensor_diff`.

6. **SVD degenerate singular values**: Not the cause — the forward tensors match
   exactly now (building in cfloat).

### Likely cause: Backward through `.to(dtype)` upcast + gate chain

The current flow:
```
theta (f32) -> gate matrices (cfloat) -> cores (cfloat) 
    -> diff_svd (cfloat) -> tensor (cfloat) 
    -> .to(complex128) -> contraction (complex128) -> loss (real)
```

Backward:
```
loss -> contraction grad (complex128) -> .to() backward -> tensor grad (cfloat)
    -> diff_svd backward (cfloat) -> cores grad -> gate backward -> theta grad
```

The contraction backward gives correct gradients (verified). But somewhere in
the chain from tensor grad back through the gates to theta, the gradient gets
corrupted. Possible issues:

- **Gate matrix `.to(cfloat)` in info.py**: Functions like RX, RY, RZ call
  `.type(torch.cfloat)` which casts real sin/cos to complex. This cast
  preserves gradients but may interact oddly with Wirtinger derivatives when
  the gate matrix is used in `_apply_single_qubit_gate`.

- **tensordot + moveaxis backward**: `_apply_single_qubit_gate` uses
  `torch.tensordot(gate_matrix, core, ([1], [2]))` then `moveaxis`. These
  operations' backward should be correct for complex tensors, but there may
  be edge cases.

- **Circuit structure from `from_qiskit`**: The transpiled circuit may have
  gate orderings or structures that trigger backward issues. Manual circuits
  (built with `c.zz_swap()` directly) work perfectly.

### Key observation

The autograd backward disagrees with FD of its **own** forward function. This
means the issue is in PyTorch's autograd backward, not in our forward code.
Specifically, it fails for transpiled circuits but works for manually built ones.

The difference: transpiled circuits go through `from_qiskit` which may produce
gates in a different order, use `_compile_fused_ops` differently, or create
circuit structures that trigger autograd backward bugs.

## Fixes Applied

1. **`_build_tensor_diff` in cfloat**: Matches `build_tensor`'s precision exactly.
   Tensor diff = 0.00. (Previously used complex128 which diverged.)

2. **qubit_perm in both gradient methods**: `autograd_gradient` and
   `BatchParameterShiftGradient.run()` both apply `hamiltonian.permuted()`.

3. **ParameterTwoQubitGate + ParameterMultiOneQubitGate support**: Added to
   `_build_tensor_diff` with proper wrap-around handling.

4. **Removed noise injection**: The 1e-8 noise was unnecessary once tensors
   match `build_tensor`.

## Next Steps to Investigate

1. **Compare `_build_tensor_diff` gate-by-gate** against `build_tensor` for a
   transpiled circuit — check if `_compile_fused_ops` (used by `build_tensor`)
   changes the gate application order in a way that affects backward.

2. **Test with unfused gates**: Try `fuse_zz_swap=False` and simpler basis gates
   (cx instead of swap+rzz) to isolate which gate type causes the backward failure.

3. **Minimal repro**: Find the simplest transpiled circuit where autograd backward
   fails (e.g., 2-qubit circuit from `from_qiskit`).

4. **Test `_compile_fused_ops` effect**: `build_tensor` uses fused 1-qubit gate
   blocks while `_build_tensor_diff` applies gates one by one. This may change
   the backward graph structure.

5. **Replace diff_svd with identity for non-entangling points**: At theta values
   where the SVD of the MPS has many zero SVs, the backward is ill-conditioned.

## Files Modified

- `src/TREV/optimization/gradients/autograd_gradient.py` — main autograd code
- `src/TREV/optimization/gradients/batch_parameter_shift.py` — added qubit_perm
- `src/TREV/gates/differentiable_svd.py` — custom SVD backward
- `src/TREV/transpile.py` — added build_parameter_mapping
