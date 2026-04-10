# Changelog

## [Unreleased]

### Performance
- **Right suffix sampling: 6x faster R_suf precompute at chi=32** — Replaced O(chi^6) explicit `kron` + matrix multiplication in `precompute_double_layer_and_right_suffix` with Kronecker-factored O(chi^5) contraction. Uses the same algebraic identity already applied in the batched code path. Overall RSS speedup: ~2-3x at chi=32. No accuracy change (R_suf matrices are numerically identical, max error ~1e-16).

- **Batched efficient contraction: O(chi^5) Kronecker factoring** — Replaced O(chi^6) `Prod @ E` matrix multiplication in `expectation_value_batch_efficient_contraction` with Kronecker-factored right-multiply using per-site A0/A1 slices. Eliminates need to build full chi^2 x chi^2 double-layer matrices (E_I, E_Z). Correctness verified against single-parameter exact contraction (max err ~1e-6).

- **Batched correct sampling R_suf: O(chi^5)** — Applied same Kronecker factoring to `expectation_value_batch_correct_sampling` R_suf precompute, replacing `torch.bmm(Ei, acc)` with O(chi^5) matmuls. R_suf numerically identical (max err ~1e-8 in cfloat).

- **Faster SVD post-processing in 2-qubit gates** — Replaced `diag_embed(s) + bmm` with broadcast multiply (`x * s.unsqueeze()`) for absorbing singular values. Eliminates diagonal matrix allocation and batched matmul. ~7-37% speedup on `build_tensor_batch`.

### Fixed
- **QAOA example in README** — Added ZZ entangling gates (CNOT-RZ-CNOT) and full MaxCut problem setup. Previous example had no entanglement.
