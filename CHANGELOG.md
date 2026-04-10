# Changelog

## [Unreleased]

### Performance
- **Right suffix sampling: 6x faster R_suf precompute at chi=32** — Replaced O(chi^6) explicit `kron` + matrix multiplication in `precompute_double_layer_and_right_suffix` with Kronecker-factored O(chi^5) contraction. Uses the same algebraic identity already applied in the batched code path. Overall RSS speedup: ~2-3x at chi=32. No accuracy change (R_suf matrices are numerically identical, max error ~1e-16).

### Fixed
- **QAOA example in README** — Added ZZ entangling gates (CNOT-RZ-CNOT) and full MaxCut problem setup. Previous example had no entanglement.
