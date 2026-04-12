# Autograd Gradient: Solutions for Deep Circuit SVD Backward

## Problem

The `diff_svd` backward errors compound through many chained SVDs in deep
transpiled circuits (56+ 2-qubit gates in TSP QAOA). Forward tensors match
exactly, but the backward gradient diverges: `cos(autograd, param_shift) ≈ 0.65`
instead of 1.0.

## Research Summary

### Most Actionable (low effort, high impact)

#### 1. Newton-Schulz Polar Decomposition
**Replace SVD with polar decomposition via Newton-Schulz iteration.**

Instead of `A = U S Vh` (SVD), compute `A = W P` (polar: W orthogonal, P positive semidefinite)
via iteration: `X_{k+1} = X_k (3I - X_k^T X_k) / 2`. Converges in 5-15 iterations.
Each iteration is just matrix multiplies — PyTorch autograd handles the backward natively.

- **No custom backward needed** — standard autograd through matrix multiplies
- **No 1/(s_i - s_j) singularities** — the iteration is smooth everywhere
- **~10 lines of code** to replace diff_svd
- **Truncation**: after computing W, truncate to rank-k via QR of W
- Ref: [Fast Differentiable Matrix Sqrt](https://arxiv.org/abs/2201.08663)

```python
def polar_decomp(A, n_iter=15):
    """A = W @ P where W is orthogonal."""
    X = A / A.norm()
    for _ in range(n_iter):
        X = 0.5 * X @ (3 * torch.eye(X.shape[1], dtype=X.dtype, device=X.device) - X.mH @ X)
    return X  # W ≈ X, P = W.mH @ A
```

#### 2. SVD-Padé / SVD-Taylor Backward (drop-in)
**Replace the F-matrix formula with a Padé or Taylor approximation.**

The standard backward uses `F_ij = 1/(s_j² - s_i²)` which diverges for
near-degenerate SVs. The Taylor variant expands the backward around the
current point, avoiding the division entirely. Padé approximants are even
more stable.

- **Drop-in replacement** for `TruncatedSVD.backward()` in `differentiable_svd.py`
- **Same forward**, only backward changes
- Code available: [DifferentiableSVD](https://github.com/KingJamesSong/DifferentiableSVD)
- Ref: Wang et al., "Robust Differentiable SVD" (arXiv:2104.03821, 2021)
- Ref: Song & Sebe, ICCV 2021 — comparison of 7 methods

#### 3. Francuz Corrected Truncation Gradient
**Fix the complement projection terms in the SVD backward.**

The standard truncated SVD backward (which TREV uses) implicitly sets truncated
singular values to zero. The corrected formula by Francuz et al. adds terms
involving the truncated subspace that are currently missing.

Current TREV code:
```python
dU_perp = dU_full - U @ (Uh @ dU_full)
dA += dU_perp @ (S_inv * Vh)
```

Corrected:
```python
dU_perp = dU_full - U @ (Uh @ dU_full)
dA += dU_perp @ (S_inv * Vh)
# + correction from truncated subspace
A_rest = A - Uk @ diag(Sk) @ Vhk  # truncation residual
dA += ... (terms involving A_rest)
```

- Reduces gradient error from ~1e-3 to ~1e-6 per SVD
- Ref: Francuz, Schuch, Vanhecke, Phys. Rev. Research 7, 013237 (2025)
  [arXiv:2311.11894](https://arxiv.org/abs/2311.11894)

#### 4. QR-Based Split
**Replace SVD with QR decomposition for the gate split.**

QR has no eigenvalue degeneracy issues and PyTorch supports complex QR backward.
The earlier attempt failed due to `torch.sgn` in-place modification. Fix: use
`torch.where` for sign fixing.

```python
Q, R = torch.linalg.qr(mps)
signs = torch.where(R.diagonal().real > 0, torch.ones_like(R.diagonal()),
                    -torch.ones_like(R.diagonal()))
Q = Q * signs.unsqueeze(0).conj()
R = signs.unsqueeze(1) * R
```

- **No 1/(s_i - s_j) singularities**
- **Loss of optimal truncation** — QR gives the "leftmost" rank-k approximation,
  not the minimum Frobenius error. But for circuits where chi is large enough
  to capture the state, this doesn't matter.
- Ref: Roberts & Roberts, arXiv:2009.10071

### Medium Effort

#### 5. Gradient Checkpointing
**Break the 56-SVD chain into segments with checkpoints.**

Use `torch.utils.checkpoint.checkpoint()` to segment the gate applications.
Each segment (5-10 gates) gets its own forward/backward. The backward error
only accumulates within each segment, not across the full chain.

```python
def apply_gate_segment(cores, gates, theta):
    for gate in gates:
        # apply gate
    return cores

# In _build_tensor_diff:
segments = chunk(circuit.gates, size=8)
for seg in segments:
    cores = torch.utils.checkpoint.checkpoint(apply_gate_segment, cores, seg, theta)
```

- Trades memory for compute (re-runs forward during backward)
- Limits error accumulation to segment length
- Easy to implement with PyTorch's checkpointing API

#### 6. Implicit Differentiation
**Don't backprop through SVDs — use implicit function theorem instead.**

At the SVD fixed point `A = U S Vh`, the gradient satisfies a linear system
that can be solved directly without backpropagating through the SVD computation.

- Ref: Liao et al., Phys. Rev. X 9, 031041 (2019) — "Differentiable Programming TN"
- Only works if the circuit has a fixed-point structure (e.g., repeated layers)

#### 7. Hybrid: Autograd Contraction + Param-Shift Circuit
**Use autograd only for the contraction, not the circuit building.**

The contraction backward works perfectly. The circuit building backward is broken.
So: build the tensor with `build_tensor` (no autograd), compute `dEV/d(tensor)`
via autograd contraction, then convert the tensor gradient to a theta gradient
via finite differences or the chain rule applied gate-by-gate.

### High Effort

#### 8. Riemannian Optimization on Isometric Manifold
**Optimize directly on the manifold of isometric tensors (W†W = I).**

No SVD at all. Each tensor core is constrained to be isometric, and optimization
uses Riemannian gradients with retraction. Proven to be barren-plateau-free.

- Ref: Hauru, Van Damme, Haegeman, SciPost Phys. 10, 040 (2021)
- Ref: [arXiv:2007.03638](https://arxiv.org/abs/2007.03638)
- Major architecture change — TREV would need isometric parametrization

#### 9. Dense TN Geometries (Tree/Star/Antenna)
**Use a denser tensor network structure instead of MPS/ring.**

Denser geometries need fewer truncation steps, reducing the SVD chain length.

- Ref: Masot-Llima & Garcia-Saez, arXiv:2412.17497 (2024)
- Architecture change for TREV

## Recommended Implementation Order

1. **Newton-Schulz polar** (10 lines, no custom backward, test immediately)
2. **QR split** (fix the in-place `sgn` issue, 15 lines)
3. **SVD-Padé backward** (drop-in for `TruncatedSVD.backward()`)
4. **Gradient checkpointing** (limit chain length to ~8 SVDs per segment)
5. **Francuz correction** (add truncation residual terms)

## References

- Francuz et al. (2025): [arXiv:2311.11894](https://arxiv.org/abs/2311.11894)
- Wang et al. (2021): [arXiv:2104.03821](https://arxiv.org/abs/2104.03821)
- Song & Sebe (2021): [DifferentiableSVD](https://github.com/KingJamesSong/DifferentiableSVD)
- Hauru et al. (2021): [arXiv:2007.03638](https://arxiv.org/abs/2007.03638)
- Liao et al. (2019): [arXiv:1903.09650](https://arxiv.org/abs/1903.09650)
- Roberts & Roberts (2020): [arXiv:2009.10071](https://arxiv.org/abs/2009.10071)
- Peddireddy et al. (2023): [arXiv:2307.03884](https://arxiv.org/abs/2307.03884)
- TensorCircuit-NG (2026): [arXiv:2602.14167](https://arxiv.org/abs/2602.14167)
- Masot-Llima & Garcia-Saez (2024): [arXiv:2412.17497](https://arxiv.org/abs/2412.17497)
