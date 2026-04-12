"""
Differentiable truncated SVD for complex tensor ring circuits.

Uses:
- TensorKit F/G split with Lorentzian broadening for degenerate SVs
- Wan-Zhang complex phase correction (arXiv:1909.02659)
- Float64 upcast in backward for numerical stability

Per-SVD accuracy: cos ≈ 0.9999 for gauge-invariant losses.
Deep circuit limitation: errors compound over many (50+) chained SVDs.
"""
import torch


def _safe_inv(x, eps=1e-12):
    """Lorentzian broadened inverse: x/(x^2 + eps) → 1/x for |x|>>eps, 0 for x→0."""
    return x / (x * x + eps)


class TruncatedSVD(torch.autograd.Function):
    """
    Forward: A (m,n) → Uk (m,k), Sk (k,), Vhk (k,n)
    Backward: F/G split + Wan-Zhang complex correction.
    """

    @staticmethod
    def forward(ctx, A, k):
        U_full, S_full, Vh_full = torch.linalg.svd(A, full_matrices=False)
        ctx.save_for_backward(U_full, S_full, Vh_full)
        ctx.k = k
        ctx.shape = A.shape
        return (U_full[:, :k].contiguous(),
                S_full[:k].contiguous(),
                Vh_full[:k, :].contiguous())

    @staticmethod
    def backward(ctx, dU, dS, dVh):
        U_full, S_full, Vh_full = ctx.saved_tensors
        m, n = ctx.shape
        k = ctx.k
        r = S_full.shape[0]
        is_complex = U_full.is_complex()
        dtype = U_full.dtype
        device = U_full.device

        # Upcast for backward stability
        orig_dtype = dtype
        if dtype in (torch.cfloat, torch.float32):
            hi = torch.complex128 if is_complex else torch.float64
            U_full = U_full.to(hi)
            S_full = S_full.to(torch.float64)
            Vh_full = Vh_full.to(hi)
            dU = dU.to(hi)
            dS = dS.to(torch.float64)
            dVh = dVh.to(hi)
            dtype = hi

        eps = 1e-12

        # Pad incoming gradients to full SVD size
        dU_full = torch.zeros(m, r, dtype=dtype, device=device)
        dU_full[:, :k] = dU
        dS_full = torch.zeros(r, dtype=S_full.dtype, device=device)
        dS_full[:k] = dS
        dVh_full = torch.zeros(r, n, dtype=dtype, device=device)
        dVh_full[:k, :] = dVh

        U = U_full[:, :r]
        S = S_full[:r]
        Vh = Vh_full[:r, :]
        V = Vh.mH

        # F/G matrices (TensorKit convention)
        S_col = S.unsqueeze(-2)  # (1, r) = Sj
        S_row = S.unsqueeze(-1)  # (r, 1) = Si
        inv_diff = _safe_inv(S_col - S_row, eps)  # 1/(Sj - Si)
        inv_sum = _safe_inv(S_col + S_row, eps)   # 1/(Sj + Si)
        eye_r = torch.eye(r, dtype=inv_diff.dtype, device=device)
        inv_diff = inv_diff * (1 - eye_r)
        inv_sum = inv_sum * (1 - eye_r)

        Uh = U.mH
        S_inv = _safe_inv(S, eps).to(dtype)

        # Anti-Hermitian parts
        UhdU = Uh @ dU_full
        VhdV = V.mH @ dVh_full.mH
        aUdU = (UhdU - UhdU.mH) / 2
        aVdV = (VhdV - VhdV.mH) / 2

        # F/G split formula
        UdAV = ((aUdU + aVdV) * inv_diff.to(dtype)
                + (aUdU - aVdV) * inv_sum.to(dtype))
        UdAV = UdAV + torch.diag_embed(dS_full.to(dtype))

        dA = U @ UdAV @ Vh

        # Complement projection terms
        if m > r:
            dU_perp = dU_full - U @ (Uh @ dU_full)
            dA = dA + dU_perp @ (S_inv.unsqueeze(-1) * Vh)
        if n > r:
            dVh_perp = dVh_full - (dVh_full @ V) @ Vh
            dA = dA + (U * S_inv.unsqueeze(-2)) @ dVh_perp

        # Wan-Zhang complex phase correction (arXiv:1909.02659)
        if is_complex:
            L_diag = UhdU.diagonal()
            L_anti = L_diag - L_diag.conj()  # = 2i * Im(L_diag)
            dA = dA + U @ torch.diag_embed(0.5 * L_anti * S_inv) @ Vh

        return dA.to(orig_dtype), None


def diff_svd(A, k):
    """
    Differentiable truncated SVD.

    Returns Uk (m,k), Sk (k,), Vhk (k,n).
    Uses F/G split (Lorentzian) + Wan-Zhang complex correction.
    """
    return TruncatedSVD.apply(A, k)
