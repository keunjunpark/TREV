"""
Differentiable truncated SVD for complex tensor ring circuits.

Implements the Wan-Zhang complex SVD backward formula (arXiv:1909.02659)
which correctly handles the phase degree of freedom via an extra
anti-Hermitian diagonal correction term. This produces correct gradients
for gauge-invariant losses (like tensor ring contractions) without
requiring gauge fixing or real-form conversion.

The F-matrix uses Lorentzian broadening for degenerate singular values.
"""
import torch


def _safe_inv(x, eps=1e-12):
    """Lorentzian broadened inverse: x/(x^2 + eps) → 1/x for |x|>>eps, 0 for x→0."""
    return x / (x * x + eps)


class TruncatedSVD(torch.autograd.Function):
    """
    Forward: A (m,n) → Uk (m,k), Sk (k,), Vhk (k,n)
    Backward: Wan-Zhang formula with correct complex phase correction.
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
        U, S, Vh = ctx.saved_tensors
        m, n = ctx.shape
        k = ctx.k
        r = S.shape[0]
        is_complex = U.is_complex()
        dtype = U.dtype
        device = U.device

        # Upcast for backward stability
        orig_dtype = dtype
        if dtype in (torch.cfloat, torch.float32):
            hi = torch.complex128 if is_complex else torch.float64
            U = U.to(hi); S = S.to(torch.float64); Vh = Vh.to(hi)
            dU = dU.to(hi); dS = dS.to(torch.float64); dVh = dVh.to(hi)
            dtype = hi

        eps = 1e-12
        V = Vh.mH
        Uh = U.mH
        S_inv = _safe_inv(S, eps).to(dtype)

        # Pad gradients to full SVD size
        gU = torch.zeros(m, r, dtype=dtype, device=device)
        gU[:, :k] = dU
        gS = torch.zeros(r, dtype=S.dtype, device=device)
        gS[:k] = dS
        gVh = torch.zeros(r, n, dtype=dtype, device=device)
        gVh[:k, :] = dVh
        gV = gVh.mH  # (n, r)

        # F-matrix: F_ij = 1/(s_i^2 - s_j^2) for i != j
        S2 = S * S
        E = S2.unsqueeze(-2) - S2.unsqueeze(-1)
        eye_r = torch.eye(r, dtype=E.dtype, device=device)
        F = _safe_inv(E, eps) * (1 - eye_r)  # Lorentzian for degenerate pairs

        S_dtype = S.to(dtype)
        dA = torch.zeros(m, n, dtype=dtype, device=device)

        # dS contribution
        dA = dA + U @ torch.diag_embed(gS.to(dtype)) @ Vh

        # dU contribution (skew-Hermitian part)
        UhgU = Uh @ gU
        skew_U = UhgU - UhgU.mH
        u_term = U @ (F * skew_U * S_dtype.unsqueeze(-2)) @ Vh
        dA = dA + u_term

        # Complement projection for U (when m > r)
        if m > r:
            proj_U = gU - U @ (Uh @ gU)
            dA = dA + proj_U @ (S_inv.unsqueeze(-1) * Vh)

        # dVh contribution (skew-Hermitian part)
        VhgV = Vh @ gV
        skew_V = VhgV - VhgV.mH
        v_term = U @ (S_dtype.unsqueeze(-1) * F * skew_V) @ Vh
        dA = dA + v_term

        # Complement projection for V (when n > r)
        if n > r:
            proj_V = gV - V @ (Vh @ gV)
            dA = dA + (U * S_inv.unsqueeze(-2)) @ proj_V.mH

        # Wan-Zhang complex phase correction
        if is_complex:
            L_diag = UhgU.diagonal()
            L_anti = L_diag - L_diag.conj()  # 2i * Im(diag(U^H gU))
            dA = dA + U @ torch.diag_embed(0.5 * L_anti * S_inv) @ Vh

        return dA.to(orig_dtype), None


def diff_svd(A, k):
    """
    Differentiable truncated SVD.

    Returns Uk (m,k), Sk (k,), Vhk (k,n).
    Both Uk and Vhk have gradients connected to A.

    Uses the Wan-Zhang formula (arXiv:1909.02659) for the complex
    backward, which correctly handles the phase degree of freedom
    for gauge-invariant losses.
    """
    return TruncatedSVD.apply(A, k)
