"""
Differentiable truncated SVD for complex tensor ring circuits.

Implements the corrected backward from:
- peps-torch (Hasik) — F/G split regularization + complex correction
- Francuz et al. (2311.11894) — truncation correction terms

Returns (Uk, Sk, Vhk) as separate tensors so both qu0=Uk*Sk and qu1=Vhk
get gradients through the single backward call.
"""
import torch


def _safe_inv(x, eps=1e-12):
    """Lorentzian broadened inverse: x/(x^2 + eps) → 1/x for |x|>>eps, 0 for x→0."""
    return x / (x * x + eps)


class TruncatedSVD(torch.autograd.Function):
    """
    Forward: A (m,n) → Uk (m,k), Sk (k,), Vhk (k,n)
    Backward: corrected formula with F/G split regularization.

    Both Uk and Vhk are autograd-connected to A. The caller forms
    qu0 = Uk * Sk and qu1 = Vhk; autograd naturally propagates
    gradients from both back through this single backward.
    """

    @staticmethod
    def forward(ctx, A, k):
        U_full, S_full, Vh_full = torch.linalg.svd(A, full_matrices=False)
        Uk = U_full[:, :k].contiguous()
        Sk = S_full[:k].contiguous()
        Vhk = Vh_full[:k, :].contiguous()
        ctx.save_for_backward(U_full, S_full, Vh_full)
        ctx.k = k
        ctx.shape = A.shape
        return Uk, Sk, Vhk

    @staticmethod
    def backward(ctx, dU, dS, dVh):
        U_full, S_full, Vh_full = ctx.saved_tensors
        m, n = ctx.shape
        k = ctx.k
        r = S_full.shape[0]  # min(m, n)
        is_complex = U_full.is_complex()
        dtype = U_full.dtype
        device = U_full.device

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

        eps = 1e-6 if S.dtype == torch.float64 else 1e-4

        # ── TensorKit.jl convention: inv_diff = 1/(Sj - Si), inv_sum = 1/(Sj + Si) ──
        S_col = S.unsqueeze(-2)  # (1, r) = Sj
        S_row = S.unsqueeze(-1)  # (r, 1) = Si
        inv_diff = _safe_inv(S_col - S_row, eps)  # 1/(Sj - Si)
        inv_sum = _safe_inv(S_col + S_row, eps)   # 1/(Sj + Si)
        eye_r = torch.eye(r, dtype=inv_diff.dtype, device=device)
        inv_diff = inv_diff * (1 - eye_r)
        inv_sum = inv_sum * (1 - eye_r)

        Uh = U.mH
        V = Vh.mH
        S_inv = _safe_inv(S, eps).to(dtype)

        # ── Anti-Hermitian parts ──
        UhdU = Uh @ dU_full
        VhdV = V.mH @ dVh_full.mH  # = Vh @ dV where dV = dVh^H
        aUdU = (UhdU - UhdU.mH) / 2
        aVdV = (VhdV - VhdV.mH) / 2

        # ── UdAV matrix (combined F-term from TensorKit.jl) ──
        UdAV = ((aUdU + aVdV) * inv_diff + (aUdU - aVdV) * inv_sum).to(dtype)
        UdAV = UdAV + torch.diag_embed(dS_full.to(dtype))

        dA = U @ UdAV @ Vh

        # ── Complement projection terms ──
        if m > r:
            dU_perp = dU_full - U @ (Uh @ dU_full)
            dA = dA + dU_perp @ (S_inv.unsqueeze(-1) * Vh)
        if n > r:
            dVh_perp = dVh_full - (dVh_full @ V) @ Vh
            dA = dA + (U * S_inv.unsqueeze(-2)) @ dVh_perp

        # ── Complex correction: imaginary diagonal of U^H dU ──
        if is_complex:
            L_diag = (Uh @ dU_full).diagonal()
            L_corr = torch.zeros_like(L_diag)
            L_corr.imag = L_diag.imag * S_inv.real
            dA = dA + (U * L_corr.unsqueeze(-2)) @ Vh
        return dA, None


def diff_svd(A, k):
    """
    Differentiable truncated SVD.

    Returns Uk (m,k), Sk (k,), Vhk (k,n).
    Both Uk and Vhk have gradients connected to A through the backward.

    Use: qu0 = Uk * Sk.unsqueeze(0), qu1 = Vhk
    """
    return TruncatedSVD.apply(A, k)
