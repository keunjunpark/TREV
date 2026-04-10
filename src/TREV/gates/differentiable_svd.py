"""
Custom differentiable SVD for complex matrices.

Strategy: use SVD in forward, but compute backward through the
full (untruncated) reconstruction A = U @ diag(S) @ Vh.
The truncation is treated as a projection with straight-through gradient.

This avoids the ill-conditioned F matrix for degenerate singular values.
"""
import torch


class DiffTruncatedSVD(torch.autograd.Function):
    """
    Forward: A → SVD → truncate to rank k → L = U_k @ diag(S_k), R = Vh_k
    Backward: dL/dA = dL/dL_trunc @ Vh_k + U_k @ diag(S_k) @ dL/dR

    This is the "straight-through" gradient: we backpropagate as if
    A_trunc = L @ R were a linear function of A, ignoring the nonlinear
    SVD truncation. For VQE optimization, this gives a useful descent
    direction even if not the exact gradient.
    """

    @staticmethod
    def forward(ctx, A, k):
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        Uk = U[:, :k].contiguous()
        Sk = S[:k].contiguous()
        Vhk = Vh[:k, :].contiguous()

        L = Uk * Sk.unsqueeze(0)  # (m, k)
        R = Vhk                    # (k, n)

        ctx.save_for_backward(Uk, Sk, Vhk)
        ctx.k = k
        return L, R

    @staticmethod
    def backward(ctx, dL, dR):
        Uk, Sk, Vhk = ctx.saved_tensors
        # Straight-through: dA = dL @ R + L @ dR
        # where L = Uk @ diag(Sk), R = Vhk
        dA = dL @ Vhk + (Uk * Sk.unsqueeze(0)) @ dR
        return dA, None


def truncated_svd(A, k):
    """
    Differentiable truncated SVD.

    Returns:
        L: (m, k) — U[:,:k] @ diag(S[:k])
        R: (k, n) — Vh[:k, :]

    Backward uses straight-through gradient (ignores truncation nonlinearity).
    """
    return DiffTruncatedSVD.apply(A, k)
