import torch
import math

def I(batch_size=None, device:str=None):
    if batch_size is None:
        return torch.eye(2, dtype=torch.cfloat).to(device)
    else:
        return torch.eye(2, dtype=torch.cfloat).expand(batch_size,2,2).to(device)

def X(batch_size=None, device:str=None):
    if batch_size is None:
        return torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat).to(device)
    else:
        return torch.tensor([[0, 1], [1, 0]], dtype=torch.cfloat).expand(batch_size,2,2).to(device)

def Y(batch_size=None, device:str=None):
    if batch_size is None:
        return torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat).to(device)
    else:
        return torch.tensor([[0, -1j], [1j, 0]], dtype=torch.cfloat).expand(batch_size,2,2).to(device)

def Z(batch_size=None, device:str=None):
    if batch_size is None:
        return torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat).to(device)
    else:
        return torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat).expand(batch_size,2,2).to(device)


def H(batch_size=None, device:str=None):
    if batch_size is None:
        return (1 / math.sqrt(2)) * torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat).to(device)
    else:
        return (1 / math.sqrt(2)) * torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat).expand(batch_size,2,2).to(device)

def RX(theta, device:str=None):  # theta can be scalar or batch
    is_scalar = not isinstance(theta, torch.Tensor) or theta.dim() == 0
    theta = torch.atleast_1d(theta)
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    rx = torch.stack([
        torch.stack([cos, -1j * sin], dim=-1),
        torch.stack([-1j * sin, cos], dim=-1)
    ], dim=-2).to(device=device)
    return rx[0] if is_scalar else rx

def RY(theta, device:str=None):
    is_scalar = not isinstance(theta, torch.Tensor) or theta.dim() == 0
    theta = torch.atleast_1d(theta)
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    ry = torch.stack([
        torch.stack([cos, -sin], dim=-1),
        torch.stack([sin,  cos], dim=-1)
    ], dim=-2).to(device=device)
    return ry[0] if is_scalar else ry

def RZ(theta, device:str=None):
    is_scalar = not isinstance(theta, torch.Tensor) or theta.dim() == 0
    theta = torch.atleast_1d(theta)
    exp_m = torch.exp(-1j * theta / 2)
    exp_p = torch.exp(1j * theta / 2)
    rz = torch.stack([
        torch.stack([exp_m, torch.zeros_like(theta)], dim=-1),
        torch.stack([torch.zeros_like(theta), exp_p], dim=-1)
    ], dim=-2).to(device=device)
    return rz[0] if is_scalar else rz

def SWAP(batch_size=None, device:str=None):
    return torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=torch.cfloat).to(device)

def ZZ(theta, device:str=None):
    """ZZ(θ) = diag(e^{-iθ/2}, e^{iθ/2}, e^{iθ/2}, e^{-iθ/2}).

    Equivalent to CX · RZ(θ) · CX, but requires only one SVD instead of two.
    """
    is_scalar = not isinstance(theta, torch.Tensor) or theta.dim() == 0
    theta = torch.atleast_1d(theta)
    a = torch.exp(-1j * theta / 2)
    b = torch.exp( 1j * theta / 2)
    z = torch.zeros_like(theta)
    # Build (B, 4, 4) diagonal matrix
    row0 = torch.stack([a, z, z, z], dim=-1)
    row1 = torch.stack([z, b, z, z], dim=-1)
    row2 = torch.stack([z, z, b, z], dim=-1)
    row3 = torch.stack([z, z, z, a], dim=-1)
    mat = torch.stack([row0, row1, row2, row3], dim=-2).to(device=device)
    return mat[0] if is_scalar else mat


def ZZ_SWAP(theta, device:str=None):
    """ZZ_SWAP(θ) = SWAP · ZZ(θ).

    Fused gate that applies a ZZ interaction and then swaps the two qubits.
    Requires only one SVD instead of two (one for ZZ + one for SWAP).

    Matrix (computational basis |00⟩, |01⟩, |10⟩, |11⟩):
        [[e^{-iθ/2},  0,          0,          0         ],
         [0,          0,          e^{iθ/2},   0         ],
         [0,          e^{iθ/2},   0,          0         ],
         [0,          0,          0,          e^{-iθ/2} ]]
    """
    is_scalar = not isinstance(theta, torch.Tensor) or theta.dim() == 0
    theta = torch.atleast_1d(theta)
    a = torch.exp(-1j * theta / 2)
    b = torch.exp( 1j * theta / 2)
    z = torch.zeros_like(theta)
    # SWAP @ ZZ: rows of ZZ permuted by SWAP
    # SWAP exchanges |01⟩ <-> |10⟩, so row1<->row2
    row0 = torch.stack([a, z, z, z], dim=-1)  # |00⟩ -> e^{-iθ/2}|00⟩
    row1 = torch.stack([z, z, b, z], dim=-1)  # |01⟩ -> e^{iθ/2}|10⟩
    row2 = torch.stack([z, b, z, z], dim=-1)  # |10⟩ -> e^{iθ/2}|01⟩
    row3 = torch.stack([z, z, z, a], dim=-1)  # |11⟩ -> e^{-iθ/2}|11⟩
    mat = torch.stack([row0, row1, row2, row3], dim=-2).to(device=device)
    return mat[0] if is_scalar else mat


def U3(params, device:str=None):
    """U3(θ,φ,λ) = Rz(φ)·Ry(θ)·Rz(λ).

    Args:
        params: shape (3,) for scalar or (batch, 3) for batch.
                params[..., 0] = theta, params[..., 1] = phi, params[..., 2] = lambda.
    """
    is_scalar = params.dim() == 1
    if is_scalar:
        params = params.unsqueeze(0)  # (1, 3)
    theta = params[:, 0]
    phi   = params[:, 1]
    lam   = params[:, 2]
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    u00 = torch.exp(-1j * (phi + lam) / 2) * cos
    u01 = -torch.exp(-1j * (phi - lam) / 2) * sin
    u10 = torch.exp(1j * (phi - lam) / 2) * sin
    u11 = torch.exp(1j * (phi + lam) / 2) * cos
    mat = torch.stack([
        torch.stack([u00, u01], dim=-1),
        torch.stack([u10, u11], dim=-1)
    ], dim=-2).to(device=device)
    return mat[0] if is_scalar else mat


def CNOT(batch_size=None, device:str=None):
    cnot=  torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.cfloat).to(device)
    if batch_size is None:
        return cnot
    else:
        return cnot.unsqueeze(0).repeat(batch_size, 1, 1)
