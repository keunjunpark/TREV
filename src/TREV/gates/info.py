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
    theta = torch.atleast_1d(theta)
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    rx = torch.stack([
        torch.stack([cos, -1j * sin], dim=-1),
        torch.stack([-1j * sin, cos], dim=-1)
    ], dim=-2).to(device).type(torch.cfloat)
    return rx if len(theta) > 1 else rx[0]

def RY(theta, device:str=None):
    theta = torch.atleast_1d(theta)
    cos = torch.cos(theta / 2)
    sin = torch.sin(theta / 2)
    ry = torch.stack([
        torch.stack([cos, -sin], dim=-1),
        torch.stack([sin,  cos], dim=-1)
    ], dim=-2).type(torch.cfloat).to(device)
    return ry if len(theta) > 1 else ry[0]

def RZ(theta, device:str=None):
    theta = torch.atleast_1d(theta)
    exp_m = torch.exp(-1j * theta / 2)
    exp_p = torch.exp(1j * theta / 2)
    rz = torch.stack([
        torch.stack([exp_m, torch.zeros_like(theta)], dim=-1),
        torch.stack([torch.zeros_like(theta), exp_p], dim=-1)
    ], dim=-2).to(device).type(torch.cfloat)
    return rz if len(theta) > 1 else rz[0]

def SWAP(batch_size=None, device:str=None):
    swap = torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=torch.cfloat).to(device)
    if batch_size is None:
        return swap
    else:
        return swap.unsqueeze(0).repeat(batch_size, 1, 1)

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


# ═══════════════════════════════════════════════════════════════
# Real-valued gate representations for compilable autograd.
#
# Convention: a complex (m, n) matrix G = G_r + i*G_i is stored
# as a real (2, m, n) tensor where [0] = real, [1] = imag.
#
# Applying G to a state in real form:
#   new_real = G_r @ state_real - G_i @ state_imag
#   new_imag = G_r @ state_imag + G_i @ state_real
# ═══════════════════════════════════════════════════════════════

def _to_ri(c):
    """Convert complex tensor to (2, ...) real representation."""
    return torch.stack([c.real, c.imag], dim=0)


def H_real(device=None):
    """Hadamard gate as (2, 2, 2) real tensor."""
    s = 1.0 / math.sqrt(2)
    r = torch.tensor([[s, s], [s, -s]], dtype=torch.float32, device=device)
    i = torch.zeros(2, 2, dtype=torch.float32, device=device)
    return torch.stack([r, i], dim=0)


def X_real(device=None):
    r = torch.tensor([[0., 1.], [1., 0.]], device=device)
    return torch.stack([r, torch.zeros_like(r)], dim=0)


def Y_real(device=None):
    """Y = [[0, -i], [i, 0]] → real=[[0,0],[0,0]], imag=[[0,-1],[1,0]]."""
    r = torch.zeros(2, 2, dtype=torch.float32, device=device)
    i = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float32, device=device)
    return torch.stack([r, i], dim=0)


def Z_real(device=None):
    r = torch.tensor([[1., 0.], [0., -1.]], device=device)
    return torch.stack([r, torch.zeros_like(r)], dim=0)


def I_real(device=None):
    r = torch.eye(2, dtype=torch.float32, device=device)
    return torch.stack([r, torch.zeros_like(r)], dim=0)


def RX_real(theta, device=None):
    """RX(θ) = [[cos, -i*sin], [-i*sin, cos]].
    Real: [[cos,0],[0,cos]], Imag: [[0,-sin],[-sin,0]]."""
    theta = torch.atleast_1d(theta).to(torch.float32)
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    z = torch.zeros_like(theta)
    r = torch.stack([torch.stack([c, z], -1), torch.stack([z, c], -1)], -2)
    i = torch.stack([torch.stack([z, -s], -1), torch.stack([-s, z], -1)], -2)
    out = torch.stack([r, i], dim=0)
    return out if len(theta) > 1 else out[:, 0]


def RY_real(theta, device=None):
    """RY(θ) = [[cos, -sin], [sin, cos]]. Purely real."""
    theta = torch.atleast_1d(theta).to(torch.float32)
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    r = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    i = torch.zeros_like(r)
    out = torch.stack([r, i], dim=0)
    return out if len(theta) > 1 else out[:, 0]


def RZ_real(theta, device=None):
    """RZ(θ) = [[e^{-iθ/2}, 0], [0, e^{iθ/2}]].
    Real: [[cos,-sin],[sin,cos]] block-diag form... no.
    Actually: diag(cos∓isin). Real=[[cos,0],[0,cos]], Imag=[[-sin,0],[0,sin]]."""
    theta = torch.atleast_1d(theta).to(torch.float32)
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    z = torch.zeros_like(theta)
    r = torch.stack([torch.stack([c, z], -1), torch.stack([z, c], -1)], -2)
    i = torch.stack([torch.stack([-s, z], -1), torch.stack([z, s], -1)], -2)
    out = torch.stack([r, i], dim=0)
    return out if len(theta) > 1 else out[:, 0]


def CNOT_real(device=None):
    """CNOT as (2, 4, 4) real tensor. CNOT is real-valued."""
    r = torch.tensor([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [0., 0., 1., 0.],
    ], device=device)
    return torch.stack([r, torch.zeros_like(r)], dim=0)


def SWAP_real(device=None):
    """SWAP as (2, 4, 4) real tensor. SWAP is real-valued."""
    r = torch.tensor([
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
    ], device=device)
    return torch.stack([r, torch.zeros_like(r)], dim=0)
