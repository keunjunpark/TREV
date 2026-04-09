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

def SWAP(device:str=None):
    return torch.tensor([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=torch.cfloat).to(device)

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
