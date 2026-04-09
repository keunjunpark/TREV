from typing import Optional, List

import torch

from ..hamiltonian.hamiltonian import Hamiltonian

def expectation_value(tensor:torch.Tensor, hamiltonian: Hamiltonian, shots:float = 1e4,device:str = None):

    ret = 0 
    paulis_tensor = hamiltonian.get_bool_pauli_tensor().to(device=device)

    co = 0 
    for paulis in paulis_tensor:
        coef = hamiltonian.coefficients[co]
        co+=1

        for i in range(0,tensor.size(0)):

            if i == int(0):
                curr_ten = tensor[i].permute(0, 2, 1)
                if paulis[i] == True: # Z 
                    AO = torch.einsum('ldr,dk->lkr', curr_ten, torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat).to(device))
                else:
                    AO = curr_ten # I 
                E_raw = torch.tensordot(torch.conj(curr_ten), AO, ([1],[1]))  # <A|O|A> (l,l',r,r') 10 x 10 x 10 x 10 
                E = E_raw.permute(0,2,1,3)  # (l,l',r,r')
                ten = E

            else:
                curr_ten = tensor[i].permute(0, 2, 1)
                if paulis[i] == True:
                    AO = torch.einsum('ldr,dk->lkr', curr_ten, torch.tensor([[1, 0],[0, -1]], dtype=torch.cfloat, device=device))
                else:
                    AO = curr_ten
                E_raw = torch.tensordot(torch.conj(curr_ten), AO, ([1],[1])) 


                E = E_raw.permute(0,2,1,3)  # (l,l',r,r')
                ten = torch.tensordot(ten, E, dims=([2,3],[0,1])) # (l1,l1',r2,r2')
        # MPS => scalar value
        # 2x2x2x2, [0,0,0,0] + [0,1,0,1] + [1,0,1,0] + [1,1,1,1] 
        ret += coef * torch.real(torch.einsum('ikik->', ten)  )
    return ret


def expectation_value_batch(
    tensors: torch.Tensor,
    hamiltonian,
    device: str | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """
    Batched <psi|H|psi> for a Tensor Ring (periodic MPS), no external helpers.

    tensors[i]: (ℓ_i, r_i, d) with d=2.
    Hamiltonian API:
      - get_bool_pauli_tensor() -> Bool tensor (T, N)  (True => apply Z)
      - coefficients -> iterable length T (real/complex)

    Returns a real scalar tensor.
    """
    device = device or tensors.device
    paulis = hamiltonian.get_bool_pauli_tensor().to(device)   # (T, N)
    if paulis.dim() != 2:
        raise ValueError("Expected paulis shape (T, N)")
    T, N = paulis.shape

    # Complex coefficients
    coeffs = torch.as_tensor(
        [c.item() if hasattr(c, "item") else c for c in hamiltonian.coefficients],
        dtype=torch.cfloat, device=device,
    )

    # Local ops
    Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=device)
    I = torch.eye(2, dtype=torch.cfloat, device=device)

    # Default: one chunk
    if chunk_size is None:
        chunk_size = T

    total = torch.zeros((), dtype=torch.cfloat, device=device)

    for start in range(0, T, chunk_size):
        stop = min(start + chunk_size, T)
        mask = paulis[start:stop]            # (B, N)
        coefs = coeffs[start:stop]           # (B,)
        B = mask.size(0)

        # Build the first site's batched transfer E0(b,ℓ,ℓ',r,r')
        A = tensors[0].to(device)            # (ℓ0, r0, d)
        AO_I = torch.einsum('lrd,dk->lrk', A, I)  # (ℓ0,r0,k)
        AO_Z = torch.einsum('lrd,dk->lrk', A, Z)  # (ℓ0,r0,k)
        E_I = torch.tensordot(A.conj(), AO_I, dims=([2],[2])).permute(0,2,1,3)  # (ℓ0,ℓ0',r0,r0')
        E_Z = torch.tensordot(A.conj(), AO_Z, dims=([2],[2])).permute(0,2,1,3)  # (ℓ0,ℓ0',r0,r0')
        m0 = mask[:, 0].view(B, 1, 1, 1, 1)
        ten = torch.where(m0, E_Z.unsqueeze(0), E_I.unsqueeze(0))               # (B,ℓ0,ℓ0',r0,r0')

        # Chain remaining sites
        for i in range(1, N):
            A = tensors[i].to(device)              # (ℓ, r, d)
            AO_I = torch.einsum('lrd,dk->lrk', A, I)
            AO_Z = torch.einsum('lrd,dk->lrk', A, Z)
            Ei_I = torch.tensordot(A.conj(), AO_I, dims=([2],[2])).permute(0,2,1,3)  # (ℓ,ℓ',r,r')
            Ei_Z = torch.tensordot(A.conj(), AO_Z, dims=([2],[2])).permute(0,2,1,3)
            mi = mask[:, i].view(B, 1, 1, 1, 1)
            Ei = torch.where(mi, Ei_Z.unsqueeze(0), Ei_I.unsqueeze(0))               # (B,ℓ,ℓ',r,r')
            # Contract: ten(b,i,j,p,q) * Ei(b,p,q,r,s) -> (b,i,j,r,s)
            ten = torch.einsum('bijpq,bpqrs->bijrs', ten, Ei)

        # Close ring with double trace over i=r and j=s
        vals = torch.einsum('bijij->b', ten)   # (B,)
        total = total + torch.sum(coefs * vals)

    return total.real  # scalar