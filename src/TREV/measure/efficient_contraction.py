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


def _kron_contract_right_4d(Prod, A0, A1, sign=1):
    """Contract Prod @ E using Kronecker-factored O(chi^5) matmuls.

    E = conj(A0)⊗A0 + sign*conj(A1)⊗A1

    Prod: (chi, chi, chi, chi)  — 4D transfer matrix (non-batched)
    A0, A1: (chi, chi)          — site matrices
    """
    temp0 = torch.matmul(A0.conj().mT, Prod)   # (chi,chi,chi,chi) — contracts bra
    r0 = torch.matmul(temp0, A0)                 # (chi,chi,chi,chi) — contracts ket
    temp1 = torch.matmul(A1.conj().mT, Prod)
    r1 = torch.matmul(temp1, A1)
    return r0 + sign * r1


@torch.no_grad()
def expectation_value_batch(
    tensors: torch.Tensor,
    hamiltonian,
    device: str | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """
    <psi|H|psi> via identity-chain factored Kronecker O(chi^5) contraction.

    Precomputes all-identity left-prefix and right-suffix products, then
    contracts only at the sparse Z-sites of each Hamiltonian term.

    tensors: (N, chi, chi, 2)
    Returns a real scalar tensor.
    """
    device = device or tensors.device
    paulis = hamiltonian.get_bool_pauli_tensor().to(device)   # (T, N)
    if paulis.dim() != 2:
        raise ValueError("Expected paulis shape (T, N)")
    Tc, N = paulis.shape

    coeffs = torch.as_tensor(
        [c.item() if hasattr(c, "item") else c for c in hamiltonian.coefficients],
        dtype=torch.cfloat, device=device,
    )

    chi = tensors.shape[1]

    # Cache per-site A0, A1 slices
    sites = []
    for i in range(N):
        A = tensors[i].to(device=device, dtype=torch.cfloat)  # (chi, chi, 2)
        sites.append((A[:, :, 0].contiguous(), A[:, :, 1].contiguous()))

    eye4 = torch.eye(chi * chi, dtype=torch.cfloat, device=device).reshape(chi, chi, chi, chi)

    # Precompute left prefix under all-identity: L_pre[i] = E_I(0) @ ... @ E_I(i-1)
    L_pre = [None] * (N + 1)
    acc = eye4
    for i in range(N):
        L_pre[i] = acc
        A0, A1 = sites[i]
        acc = _kron_contract_right_4d(acc, A0, A1)
    L_pre[N] = acc

    # Precompute transposed right suffix under all-identity
    # R_suf_T[i] = (E_I(i) @ ... @ E_I(N-1))^T — built from transposed cores
    # Trace formula: Tr(run @ R_suf[i]) = (run * R_suf_T[i]).sum()
    R_suf_T = [None] * (N + 1)
    R_suf_T[N] = eye4
    acc = eye4
    for i in range(N - 1, -1, -1):
        A0, A1 = sites[i]
        acc = _kron_contract_right_4d(acc, A0.mT, A1.mT)
        R_suf_T[i] = acc

    # Per-term contraction: only at Z-sites
    total = torch.zeros((), dtype=torch.cfloat, device=device)

    for t in range(Tc):
        z_sites = torch.where(paulis[t])[0].tolist()

        if len(z_sites) == 0:
            total += coeffs[t] * (L_pre[N] * R_suf_T[N]).sum()
            continue

        s_first = z_sites[0]
        s_last = z_sites[-1]

        run = L_pre[s_first].clone()

        for i in range(s_first, s_last + 1):
            A0, A1 = sites[i]
            if paulis[t, i]:
                run = _kron_contract_right_4d(run, A0, A1, sign=-1)
            else:
                run = _kron_contract_right_4d(run, A0, A1)

        total += coeffs[t] * (run * R_suf_T[s_last + 1]).sum()

    return total.real