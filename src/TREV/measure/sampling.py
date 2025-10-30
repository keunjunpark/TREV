from typing import Optional, List

import torch

from ..hamiltonian.hamiltonian import Hamiltonian

def measure(tensor: torch.Tensor, shots:float|int, device:str=None):
    q0 = torch.tensor([[1], [0]], dtype=torch.cfloat).to(device)
    q1 = torch.tensor([[0], [1]], dtype=torch.cfloat).to(device)
    prob_dist = [0]*(2**tensor.size(0))
    increment = 1/shots
    for _ in range(int(shots)):
        prev: Optional[torch.Tensor] = None
        key: float = 0.0
        first_0, first_1 = 0.0, 0.0
        prev_0, prev_1 = None, None
        for i in range(tensor.size(0)):
            if i == int(0):
                # curr_ten =  χ1 × χ2 * 2
                curr_ten = tensor[i]

                qubit_0 = torch.tensordot(curr_ten, q0, ([2], [0])).squeeze(-1)
                qubit_1 = torch.tensordot(curr_ten, q1, ([2], [0])).squeeze(-1)

                prob_0 = torch.tensordot(qubit_0, qubit_0.mH, ([1, 0], [0, 1])).real.item()
                prob_1 = torch.tensordot(qubit_1, qubit_1.mH, ([1, 0], [0, 1])).real.item()
                total = prob_0 + prob_1
                #total = prob_0 + prob_1 + 1e-8

            else:
                # curr_ten =  X1 * X2 * 2
                curr_ten = tensor[i]
                if prev is not None:
                    curr_ten = torch.tensordot(prev, curr_ten, ([1], [0]))
                qubit_0 = torch.tensordot(curr_ten, q0, ([2], [0])).squeeze(-1)
                qubit_1 = torch.tensordot(curr_ten, q1, ([2], [0])).squeeze(-1)

                prob_0 = torch.tensordot(qubit_0, qubit_0.mH, ([1, 0], [0, 1])).real.item()
                prob_1 = torch.tensordot(qubit_1, qubit_1.mH, ([1, 0], [0, 1])).real.item()
                total = prob_0 + prob_1
                #total = prob_0 + prob_1 + 1e-8
            rnd = torch.rand((1, 1))[0].item()
            if total == 0:
                prob_0 = 0.5
                total = 1

            if rnd > prob_0 / total:
                prev = qubit_1
                key += float(2 ** i) * 1
            else:
                prev = qubit_0
                key += float(2 ** i) * 0
        
        prob_dist[int(key)]+=increment
    return prob_dist

# def expectation_value1(tensors: torch.Tensor,
#                       hamiltonian,
#                       device: str = None,
#                       periodic: bool = True):
#     """
#     tensors[i]: site tensor with shape (ℓ, r, d)
#     """
#     Z = torch.tensor([[1,0],[0,-1]], dtype=torch.cfloat, device=device)
#     I = torch.eye(2, dtype=torch.cfloat, device=device)

#     paulis_tensor = hamiltonian.get_bool_pauli_tensor().to(device)
#     coeffs = hamiltonian.coefficients
#     ret = 0.0

#     for co, paulis in enumerate(paulis_tensor):
#         # initialize transfer chain
#         A_lrd = tensors[0]
#         O = Z if paulis[0] else I
#         AO = torch.einsum('lrd,dk->lrk', A_lrd, O)
#         E = torch.tensordot(A_lrd.conj(), AO, dims=([2],[2])).permute(0,2,1,3)
#         ten = E

#         # middle sites
#         for i in range(1, tensors.size(0)):
#             A_lrd = tensors[i]
#             O = Z if paulis[i] else I
#             AO = torch.einsum('lrd,dk->lrk', A_lrd, O)
#             E = torch.tensordot(A_lrd.conj(), AO, dims=([2],[2])).permute(0,2,1,3)
#             ten = torch.tensordot(ten, E, dims=([2,3],[0,1]))

#         # double trace: sum over (ℓ= r_last, ℓ'= r_last')
#         val = torch.einsum('ikik->', ten) 
#         ret += coeffs[co] * torch.real(val)

#     return ret


# def expectation_value(tensor:torch.Tensor, hamiltonian: Hamiltonian, shots:float = 1e4,device:str = None):

#     ret = 0 
#     paulis_tensor = hamiltonian.get_bool_pauli_tensor().to(device=device)

#     co = 0 
#     for paulis in paulis_tensor:
#         coef = hamiltonian.coefficients[co]
#         co+=1

#         for i in range(0,tensor.size(0)):

#             if i == int(0):
#                 curr_ten = tensor[i].permute(0, 2, 1)
#                 if paulis[i] == True:
#                     AO = torch.einsum('ldr,dk->lkr', curr_ten, torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat).to(device))
#                 else:
#                     AO = curr_ten
#                 E_raw = torch.tensordot(torch.conj(curr_ten), AO, ([1],[1])) 
#                 E = E_raw.permute(0,2,1,3)  # (l,l',r,r')
#                 ten = E

#             else:
#                 curr_ten = tensor[i].permute(0, 2, 1)
#                 if paulis[i] == True:
#                     AO = torch.einsum('ldr,dk->lkr', curr_ten, torch.tensor([[1, 0],[0, -1]], dtype=torch.cfloat, device=device))
#                 else:
#                     AO = curr_ten
#                 E_raw = torch.tensordot(torch.conj(curr_ten), AO, ([1],[1])) 
#                 E = E_raw.permute(0,2,1,3)  # (l,l',r,r')
#                 ten = torch.tensordot(ten, E, dims=([2,3],[0,1])) # (l1,l1',r2,r2')
#         ret += coef * torch.real(torch.einsum('ikik->', ten)  )
#     return ret


# def expectation_value_batch(
#     tensors: torch.Tensor,
#     hamiltonian,
#     device: str | None = None,
#     chunk_size: int | None = None,
# ) -> torch.Tensor:
#     """
#     Batched <psi|H|psi> for a Tensor Ring (periodic MPS), no external helpers.

#     tensors[i]: (ℓ_i, r_i, d) with d=2.
#     Hamiltonian API:
#       - get_bool_pauli_tensor() -> Bool tensor (T, N)  (True => apply Z)
#       - coefficients -> iterable length T (real/complex)

#     Returns a real scalar tensor.
#     """
#     device = device or tensors.device
#     paulis = hamiltonian.get_bool_pauli_tensor().to(device)   # (T, N)
#     if paulis.dim() != 2:
#         raise ValueError("Expected paulis shape (T, N)")
#     T, N = paulis.shape

#     # Complex coefficients
#     coeffs = torch.as_tensor(
#         [c.item() if hasattr(c, "item") else c for c in hamiltonian.coefficients],
#         dtype=torch.cfloat, device=device,
#     )

#     # Local ops
#     Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=device)
#     I = torch.eye(2, dtype=torch.cfloat, device=device)

#     # Default: one chunk
#     if chunk_size is None:
#         chunk_size = T

#     total = torch.zeros((), dtype=torch.cfloat, device=device)

#     for start in range(0, T, chunk_size):
#         stop = min(start + chunk_size, T)
#         mask = paulis[start:stop]            # (B, N)
#         coefs = coeffs[start:stop]           # (B,)
#         B = mask.size(0)

#         # Build the first site's batched transfer E0(b,ℓ,ℓ',r,r')
#         A = tensors[0].to(device)            # (ℓ0, r0, d)
#         AO_I = torch.einsum('lrd,dk->lrk', A, I)  # (ℓ0,r0,k)
#         AO_Z = torch.einsum('lrd,dk->lrk', A, Z)  # (ℓ0,r0,k)
#         E_I = torch.tensordot(A.conj(), AO_I, dims=([2],[2])).permute(0,2,1,3)  # (ℓ0,ℓ0',r0,r0')
#         E_Z = torch.tensordot(A.conj(), AO_Z, dims=([2],[2])).permute(0,2,1,3)  # (ℓ0,ℓ0',r0,r0')
#         m0 = mask[:, 0].view(B, 1, 1, 1, 1)
#         ten = torch.where(m0, E_Z.unsqueeze(0), E_I.unsqueeze(0))               # (B,ℓ0,ℓ0',r0,r0')

#         # Chain remaining sites
#         for i in range(1, N):
#             A = tensors[i].to(device)              # (ℓ, r, d)
#             AO_I = torch.einsum('lrd,dk->lrk', A, I)
#             AO_Z = torch.einsum('lrd,dk->lrk', A, Z)
#             Ei_I = torch.tensordot(A.conj(), AO_I, dims=([2],[2])).permute(0,2,1,3)  # (ℓ,ℓ',r,r')
#             Ei_Z = torch.tensordot(A.conj(), AO_Z, dims=([2],[2])).permute(0,2,1,3)
#             mi = mask[:, i].view(B, 1, 1, 1, 1)
#             Ei = torch.where(mi, Ei_Z.unsqueeze(0), Ei_I.unsqueeze(0))               # (B,ℓ,ℓ',r,r')
#             # Contract: ten(b,i,j,p,q) * Ei(b,p,q,r,s) -> (b,i,j,r,s)
#             ten = torch.einsum('bijpq,bpqrs->bijrs', ten, Ei)

#         # Close ring with double trace over i=r and j=s
#         vals = torch.einsum('bijij->b', ten)   # (B,)
#         total = total + torch.sum(coefs * vals)

#     return total.real  # scalar


# # ---------- helpers ----------

# def _site_transfer_lrd(A_lrd: torch.Tensor, O_2x2: torch.Tensor) -> torch.Tensor:
#     """
#     A_lrd: (ℓ, r, d), O_2x2: (2,2)
#     returns E(ℓ,ℓ',r,r')
#     """
#     AO = torch.einsum('lrd,dk->lrk', A_lrd, O_2x2)                         # (ℓ,r,k)
#     E  = torch.tensordot(A_lrd.conj(), AO, dims=([2],[2])).permute(0,2,1,3) # (ℓ,ℓ',r,r')
#     return E

# def _precompute_transfers_I_Z_lrd(tensors, device=None):
#     """Precompute per-site transfer tensors for I and Z, plus ΔE = E_Z - E_I."""
#     Z = torch.tensor([[1,0],[0,-1]], dtype=torch.cfloat, device=device)
#     I = torch.eye(2, dtype=torch.cfloat, device=device)
#     E_I, dE = [], []
#     for i in range(tensors.size(0)):
#         A = tensors[i].to(device)          # (ℓ,r,d)
#         Ei = _site_transfer_lrd(A, I)
#         Ez = _site_transfer_lrd(A, Z)
#         E_I.append(Ei)
#         dE.append(Ez - Ei)
#     return E_I, dE


# def expectation_value_batch(tensors: torch.Tensor,
#                                        hamiltonian,
#                                        device: str = None,
#                                        chunk_size: int | None = None) -> torch.Tensor:
#     """
#     Fast batched <ψ|H|ψ> for a Tensor Ring (periodic MPS).
#     tensors[i] shape: (ℓ_i, r_i, d) with d=2.
#     Hamiltonian must expose:
#       - .get_bool_pauli_tensor() -> Bool tensor of shape (n_terms, n_sites)
#         (True => apply Z at that site, False => Identity)
#       - .coefficients -> list/1D tensor of length n_terms (real or complex)
#     Returns a scalar tensor (real).
#     """
#     device = device or tensors.device
#     paulis = hamiltonian.get_bool_pauli_tensor().to(device)   # (T, N)
#     if paulis.dim() != 2:
#         raise ValueError("Expected paulis tensor of shape (n_terms, n_sites)")
#     T, N = paulis.shape
#     # Convert coefficients to tensor, handling mixed tensor/scalar lists
#     coeffs_list = [c.item() if hasattr(c, 'item') else c for c in hamiltonian.coefficients]
#     coeffs = torch.tensor(coeffs_list, dtype=torch.cfloat, device=device)  # (T,)

#     # Precompute once per site
#     E_I, dE = _precompute_transfers_I_Z_lrd(tensors, device=device)  # lists of length N

#     # Process all terms (or in chunks for memory control)
#     if chunk_size is None:
#         chunk_size = T

#     total = torch.zeros((), dtype=torch.cfloat, device=device)

#     for start in range(0, T, chunk_size):
#         stop = min(start + chunk_size, T)
#         mask_chunk = paulis[start:stop]            # (B, N) booleans
#         coeff_chunk = coeffs[start:stop]           # (B,)
#         B = mask_chunk.size(0)
        
#         # Build first site's batched transfer: E0 = E_I[0] + m0 * dE[0]
#         m0 = mask_chunk[:, 0].view(B, 1, 1, 1, 1)  # (B,1,1,1,1)
#         ten = E_I[0].unsqueeze(0) + m0 * dE[0].unsqueeze(0)  # (B, ℓ0,ℓ0', r0,r0')

#         # Chain the rest: ten(b,i,j,p,q) @ Ei(b,p,q,r,s) -> (b,i,j,r,s)
#         for i in range(1, N):
#             mi = mask_chunk[:, i].view(B, 1, 1, 1, 1)
#             Ei = E_I[i].unsqueeze(0) + mi * dE[i].unsqueeze(0)  # (B, ℓ,ℓ',r,r')
#             ten = torch.einsum('bijpq,bpqrs->bijrs', ten, Ei)

#         # Close the ring with a double trace: sum over i=r and j=s
#         vals = torch.einsum('bijij->b', ten)  # (B,) keep complex

#         # Since vals are complex and coeffs are complex, multiply and take real part
#         chunk_contribution = torch.sum(coeff_chunk * vals)
#         total = total + chunk_contribution

#     return total.real  # scalar, real

# COMMENTED OUT - WORKING VERSION WITHOUT COMPILATION
# def expectation_value(tensors: torch.Tensor,
#                                        hamiltonian,
#                                        device: str = None,
#                                        chunk_size: int | None = None) -> torch.Tensor:
#     """
#     Fast batched <ψ|H|ψ> for a Tensor Ring (periodic MPS).
#     tensors[i] shape: (ℓ_i, r_i, d) with d=2.
#     Hamiltonian must expose:
#       - .get_bool_pauli_tensor() -> Bool tensor of shape (n_terms, n_sites)
#         (True => apply Z at that site, False => Identity)
#       - .coefficients -> list/1D tensor of length n_terms (real or complex)
#     Returns a scalar tensor (real).
#     """
#     device = device or tensors.device
#     paulis = hamiltonian.get_bool_pauli_tensor().to(device)   # (T, N)
#     if paulis.dim() != 2:
#         raise ValueError("Expected paulis tensor of shape (n_terms, n_sites)")
#     T, N = paulis.shape
#     # Convert coefficients to tensor, handling mixed tensor/scalar lists
#     coeffs_list = [c.item() if hasattr(c, 'item') else c for c in hamiltonian.coefficients]
#     coeffs = torch.tensor(coeffs_list, dtype=torch.cfloat, device=device)  # (T,)
# 
#     # Precompute once per site
#     E_I, dE = _precompute_transfers_I_Z_lrd(tensors, device=device)  # lists of length N
# 
#     # Process all terms (or in chunks for memory control)
#     if chunk_size is None:
#         chunk_size = T
# 
#     total = torch.zeros((), dtype=torch.cfloat, device=device)
# 
#     for start in range(0, T, chunk_size):
#         stop = min(start + chunk_size, T)
#         mask_chunk = paulis[start:stop]            # (B, N) booleans
#         coeff_chunk = coeffs[start:stop]           # (B,)
#         B = mask_chunk.size(0)
#         val = mask_chunk[:, 0]
#         # Build first site's batched transfer: E0 = E_I[0] + m0 * dE[0]
#         m0 = mask_chunk[:, 0].view(B, 1, 1, 1, 1)  # (B,1,1,1,1)
#         ten = E_I[0].unsqueeze(0) + m0 * dE[0].unsqueeze(0)  # (B, ℓ0,ℓ0', r0,r0')
# 
#         # Chain the rest: ten(b,i,j,p,q) @ Ei(b,p,q,r,s) -> (b,i,j,r,s)
#         for i in range(1, N):
#             mi = mask_chunk[:, i].view(B, 1, 1, 1, 1)
#             Ei = E_I[i].unsqueeze(0) + mi * dE[i].unsqueeze(0)  # (B, ℓ,ℓ',r,r')
#             ten = torch.einsum('bijpq,bpqrs->bijrs', ten, Ei)
# 
#         # Close the ring with a double trace: sum over i=r and j=s
#         vals = torch.einsum('bijij->b', ten)  # (B,) keep complex
# 
#         # Since vals are complex and coeffs are complex, multiply and take real part
#         chunk_contribution = torch.sum(coeff_chunk * vals)
#         total = total + chunk_contribution
# 
#     return total.real  # scalar, real

# Optional jit/fusion (PyTorch 2.x) - disabled due to compilation issues with complex tensor operations
# The current function uses complex tensor arithmetic that torch.compile struggles with
# expectation_value = torch.compile(expectation_value)


# def expectation_value(tensor:torch.Tensor, hamiltonian: Hamiltonian, shot:float = 1e4,device:str = None):
#     prob_dist = measure(tensor, shot, device=device) # list of size 2**n 2 qubits {00 : 0, 01: 0.5, 10: 0.25, 11: 0.25} 
#     ret = 0.0
#     paulis_tensor = hamiltonian.get_bool_pauli_tensor().to(device=device) 
#     co = 0 
#     for paulis in paulis_tensor: # IZZI 
        
#         coef = hamiltonian.coefficients[co]
#         co+=1
#         val = 0
#         for i in range(0,len(prob_dist)):
#             prod = 1
#             #paulis = torch.flip(paulis, dims=(0,))
#             for j in range(0,len(paulis)): 
#                 if paulis[j] == True: # Z 
#                     if (i >> j) & 1 == 0: # 0
#                         prod *= 1
#                     else:
#                         prod *= -1
#                 else:
#                     prod *= 1
#             val += prob_dist[i]*prod
#         ret += coef * val
#     return ret

def expectation_value(tensor:torch.Tensor, hamiltonian: Hamiltonian, shot:float = 1e4,device:str = None):
    q0 = torch.tensor([[1], [0]], dtype=torch.cfloat).to(device)
    q1 = torch.tensor([[0], [1]], dtype=torch.cfloat).to(device)
    batch_prev: Optional[torch.Tensor] = None  # will be (shots, χ, χ)
    batch_coefs = torch.tensor(hamiltonian.coefficients, dtype=torch.cfloat, device=device).unsqueeze(0).repeat(int(shot), 1)


    paulis_tensor = hamiltonian.get_bool_pauli_tensor().to(device=device)
    for i, curr_ten in enumerate(tensor):
        # curr_ten: (χ1, χ2, 2)
        curr_ten = curr_ten.contiguous()
        if i == 0:
            # Project on q0/q1
            qubit_0 = torch.einsum('ijk,kl->ijl', curr_ten, q0).squeeze(-1)  # (χ1, χ2)
            qubit_1 = torch.einsum('ijk,kl->ijl', curr_ten, q1).squeeze(-1)
            batch_qubit_0 = qubit_0.unsqueeze(0).expand(int(shot), -1, -1).contiguous()  # (shots, χ1, χ2)
            batch_qubit_1 = qubit_1.unsqueeze(0).expand(int(shot), -1, -1).contiguous()
        else:
            # batch_prev: (shots, χ1, χ2)
            # curr_ten: (χ2, χ3, 2)
            if batch_prev is not None:
                contracted = torch.einsum('bij,jkl->bikl', batch_prev, curr_ten).contiguous()  # (shots, χ1, χ3, 2) 1e4 * rank^2 
            else:
                contracted = torch.tensor([])
            batch_qubit_0 = torch.einsum('bijk,kl->bijl', contracted, q0).squeeze(-1).contiguous()
            batch_qubit_1 = torch.einsum('bijk,kl->bijl', contracted, q1).squeeze(-1).contiguous()

        prob_0 = torch.einsum('bij,bij->b', batch_qubit_0.conj(), batch_qubit_0).real  # (shots,)
        prob_1 = torch.einsum('bij,bij->b', batch_qubit_1.conj(), batch_qubit_1).real
        total = prob_0 + prob_1  # (shots,)

        # If total == 0, assign prob_0 = 0.5, prob_1 = 0.5
        zero_mask = (total == 0)
        prob_0 = torch.where(zero_mask, torch.full_like(prob_0, 0.5), prob_0)
        prob_1 = torch.where(zero_mask, torch.full_like(prob_1, 0.5), prob_1)

        # Recompute total
        total = prob_0 + prob_1

        # Safe p0 computation
        p0 = prob_0 / total

        # Sampling
        rnd = torch.rand(int(shot), device=device)
        choose_1 = rnd > p0  # (shots,)

        # Select outcome
        batch_prev = torch.where(
            choose_1[:, None, None],  # (shots, 1, 1)
            batch_qubit_1,
            batch_qubit_0
        )

        # Update coefficients where Pauli is Z and outcome is 1
        mask = paulis_tensor[:, i].unsqueeze(0).expand(int(shot), -1)  # shape: (shots, num_terms)
        flip_mask = mask & choose_1.unsqueeze(1)  # shots × terms where to flip
        # batch_coefs[flip_mask] *= -1
        batch_coefs = torch.where(flip_mask, -batch_coefs, batch_coefs)
    return torch.mean(batch_coefs.sum(dim=1)).detach().cpu().item()  # (shots,)