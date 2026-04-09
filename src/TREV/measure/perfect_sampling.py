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
        for i in range(tensor.size(0)):
            if i == int(0):
                # curr_ten =  χ1 × χ2 * 2
                curr_ten = tensor[i]

                qubit_0 = torch.tensordot(curr_ten, q0, ([2], [0])).squeeze(-1)
                qubit_1 = torch.tensordot(curr_ten, q1, ([2], [0])).squeeze(-1)

                prob_0 = torch.tensordot(qubit_0, qubit_0.mH, ([1, 0], [0, 1])).real.item()
                prob_1 = torch.tensordot(qubit_1, qubit_1.mH, ([1, 0], [0, 1])).real.item()
                total = prob_0 + prob_1
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