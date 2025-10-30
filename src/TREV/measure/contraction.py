import torch, numpy as np
from torch import Tensor

from ..hamiltonian.hamiltonian import Hamiltonian


def expectation_value(tensor:Tensor, hamiltonian:Hamiltonian, device:str = None):
    prob = torch.tensor(measure(tensor), dtype= torch.cfloat, device=device)
    return torch.sum(prob@hamiltonian.get_density_matrix().to(device)).detach().cpu().item()

def contract_tensor_ring(psi: torch.Tensor) -> torch.Tensor:
    """
    psi  : (N, χ1, χ2, 2)
    return: (2,)*N
    """
    N = psi.shape[0]                 # number of qubits/sites

    psi_new = psi[0]                 # (χ1, χ2, 2)

    # sequentially fuse sites 1 … N‑2
    for i in range(1, N - 1):
        # contract right bond of psi_new with left bond of psi[i]
        psi_new = torch.tensordot(psi_new, psi[i], dims=([1], [0]))
        # put new right bond back in axis 1 so the pattern repeats
        psi_new = torch.movedim(psi_new, -2, 1)

    # close the ring with the last site  (swap indices to match bonds)
    psi_new = torch.tensordot(psi_new, psi[-1], dims=([0, 1], [1, 0]))
    return psi_new                   # shape (2, 2, …, 2)

def measure(ring_tensors: torch.Tensor) -> Tensor:
    """
    ring_tensors : (N, χ1, χ2, 2)
    returns      : (2**N,)  –– probabilities of all computational‑basis states
    """
    N   = ring_tensors.shape[0]
    psi = contract_tensor_ring(ring_tensors)          # (2,…,2)

    # reorder indices so qubit‑0 is least‑significant (optional, same as before)
    for i in range(N - 1, 0, -1):
        psi = psi.movedim(0, i)

    prob = (psi * psi.conj()).real                    # (2,…,2)
    return prob.reshape(-1).detach().cpu().numpy()            # (2**N,)


def get_value_of_highest_probability(tensors: torch.Tensor, device:str):
    q0 = torch.tensor([[1], [0]], dtype=torch.cfloat, device=device)
    q1 = torch.tensor([[0], [1]], dtype=torch.cfloat, device=device)
    prev = None
    val = ''
    for i in range(int(len(tensors))):
        if i == int(0):
            # curr_ten =  χ1 × χ2 * 2
            curr_ten = tensors[i]

            qubit_0 = torch.tensordot(curr_ten, q0, ([2], [0])).squeeze(-1)
            qubit_1 = torch.tensordot(curr_ten, q1, ([2], [0])).squeeze(-1)

            prob_0 = torch.tensordot(qubit_0, qubit_0.mH, ([1, 0], [0, 1])).real.item()
            prob_1 = torch.tensordot(qubit_1, qubit_1.mH, ([1, 0], [0, 1])).real.item()
            total = prob_0 + prob_1

        else:
            # curr_ten =  X1 * X2 * 2
            curr_ten = tensors[i]
            curr_ten = torch.tensordot(prev, curr_ten, ([1], [0]))
            qubit_0 = torch.tensordot(curr_ten, q0, ([2], [0])).squeeze(-1)
            qubit_1 = torch.tensordot(curr_ten, q1, ([2], [0])).squeeze(-1)

            prob_0 = torch.tensordot(qubit_0, qubit_0.mH, ([1, 0], [0, 1])).real.item()
            prob_1 = torch.tensordot(qubit_1, qubit_1.mH, ([1, 0], [0, 1])).real.item()
            total = prob_0 + prob_1
        if total == 0:
            rnd = torch.rand((1, 1))[0].item()
            if rnd > 0.5:
                prev = qubit_1
                val = '1' + val
            else:
                prev = qubit_0
                val = '0' + val
        elif prob_1 / total > prob_0 / total:
            prev = qubit_1
            val = '1' + val
        else:
            prev = qubit_0
            val = '0' + val
    return val

def kron(A, B):  # alias
    return torch.kron(A, B)

def E_site(core):  # core: [chi,2,chi] -> (E, (E0,E1)) in double layer
    B0, B1 = core[:,:,0], core[:,:,1]
    E0 = kron(B0, B0.conj())
    E1 = kron(B1, B1.conj())
    # E0 = B0
    # E1 = B1
    return (E0 + E1), (E0, E1)

@torch.no_grad()
def precompute_double_layer_and_right_suffix(cores):
    """
    Returns:
      Es: [(E0,E1)] length n, each [chi^2,chi^2]
      R_suf: [n] where R_suf[i] = E_{i+1} ... E_{n-1} (identity if i==n-1)
      d2, device, dtype
    """
    device = cores[0].device
    dtype  = torch.complex128 if torch.is_complex(cores[0]) else torch.float64

    E_list, Es = [], []
    for c in cores:
        c = c.to(dtype)
        Ei, (Ei0, Ei1) = E_site(c)
        E_list.append(Ei)
        Es.append((Ei0, Ei1))

    n  = len(E_list)
    d2 = E_list[0].shape[0]
    I  = torch.eye(d2, dtype=E_list[0].dtype, device=device)

    # Build right suffixes with a left->right pass on the reversed list,
    # but crucially LEFT-multiply to preserve forward order:
    # Rpref_rev[j] = E_{n-1} ... E_{n-j}
    Rpref_rev = [None] * (n + 1)
    acc = I
    Rpref_rev[0] = acc
    E_rev = E_list[::-1]  # [E_{n-1}, E_{n-2}, ..., E_0]
    for j in range(1, n + 1):
        acc = E_rev[j - 1] @ acc        # <-- left-multiply (critical)
        Rpref_rev[j] = acc

    # Map back: for site i, R_suf[i] = E_{i+1} ... E_{n-1} = Rpref_rev[n-(i+1)]
    R_suf = [Rpref_rev[n - (i + 1)] for i in range(n)]
    return Es, R_suf, d2, device, dtype

@torch.no_grad()
def argmax_tr_noinv_LE(cores, tie_break='random'):
    """
    Greedy highest-probability bitstring on a periodic MPS / tensor ring.
    - little-endian (site 0 = LSB)
    - no solves/inverses
    - single pass using left & right environments
    - tie_break: '0', '1', or 'random'
    """
    Es, R_suf, d2, device, dtype = precompute_double_layer_and_right_suffix(cores)
    n = len(Es)

    out = torch.empty(n, dtype=torch.long, device='cpu')

    # Left environment at the cut
    L = torch.eye(d2, dtype=dtype, device=device)

    for i in range(n):
        Ei0, Ei1 = Es[i]
        T0 = L @ Ei0
        T1 = L @ Ei1
        Ri = R_suf[i]

        # Unnormalized weights: w_s = Tr( (L E_i(s)) Ri )
        w0 = torch.trace(T0 @ Ri).real
        w1 = torch.trace(T1 @ Ri).real

        # Clamp small/negative numerical noise
        w0c = torch.clamp(w0, min=0.0)
        w1c = torch.clamp(w1, min=0.0)

        if (w0c == 0) and (w1c == 0):
            # Pure tie / underflow — apply chosen rule
            if tie_break == '1':
                si = 1
            elif tie_break == 'random':
                si = int(torch.rand((), device=device).item() >= 0.5)
            else:
                si = 0
        else:
            si = 1 if w1c > w0c else 0

        out[i] = si
        L = T1 if si == 1 else T0  # update left environment

        # Optional stabilization
        nL = torch.linalg.norm(L.reshape(-1)).clamp_min(1e-300)
        L = L / nL

    return out
