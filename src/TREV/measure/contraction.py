import torch, numpy as np
from torch import Tensor

from ..hamiltonian.hamiltonian import Hamiltonian


def expectation_value(tensor:Tensor, hamiltonian:Hamiltonian, device:str = None):
    """
    Compute expectation value <psi|H|psi> using full contraction.
    
    Args:
        tensor: Tensor ring representation of quantum state
        hamiltonian: Hamiltonian operator
        device: Device for computation
        
    Returns:
        Real-valued expectation value
    """
    prob = torch.tensor(measure(tensor), dtype=torch.cfloat, device=device)
    density_matrix = hamiltonian.get_density_matrix().to(device)
    print(prob)
    print(density_matrix)
    # prob is shape (2^N,), density_matrix is (2^N, 2^N)
    # Expectation value: sum of prob[i] * H[i,i] (diagonal elements)
    diagonal = torch.diag(density_matrix)
    return torch.sum(prob * diagonal).real.detach().cpu().item()

def contract_tensor_ring(psi: torch.Tensor) -> torch.Tensor:
    """
    psi  : (N, χ1, χ2, 2)
    return: (2,)*N
    """
    N = psi.shape[0]                 # number of qubits/sites

    if N == 1:
        # Special case: single qubit tensor ring
        # Contract left and right bonds: trace over χ dimensions
        # psi[0] has shape (χ1, χ2, 2)
        # Close the ring by contracting left and right bonds
        return torch.einsum('iid->d', psi[0])  # (2,)

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


def contract_mps(psi: torch.Tensor) -> torch.Tensor:
    """
    Open-boundary MPS contraction.
    psi  : (N, χ, χ, 2)
    return: (2,)*N

    Same as contract_tensor_ring but picks boundary index 0
    instead of tracing over the periodic bond.
    """
    N = psi.shape[0]

    if N == 1:
        return psi[0][0, 0, :]           # (2,)

    # Start: select left boundary index 0 → (χ_right, 2)
    psi_new = psi[0][0, :, :]            # (χ, 2)

    for i in range(1, N - 1):
        # psi_new: (χ, 2, ..., 2) with right bond at dim 0
        # psi[i]:  (χ_left, χ_right, 2)
        psi_new = torch.tensordot(psi_new, psi[i], dims=([0], [0]))
        # move new right bond (second-to-last dim) back to position 0
        psi_new = torch.movedim(psi_new, -2, 0)

    # End: select right boundary index 0 from last site
    last = psi[-1][:, 0, :]              # (χ_left, 2)
    psi_new = torch.tensordot(psi_new, last, dims=([0], [0]))
    return psi_new                        # (2, 2, ..., 2)


def measure(ring_tensors: torch.Tensor) -> Tensor:
    """
    ring_tensors : (N, χ1, χ2, 2)
    returns      : (2**N,)  –– probabilities of all computational‑basis states
    Big-endian: site 0 = MSB
    """
    N   = ring_tensors.shape[0]
    psi = contract_tensor_ring(ring_tensors)          # (2,…,2)

    # Big-endian ordering: site 0 is most significant bit (MSB)
    # No reordering needed - natural tensor order

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

def _left_mul_E_4d(A0, A1, R_4d, chi):
    """Compute E @ R in 4D using Kronecker-factored O(chi^5) contractions.

    E = kron(A0, conj(A0)) + kron(A1, conj(A1))  (full transfer matrix)
    R_4d: (chi, chi, chi, chi) — right environment in 4D form.
    Result: (chi, chi, chi, chi)
    """
    cd = chi * chi
    R_r = R_4d.reshape(chi, chi, cd)              # (i, j, c*d)
    # Contract ket index j with conj(As), then bra index i with As
    temp0 = torch.matmul(A0.conj(), R_r)           # (i, b, c*d)
    temp1 = torch.matmul(A1.conj(), R_r)           # (i, b, c*d)
    bcd = chi * cd
    temp0 = temp0.reshape(chi, bcd)                 # (i, b*c*d)
    temp1 = temp1.reshape(chi, bcd)
    r0 = torch.matmul(A0, temp0).reshape(chi, chi, chi, chi)  # (a, b*c*d) → 4D
    r1 = torch.matmul(A1, temp1).reshape(chi, chi, chi, chi)
    return r0 + r1


def _right_mul_L_Es_4d(L_4d, As, chi):
    """Compute L @ E_s in 4D for a single spin value s.

    E_s = kron(As, conj(As))
    L_4d: (chi, chi, chi, chi) — left environment in 4D form.
    Result: (chi, chi, chi, chi)
    """
    # Step 1: contract last dim of L (ket) with conj(As)
    temp = torch.matmul(L_4d, As.conj())           # (a, b, i, d) — O(chi^5)
    # Step 2: contract dim i (bra) with As
    temp_3d = temp.reshape(chi * chi, chi, chi)     # (a*b, i, d)
    result = torch.matmul(As.mT, temp_3d)           # (a*b, c, d) — O(chi^5)
    return result.reshape(chi, chi, chi, chi)


@torch.no_grad()
def argmax_tr_noinv_BE(cores, tie_break='random'):
    """
    Greedy highest-probability bitstring on a periodic MPS / tensor ring.
    - big-endian (site 0 = MSB)
    - no solves/inverses
    - single pass using left & right environments
    - tie_break: '0', '1', or 'random'

    Uses Kronecker-factored O(chi^5) contractions instead of O(chi^6).
    """
    n = len(cores)
    device = cores[0].device
    dtype = torch.complex128 if torch.is_complex(cores[0]) else torch.float64

    # Extract A0, A1 matrices from cores: each core is (chi, chi, 2)
    core_list = []
    for c in cores:
        c = c.to(dtype)
        core_list.append((c[:, :, 0], c[:, :, 1]))

    chi = cores[0].shape[0]

    # Identity in 4D: eye(chi^2).reshape(chi, chi, chi, chi)
    I_4d = torch.eye(chi * chi, dtype=dtype, device=device).reshape(chi, chi, chi, chi)

    # Build right suffixes in 4D
    # R_suf[i] = E_{i+1} @ ... @ E_{n-1}
    R_suf = [None] * n
    acc = I_4d.clone()
    R_suf[n - 1] = acc
    for k in range(n - 2, -1, -1):
        A0, A1 = core_list[k + 1]
        acc = _left_mul_E_4d(A0, A1, acc, chi)
        R_suf[k] = acc

    out = torch.empty(n, dtype=torch.long, device='cpu')

    # Left environment in 4D
    L = I_4d.clone()

    for i in range(n):
        A0, A1 = core_list[i]

        # T_s = L @ E_i(s) for s=0,1
        T0 = _right_mul_L_Es_4d(L, A0, chi)
        T1 = _right_mul_L_Es_4d(L, A1, chi)

        Ri = R_suf[i]
        Ri_perm = Ri.permute(2, 3, 0, 1)

        # w_s = Tr(T_s @ Ri) = (T_s * Ri_perm).sum()
        w0 = (T0 * Ri_perm).sum().real
        w1 = (T1 * Ri_perm).sum().real

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

        # Stabilization
        nL = L.reshape(-1).norm().clamp_min(1e-300)
        L = L / nL

    return out
