"""Test torch.jit.script on contraction for autograd speedup."""
import torch, time
from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.optimization.gradients.autograd_gradient import _build_tensor_diff, _contraction_diff

device = 'cuda'
torch.manual_seed(42)


@torch.jit.script
def contraction_jit(tensor: torch.Tensor, paulis: torch.Tensor,
                    coeffs: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    N = tensor.shape[0]
    total = torch.zeros((), dtype=tensor.dtype, device=tensor.device)
    T = paulis.shape[0]
    for t in range(T):
        ten = torch.zeros(1, dtype=tensor.dtype, device=tensor.device)
        first: bool = True
        for i in range(N):
            curr = tensor[i].permute(0, 2, 1)
            if bool(paulis[t, i]):
                AO = torch.einsum('ldr,dk->lkr', curr, Z)
            else:
                AO = curr
            E = torch.tensordot(curr.conj(), AO, ([1], [1])).permute(0, 2, 1, 3)
            if first:
                ten = E
                first = False
            else:
                ten = torch.tensordot(ten, E, ([2, 3], [0, 1]))
        total = total + coeffs[t] * torch.einsum('ikik->', ten)
    return total.real


N, chi, L = 12, 10, 2
c = Circuit(num_qubit=N, rank=chi, device=device)
for i in range(N): c.h(i)
for _ in range(L):
    for i in range(N): c.cx(i, (i + 1) % N)
    for i in range(N): c.ry(i); c.rz(i)
h = Hamiltonian(num_qubits=N)
for i in range(N):
    j = (i + 1) % N; p = ['I'] * N; p[i] = 'Z'; p[j] = 'Z'
    h.add_pauli(''.join(p), 0.5)
h.add_pauli('I' * N, 1.0)
theta = torch.randn(c.params_size, device=device)

Z = torch.tensor([[1, 0], [0, -1]], dtype=torch.cfloat, device=device)
paulis = h.get_bool_pauli_tensor().to(device)
coeffs = torch.tensor(h.coefficients, dtype=torch.cfloat, device=device)


def ad_jit():
    with torch.enable_grad():
        t = theta.detach().clone().requires_grad_(True)
        tensor = _build_tensor_diff(t, c, torch.cfloat)
        loss = contraction_jit(tensor, paulis, coeffs, Z)
        loss.backward()
    return t.grad


def ad_py():
    with torch.enable_grad():
        t = theta.detach().clone().requires_grad_(True)
        tensor = _build_tensor_diff(t, c, torch.cfloat)
        loss = _contraction_diff(tensor, h, torch.cfloat)
        loss.backward()
    return t.grad


# Warmup
for _ in range(5): ad_jit()
for _ in range(5): ad_py()

# Time
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(10): ad_jit()
torch.cuda.synchronize()
ms_jit = (time.perf_counter() - t0) / 10 * 1000

torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(10): ad_py()
torch.cuda.synchronize()
ms_py = (time.perf_counter() - t0) / 10 * 1000

# Verify
g_jit = ad_jit()
g_py = ad_py()
cos = torch.nn.functional.cosine_similarity(g_jit.unsqueeze(0), g_py.unsqueeze(0)).item()

print(f'N={N} chi={chi} P={c.params_size} T={len(h.paulis)}')
print(f'  Python:  {ms_py:.0f}ms')
print(f'  JIT:     {ms_jit:.0f}ms')
print(f'  Speedup: {ms_py / ms_jit:.2f}x')
print(f'  cos:     {cos:.6f}')
