"""
Accuracy verification for O(chi^5) Kronecker optimizations.
Run: python test_optimization_accuracy.py
"""
import torch
from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.efficient_contraction import expectation_value_batch as exact_ev
from TREV.measure.right_suffix_sampling import (
    expectation_value as rss_ev,
    precompute_double_layer_and_right_suffix,
)
from TREV.optimization.gradients.batch_parameter_shift import (
    expectation_value_batch_efficient_contraction,
    batch_gradient,
)
from TREV.measure.enums import MeasureMethod

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_circuit(N, chi):
    c = Circuit(num_qubit=N, rank=chi, device=device)
    for i in range(N): c.h(i)
    for i in range(N): c.cx(i, (i+1)%N)
    for i in range(N): c.ry(i); c.rz(i)
    return c

def build_ham(N):
    h = Hamiltonian(num_qubits=N)
    for i in range(N):
        j = (i+1) % N; p = ['I']*N; p[i]='Z'; p[j]='Z'
        h.add_pauli(''.join(p), 0.5)
    h.add_pauli('I'*N, 1.0)
    return h

def old_precompute(cores):
    dtype = torch.complex128
    E_list = []
    for c in cores:
        c = c.to(dtype)
        B0, B1 = c[:,:,0], c[:,:,1]
        E_list.append(torch.kron(B0, B0.conj()) + torch.kron(B1, B1.conj()))
    n = len(E_list); d2 = E_list[0].shape[0]
    acc = torch.eye(d2, dtype=dtype, device=cores[0].device)
    Rp = [None]*(n+1); Rp[0] = acc
    for j in range(1, n+1):
        acc = E_list[n-j] @ acc; Rp[j] = acc
    return [Rp[n-(i+1)] for i in range(n)]

all_pass = True

# --- Test 1: R_suf numerical identity ---
print("1. R_suf (new Kronecker vs old kron)")
for N, chi in [(4,10), (8,32)]:
    circuit = build_circuit(N, chi)
    tensor = circuit.build_tensor(torch.randn(circuit.params_size, device=device))
    cores = [tensor[i] for i in range(N)]
    old_R = old_precompute(cores)
    _, new_R, _, _, _ = precompute_double_layer_and_right_suffix(cores)
    err = max(torch.abs(old_R[i] - new_R[i]).max().item() for i in range(N))
    ok = err < 1e-12
    all_pass &= ok
    print(f"   N={N:>2} chi={chi:>3}: err={err:.2e} {'PASS' if ok else 'FAIL'}")

# --- Test 2: Batched efficient contraction ---
print("\n2. Batched efficient contraction (vs exact)")
for N, chi in [(4,10), (8,32)]:
    circuit = build_circuit(N, chi); h = build_ham(N)
    B = 4; params = torch.randn(B, circuit.params_size, device=device)
    batch_vals = expectation_value_batch_efficient_contraction(params, circuit, h, shots=0)
    ref = torch.tensor([exact_ev(circuit.build_tensor(params[b]), h, device=device).item() for b in range(B)])
    err = (batch_vals.cpu() - ref).abs().max().item()
    ok = err < 1e-3
    all_pass &= ok
    print(f"   N={N:>2} chi={chi:>3}: err={err:.2e} {'PASS' if ok else 'FAIL'}")

# --- Test 3: Gradient cosine similarity ---
print("\n3. Gradient (param shift vs finite diff)")
for N, chi in [(4,10),]:
    circuit = build_circuit(N, chi); h = build_ham(N)
    theta = torch.randn(circuit.params_size, device=device)
    grad = batch_gradient(theta, circuit, h, chunk_size=16, shots=0, shift=0.5*3.14159,
                         depth=1, curr_depth=0, is_partial=False,
                         measure_method=MeasureMethod.EFFICIENT_CONTRACTION)
    eps = 1e-4; fd = torch.zeros_like(grad)
    for p in range(theta.numel()):
        tp = theta.clone(); tp[p] += eps
        fp = exact_ev(circuit.build_tensor(tp), h, device=device).item()
        tp[p] -= 2*eps
        fm = exact_ev(circuit.build_tensor(tp), h, device=device).item()
        fd[p] = (fp - fm) / (2*eps)
    cos = torch.nn.functional.cosine_similarity(grad.unsqueeze(0).float(), fd.unsqueeze(0).float()).item()
    ok = cos > 0.99
    all_pass &= ok
    print(f"   N={N:>2} chi={chi:>3}: cosine_sim={cos:.6f} {'PASS' if ok else 'FAIL'}")

# --- Test 4: RSS sampling ---
print("\n4. RSS sampling (vs exact, stochastic)")
for N, chi in [(4,10),]:
    circuit = build_circuit(N, chi); h = build_ham(N)
    tensor = circuit.build_tensor(torch.randn(circuit.params_size, device=device))
    cores = [tensor[i] for i in range(N)]
    exact = exact_ev(tensor, h, device=device).item()
    rss = rss_ev(cores, h, shots=10000, chunk_size=512, seed=42)
    err = abs(exact - rss)
    ok = err < 0.3
    all_pass &= ok
    print(f"   N={N:>2} chi={chi:>3}: exact={exact:.4f} rss={rss:.4f} err={err:.4f} {'PASS' if ok else 'FAIL'}")

print(f"\n{'='*40}")
print(f"ALL TESTS {'PASSED' if all_pass else 'FAILED'}")
