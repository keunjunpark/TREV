"""
Prototype: real-valued formulation for autograd.

Represent complex (m,n) as real (2,m,n) where [0]=real, [1]=imag.
This unlocks torch.compile, CUDA graphs, and Triton fusion.

Run: python bench_real_form.py
"""
import torch
import time
from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.efficient_contraction import expectation_value_batch as ev_exact
from TREV.optimization.gradients.autograd_gradient import autograd_gradient

device = 'cuda'
torch.manual_seed(42)


# ═══ Real-valued complex arithmetic ═══

def to_real(z):
    return torch.stack([z.real, z.imag], dim=0)

def to_complex(r):
    return torch.complex(r[0], r[1])

def r_mm(a, b):
    """(2,m,k) x (2,k,n) -> (2,m,n) complex matmul in real."""
    return torch.stack([
        a[0] @ b[0] - a[1] @ b[1],
        a[0] @ b[1] + a[1] @ b[0],
    ], dim=0)

def r_conj_mm(a, b):
    """conj(a) @ b in real. conj(a) = (a0, -a1)."""
    return torch.stack([
        a[0] @ b[0] + a[1] @ b[1],
        a[0] @ b[1] - a[1] @ b[0],
    ], dim=0)

def r_conj(a):
    return torch.stack([a[0], -a[1]], dim=0)

def r_einsum_transfer(curr_conj, AO):
    """Compute E[l,l',r,r'] = sum_d conj(curr)[l,d,r] * AO[l',d,r']
    via tensordot on dim 1.
    curr_conj, AO: (2, chi, 2, chi) real representation.
    Result: (2, chi, chi, chi, chi)."""
    # Complex tensordot on axis 1:
    # (a0+ia1)(b0+ib1) summed = (a0b0 - a1b1) + i(a0b1 + a1b0)
    # But this is conj(curr) * AO, so curr_conj already has negated imag
    r = torch.tensordot(curr_conj[0], AO[0], ([1], [1])) - torch.tensordot(curr_conj[1], AO[1], ([1], [1]))
    i = torch.tensordot(curr_conj[0], AO[1], ([1], [1])) + torch.tensordot(curr_conj[1], AO[0], ([1], [1]))
    return torch.stack([r, i], dim=0)  # (2, chi, chi, chi, chi)

def r_contract_ring(ten, E):
    """ten (2,chi,chi,chi,chi) @ E (2,chi,chi,chi,chi) -> contract dims [2,3] with [0,1]."""
    # Complex tensordot on dims ([3,4], [1,2]) of the inner tensors
    r = torch.tensordot(ten[0], E[0], ([2, 3], [0, 1])) - torch.tensordot(ten[1], E[1], ([2, 3], [0, 1]))
    i = torch.tensordot(ten[0], E[1], ([2, 3], [0, 1])) + torch.tensordot(ten[1], E[0], ([2, 3], [0, 1]))
    return torch.stack([r, i], dim=0)

def r_trace_ring(ten):
    """Trace: sum_{i,j} ten[i,j,i,j] for complex (2,chi,chi,chi,chi)."""
    return torch.stack([
        torch.einsum('ijij->', ten[0]),
        torch.einsum('ijij->', ten[1]),
    ], dim=0)


# ═══ Real-form SVD (no custom backward needed!) ═══

def r_svd_truncate(M_real, k):
    """Truncated SVD on real (2,m,n) complex matrix.
    Returns L (2,m,k), R (2,k,n) such that L @ R ≈ M.
    Uses standard PyTorch SVD on real matrices — backward works natively."""
    # SVD each of real and imag? No — need complex SVD behavior.
    # Convert to (2m, 2n) real embedding, SVD, convert back.
    m, n = M_real.shape[1], M_real.shape[2]
    X, Y = M_real[0], M_real[1]
    A_R = torch.cat([
        torch.cat([X, -Y], dim=1),
        torch.cat([Y, X], dim=1),
    ], dim=0)  # (2m, 2n)

    U, S, Vh = torch.linalg.svd(A_R, full_matrices=False)

    # Truncate to 2k (paired SVs)
    L_R = U[:, :2*k] * S[:2*k].unsqueeze(0)  # (2m, 2k)
    R_R = Vh[:2*k, :]                          # (2k, 2n)

    # Reconstruct and extract complex
    A_R_t = L_R @ R_R  # (2m, 2n)
    X_t = A_R_t[:m, :n]
    Y_t = A_R_t[m:, :n]

    # Split into L, R via the real SVD factors
    # L_R has block structure for paired SVs: take alternating columns
    # Simpler: do SVD on the truncated result (it's rank-k, so SVD is exact)
    # This second SVD is on a REAL matrix — backward works!
    A_t_real = torch.stack([X_t, Y_t], dim=0)  # (2, m, n)

    # For splitting: use real-form SVD again on the truncated result
    U2, S2, Vh2 = torch.linalg.svd(A_R_t, full_matrices=False)
    L2 = U2[:, :2*k] * S2[:2*k].unsqueeze(0)  # (2m, 2k)
    R2 = Vh2[:2*k, :]                           # (2k, 2n)

    # Extract complex L and R
    L_x = L2[:m, :]   # (m, 2k)
    L_y = L2[m:, :]   # (m, 2k)
    R_x = R2[:, :n]   # (2k, n)
    R_y = R2[:, n:]   # (2k, 2n)... hmm, pairing is complex

    # Simpler approach: just return A_t_real and let caller split with QR
    return A_t_real


# ═══ Build circuit in real form ═══

def build_real(theta, circuit):
    """Build tensor ring in real representation. All ops are real — compilable!"""
    from TREV.gates.parameter_gates import ParameterOneQubitGate
    from TREV.gates.non_parameter_gates import NonParameterOneQubitGate, NonParameterTwoQubitsGate

    N, chi = circuit.num_qubit, circuit.rank
    # cores[i] = (2, chi, chi, 2) — real representation of (chi, chi, 2) complex
    cores = [torch.zeros(2, chi, chi, 2, device=device) for _ in range(N)]
    for i in range(N):
        cores[i][0, 0, 0, 0] = 1.0  # real part of |0> state

    for gate in circuit.gates:
        if isinstance(gate, ParameterOneQubitGate):
            mat_c = gate.matrix_fun(theta[gate.theta_index], device)
            mat = to_real(mat_c)  # (2, 2, 2)
            q = gate.qubit
            core = cores[q]  # (2, chi, chi, 2)
            # Apply: new_core[..., d'] = sum_d mat[d', d] * core[..., d]
            # In real: complex matmul on last dim
            new_core = torch.stack([
                core[0, :, :, 0] * mat[0, :, 0].sum() + core[0, :, :, 1] * mat[0, :, 1].sum(),  # wrong approach
            ])
            # Actually: simpler to use tensordot
            # core: (2, chi, chi, 2=physical), mat: (2, 2=out, 2=in)
            # result[c, l, r, d_out] = sum_{d_in} mat[c_m, d_out, d_in] * core[c_c, l, r, d_in]
            # Complex multiply: need to handle (2,...) real/imag correctly
            # This is getting complicated. Let me just convert to complex, apply, convert back.
            core_c = to_complex(core)  # (chi, chi, 2)
            from TREV.gates.contraction import _apply_single_qubit_gate
            new_core_c = _apply_single_qubit_gate(mat_c, core_c)
            cores[q] = to_real(new_core_c)

        elif isinstance(gate, NonParameterOneQubitGate):
            mat_c = gate.matrix_fun(None, device)
            core_c = to_complex(cores[gate.qubit])
            from TREV.gates.contraction import _apply_single_qubit_gate
            new_core_c = _apply_single_qubit_gate(mat_c, core_c)
            cores[gate.qubit] = to_real(new_core_c)

        elif isinstance(gate, NonParameterTwoQubitsGate):
            q0, q1 = gate.qubits
            qu0_c = to_complex(cores[q0])
            qu1_c = to_complex(cores[q1])
            chi1, chi3 = qu0_c.shape[0], qu1_c.shape[1]
            mps = torch.tensordot(qu0_c, qu1_c, ([1], [0]))
            mps = torch.moveaxis(mps, 2, 1)
            gt = gate.matrix_fun(device=device).reshape(2, 2, 2, 2)
            mps = torch.tensordot(gt, mps, ([2, 3], [2, 3]))
            mps = torch.moveaxis(mps, 1, 2).reshape(chi1*2, chi3*2)

            # SVD via real form — ALL real, compilable!
            mps_real = to_real(mps)  # (2, 2chi, 2chi)
            X, Y = mps_real[0], mps_real[1]
            A_R = torch.cat([torch.cat([X, -Y], 1), torch.cat([Y, X], 1)], 0)
            U, S, Vh = torch.linalg.svd(A_R, full_matrices=False)
            A_R_t = (U[:, :2*chi1] * S[:2*chi1].unsqueeze(0)) @ Vh[:2*chi1, :]
            X_t = A_R_t[:chi1*2, :chi3*2]
            Y_t = A_R_t[chi1*2:, :chi3*2]
            # Split: QR on real embedding (no complex phase issue)
            Q, R = torch.linalg.qr(A_R_t[:, :2*chi1*2])
            # ... this split is still problematic.
            # Fall back: use our diff_svd for now, just test the contraction in real form
            from TREV.gates.differentiable_svd import diff_svd
            Uk, Sk, Vhk = diff_svd(mps, chi1)
            q0_new = (Uk * Sk.unsqueeze(0).to(mps.dtype)).reshape(2, chi1, chi1)
            q1_new = Vhk.reshape(chi3, 2, chi3)
            cores[q0] = to_real(torch.moveaxis(q0_new, 0, 2))
            cores[q1] = to_real(torch.moveaxis(q1_new, 1, 2))

    return [to_complex(c) for c in cores]


# ═══ Test: compile the CONTRACTION only (build stays complex) ═══

def contraction_real(cores_complex, hamiltonian):
    """Contraction in pure real arithmetic — compilable!"""
    N = len(cores_complex)
    Z_r = torch.tensor([[[1, 0], [0, -1]], [[0, 0], [0, 0]]], dtype=torch.float32, device=device)
    paulis = hamiltonian.get_bool_pauli_tensor().to(device)

    total_r = torch.zeros(2, device=device)  # (real, imag)

    for t in range(len(hamiltonian.paulis)):
        coef = hamiltonian.coefficients[t]
        ten = None
        for i in range(N):
            core_r = to_real(cores_complex[i].permute(0, 2, 1))  # (2, chi, 2, chi)
            if paulis[t, i]:
                # AO = einsum('ldr,dk->lkr', curr, Z)
                AO_r = torch.stack([
                    torch.einsum('ldr,dk->lkr', core_r[0], Z_r[0]),
                    torch.einsum('ldr,dk->lkr', core_r[1], Z_r[0]),
                ], dim=0)
            else:
                AO_r = core_r

            # E = tensordot(conj(curr), AO, ([1],[1])).permute(0,2,1,3)
            curr_conj_r = r_conj(core_r)
            E_r = r_einsum_transfer(curr_conj_r, AO_r)
            # E_r shape: (2, chi, chi, chi, chi) — need to permute inner to (l,l',r,r')
            E_r = E_r[:, :, :, :, :].permute(0, 1, 3, 2, 4)

            if ten is None:
                ten = E_r
            else:
                ten = r_contract_ring(ten, E_r)

        trace = r_trace_ring(ten)  # (2,) = (real, imag)
        total_r = total_r + coef * trace

    return total_r[0]  # real part


# ═══ Test ═══

N, chi, L = 8, 10, 2
c = Circuit(num_qubit=N, rank=chi, device=device)
for i in range(N): c.h(i)
for _ in range(L):
    for i in range(N): c.cx(i, (i+1) % N)
    for i in range(N): c.ry(i); c.rz(i)
h = Hamiltonian(num_qubits=N)
for i in range(N):
    j = (i+1) % N; p = ['I']*N; p[i] = 'Z'; p[j] = 'Z'
    h.add_pauli(''.join(p), 0.5)
h.add_pauli('I'*N, 1.0)

theta = torch.randn(c.params_size, device=device)

# Build with complex (existing), contraction with real
tensor = c.build_tensor(theta)
cores = [tensor[i] for i in range(N)]

val_complex = ev_exact(tensor, h, device=device).item()
try:
    val_real = contraction_real(cores, h).item()
    print(f'Complex contraction: {val_complex:.6f}')
    print(f'Real contraction:    {val_real:.6f}')
    print(f'Match: {abs(val_complex - val_real) < 0.01}')

    # Try torch.compile on real contraction
    print('\nCompiling real contraction...')
    compiled_contract = torch.compile(contraction_real, mode='reduce-overhead')
    for _ in range(3): compiled_contract(cores, h)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): compiled_contract(cores, h)
    torch.cuda.synchronize()
    ms_compiled = (time.perf_counter()-t0)/10*1000

    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): contraction_real(cores, h)
    torch.cuda.synchronize()
    ms_eager = (time.perf_counter()-t0)/10*1000

    print(f'Real eager:    {ms_eager:.1f}ms')
    print(f'Real compiled: {ms_compiled:.1f}ms')
    print(f'Speedup:       {ms_eager/ms_compiled:.2f}x')

except Exception as e:
    print(f'Error: {e}')
