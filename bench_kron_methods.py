"""
Benchmark different implementations of the Kronecker contraction inner loop.
Run: python bench_kron_methods.py
"""
import time, torch, numpy as np

device = 'cuda'

def bench(fn, warmup=10, repeats=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        times.append((time.perf_counter()-t0)*1000)
    return np.median(times)


# --- JIT scripted version ---
@torch.jit.script
def jit_kron(Prod: torch.Tensor, A0: torch.Tensor, A1: torch.Tensor,
             sign: torch.Tensor, chi: int, chi2: int) -> torch.Tensor:
    BT = Prod.shape[0]
    P4 = Prod.reshape(BT, chi2, chi, chi)
    t = torch.matmul(P4, A0.unsqueeze(1))
    r0 = torch.matmul(A0.conj().mT.unsqueeze(1), t).reshape(BT, chi2, chi2)
    t2 = torch.matmul(P4, A1.unsqueeze(1))
    r1 = torch.matmul(A1.conj().mT.unsqueeze(1), t2).reshape(BT, chi2, chi2)
    return r0 + sign * r1


# --- JIT: full N-site loop ---
@torch.jit.script
def jit_kron_loop(Prod: torch.Tensor, A0_all: torch.Tensor, A1_all: torch.Tensor,
                  signs: torch.Tensor, N: int, chi: int, chi2: int) -> torch.Tensor:
    BT = Prod.shape[0]
    for i in range(N):
        A0 = A0_all[:, i]
        A1 = A1_all[:, i]
        s = signs[:, i].unsqueeze(1).unsqueeze(2)  # (BT, 1, 1)
        P4 = Prod.reshape(BT, chi2, chi, chi)
        t = torch.matmul(P4, A0.unsqueeze(1))
        r0 = torch.matmul(A0.conj().mT.unsqueeze(1), t).reshape(BT, chi2, chi2)
        t2 = torch.matmul(P4, A1.unsqueeze(1))
        r1 = torch.matmul(A1.conj().mT.unsqueeze(1), t2).reshape(BT, chi2, chi2)
        Prod = r0 + s * r1
    return Prod


for chi in [4, 10]:
    N = 8
    BT = 128
    chi2 = chi * chi
    ctype = torch.cfloat

    Prod = torch.randn(BT, chi2, chi2, dtype=ctype, device=device)
    A0_all = torch.randn(BT, N, chi, chi, dtype=ctype, device=device)
    A1_all = torch.randn(BT, N, chi, chi, dtype=ctype, device=device)
    signs_all = torch.randint(0, 2, (BT, N), device=device).float().to(ctype) * -2 + 1  # +1 or -1
    sign_one = signs_all[:, 0:1].unsqueeze(-1)  # (BT, 1, 1)

    A0 = A0_all[:, 0]
    A1 = A1_all[:, 0]

    # --- Current: Python loop of 4 matmuls per site ---
    def current_loop():
        P = Prod.clone()
        for i in range(N):
            a0 = A0_all[:, i]; a1 = A1_all[:, i]
            s = signs_all[:, i].view(BT, 1, 1)
            P4 = P.reshape(BT, chi2, chi, chi)
            t = torch.matmul(P4, a0.unsqueeze(1))
            r0 = torch.matmul(a0.conj().mT.unsqueeze(1), t).reshape(BT, chi2, chi2)
            t = torch.matmul(P4, a1.unsqueeze(1))
            r1 = torch.matmul(a1.conj().mT.unsqueeze(1), t).reshape(BT, chi2, chi2)
            P = r0 + s * r1
        return P

    # --- Einsum per site ---
    def einsum_loop():
        P = Prod.clone()
        for i in range(N):
            a0 = A0_all[:, i]; a1 = A1_all[:, i]
            s = signs_all[:, i].view(BT, 1, 1)
            P5 = P.reshape(BT, chi, chi, chi, chi)
            r0 = torch.einsum('xbc,xaAbB,xBC->xaAcC', a0.conj(), P5, a0).reshape(BT, chi2, chi2)
            r1 = torch.einsum('xbc,xaAbB,xBC->xaAcC', a1.conj(), P5, a1).reshape(BT, chi2, chi2)
            P = r0 + s * r1
        return P

    # --- JIT loop ---
    def jit_loop():
        return jit_kron_loop(Prod.clone(), A0_all, A1_all, signs_all, N, chi, chi2)

    # --- CUDA graph ---
    def make_cuda_graph():
        P_in = Prod.clone()
        A0_in = A0_all.clone()
        A1_in = A1_all.clone()
        s_in = signs_all.clone()
        # warmup
        for _ in range(3):
            jit_kron_loop(P_in, A0_in, A1_in, s_in, N, chi, chi2)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = jit_kron_loop(P_in, A0_in, A1_in, s_in, N, chi, chi2)
        def run():
            P_in.copy_(Prod)
            g.replay()
            return out
        return run

    t_cur = bench(current_loop)
    t_ein = bench(einsum_loop)

    # JIT warmup
    for _ in range(5): jit_loop()
    t_jit = bench(jit_loop)

    # CUDA graph
    try:
        graph_fn = make_cuda_graph()
        t_graph = bench(graph_fn)
    except Exception as e:
        t_graph = float('nan')
        print(f'  CUDA graph failed: {e}')

    print(f'chi={chi} N={N} BT={BT}:')
    print(f'  current (Python loop):  {t_cur:.3f}ms')
    print(f'  einsum loop:            {t_ein:.3f}ms  ({t_cur/t_ein:.2f}x)')
    print(f'  jit.script loop:        {t_jit:.3f}ms  ({t_cur/t_jit:.2f}x)')
    if not np.isnan(t_graph):
        print(f'  CUDA graph:             {t_graph:.3f}ms  ({t_cur/t_graph:.2f}x)')
    print()
