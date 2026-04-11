"""
Comprehensive TREV profiling script.

Profiles every major component to identify actual bottlenecks before
deciding on optimization strategy (CUDA kernels, torch.compile, etc.)

Usage:
    python profiling/profile_bottleneck.py [--n N] [--rank RANK] [--depth DEPTH]

Requires: pip install -e . (or run from TREV root with src in path)
"""
import torch
import time
import argparse
import sys
sys.path.insert(0, "src")

from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.enums import MeasureMethod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gpu_info():
    if not torch.cuda.is_available():
        return "CPU only"
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    return f"{name} | {free/1e9:.1f} GB free / {total/1e9:.1f} GB total"


def timed(fn, warmup=1, repeats=3, label=""):
    """Run fn with warmup, return (avg_seconds, results_of_last_run)."""
    for _ in range(warmup):
        result = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    print(f"  {label:40s} {avg:.4f}s  (peak {peak_mb:.0f} MB)")
    return avg, result


# ---------------------------------------------------------------------------
# Build test objects
# ---------------------------------------------------------------------------

def build_circuit_and_ham(N, rank, depth, device):
    circuit = Circuit(num_qubit=N, rank=rank, device=device)
    for _ in range(depth):
        for q in range(N):
            circuit.ry(q)
        for q in range(N - 1):
            circuit.cx(q, q + 1)

    ham = Hamiltonian(num_qubits=N)
    for i in range(N - 1):
        pauli = "I" * i + "ZZ" + "I" * (N - i - 2)
        ham.add_pauli(pauli, 1.0)
    for i in range(N):
        pauli = "I" * i + "Z" + "I" * (N - i - 1)
        ham.add_pauli(pauli, 0.5)

    return circuit, ham


# ---------------------------------------------------------------------------
# 1. Gate-level profiling
# ---------------------------------------------------------------------------

def profile_gates(circuit, device):
    print("\n" + "=" * 60)
    print("1. GATE-LEVEL PROFILING (single state)")
    print("=" * 60)

    P = circuit.params_size
    theta = torch.randn(P, device=device)

    # Single build
    timed(lambda: circuit.build_tensor(theta), label="build_tensor (single)")


def profile_gates_batch(circuit, device, batch_sizes):
    print("\n" + "=" * 60)
    print("2. BATCH BUILD PROFILING")
    print("=" * 60)

    P = circuit.params_size
    for B in batch_sizes:
        theta_batch = torch.randn(B, P, device=device)
        try:
            timed(
                lambda tb=theta_batch, b=B: circuit.build_tensor_batch(tb, b),
                label=f"build_tensor_batch B={B}",
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  {'build_tensor_batch B=' + str(B):40s} OOM")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 2. SVD profiling (the usual suspect)
# ---------------------------------------------------------------------------

def profile_svd(rank, device):
    print("\n" + "=" * 60)
    print("3. SVD MICROBENCHMARK")
    print("=" * 60)

    for dtype_label, dtype in [("cfloat", torch.cfloat), ("cdouble", torch.cdouble)]:
        M = torch.randn(2 * rank, 2 * rank, device=device, dtype=dtype)
        timed(
            lambda m=M: torch.linalg.svd(m, full_matrices=False),
            label=f"svd ({2*rank}x{2*rank}) {dtype_label}",
        )

    # Batched SVD
    for B in [4, 16, 48]:
        M = torch.randn(B, 2 * rank, 2 * rank, device=device, dtype=torch.cfloat)
        try:
            timed(
                lambda m=M: torch.linalg.svd(m, full_matrices=False),
                label=f"batched svd B={B} ({2*rank}x{2*rank})",
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  batched svd B={B} OOM")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 3. Measurement profiling
# ---------------------------------------------------------------------------

def profile_measurement(circuit, ham, device):
    print("\n" + "=" * 60)
    print("4. MEASUREMENT METHOD PROFILING")
    print("=" * 60)

    from TREV.measure.efficient_contraction import expectation_value_efficient_contraction
    from TREV.measure.contraction import expectation_value_contraction

    P = circuit.params_size
    theta = torch.randn(P, device=device)
    ring = circuit.build_tensor(theta)

    # Efficient contraction
    timed(
        lambda: expectation_value_efficient_contraction(ring, ham),
        label="efficient_contraction",
    )

    # Full contraction (only feasible for small N)
    if circuit.num_qubit <= 16:
        try:
            timed(
                lambda: expectation_value_contraction(ring, ham),
                label="full_contraction",
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            print(f"  {'full_contraction':40s} OOM/error")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 4. Gradient profiling
# ---------------------------------------------------------------------------

def profile_gradient(circuit, ham, device, batch_sizes):
    print("\n" + "=" * 60)
    print("5. GRADIENT COMPUTATION PROFILING")
    print("=" * 60)

    from TREV.optimization.gradients.batch_parameter_shift import BatchParameterShiftGradient

    P = circuit.params_size
    theta = torch.randn(P, device=device)

    for bs in batch_sizes:
        try:
            grad_obj = BatchParameterShiftGradient(
                shift=0.5, batch_size=bs, shots=0,
                measure_method=MeasureMethod.EFFICIENT_CONTRACTION,
                depth=1, is_partial=False,
            )
            grad_obj._autotuned = True

            timed(
                lambda: grad_obj.run(theta, circuit, ham),
                label=f"gradient batch_size={bs}",
                warmup=1, repeats=2,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"  {'gradient batch_size=' + str(bs):40s} OOM")
        except Exception as e:
            print(f"  {'gradient batch_size=' + str(bs):40s} Error: {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 5. Kernel launch overhead estimation
# ---------------------------------------------------------------------------

def profile_kernel_overhead(device):
    print("\n" + "=" * 60)
    print("6. KERNEL LAUNCH OVERHEAD")
    print("=" * 60)
    print("  (Measures Python->CUDA dispatch cost)")

    # Many tiny operations vs one big one
    a = torch.randn(64, 64, device=device)
    b = torch.randn(64, 64, device=device)

    N_ops = 1000

    def many_small():
        x = a
        for _ in range(N_ops):
            x = x @ b
        return x

    timed(many_small, label=f"{N_ops}x matmul (64x64)")

    # Compare with one big matmul
    A = torch.randn(1024, 1024, device=device)
    B = torch.randn(1024, 1024, device=device)
    timed(lambda: A @ B, label="1x matmul (1024x1024)")


# ---------------------------------------------------------------------------
# 6. torch.compile potential
# ---------------------------------------------------------------------------

def profile_compile(circuit, device):
    print("\n" + "=" * 60)
    print("7. TORCH.COMPILE POTENTIAL")
    print("=" * 60)

    P = circuit.params_size
    theta = torch.randn(P, device=device)

    # Eager
    timed(lambda: circuit.build_tensor(theta), label="build_tensor (eager)")

    # Try torch.compile
    try:
        compiled_build = torch.compile(circuit.build_tensor)
        # First call triggers compilation
        print("  Compiling... (this may take a minute)")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = compiled_build(theta)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        compile_time = time.perf_counter() - t0
        print(f"  {'compile time':40s} {compile_time:.2f}s")

        timed(lambda: compiled_build(theta), label="build_tensor (compiled)")
    except Exception as e:
        print(f"  torch.compile failed: {e}")


# ---------------------------------------------------------------------------
# 7. CUDA Graphs potential
# ---------------------------------------------------------------------------

def profile_cuda_graphs(circuit, device):
    if not torch.cuda.is_available():
        return

    print("\n" + "=" * 60)
    print("8. CUDA GRAPHS POTENTIAL")
    print("=" * 60)

    P = circuit.params_size
    theta = torch.randn(P, device=device)

    # Warmup
    _ = circuit.build_tensor(theta)
    torch.cuda.synchronize()

    try:
        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            result = circuit.build_tensor(theta)
        torch.cuda.synchronize()

        timed(lambda: g.replay(), label="build_tensor (CUDA Graph)")
    except Exception as e:
        print(f"  CUDA Graphs failed: {e}")
        print("  (Common: dynamic shapes, CPU ops, or data-dependent control flow)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile TREV bottlenecks")
    parser.add_argument("--n", type=int, default=8, help="Number of qubits")
    parser.add_argument("--rank", type=int, default=64, help="Bond dimension")
    parser.add_argument("--depth", type=int, default=3, help="Circuit depth")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GPU: {gpu_info()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Config: N={args.n}, rank={args.rank}, depth={args.depth}")
    print()

    circuit, ham = build_circuit_and_ham(args.n, args.rank, args.depth, device)
    P = circuit.params_size
    print(f"Circuit: {len(circuit.gates)} gates, {P} parameters")
    print(f"Hamiltonian: {len(ham.paulis)} terms")

    batch_sizes = [2, 8, 16, 48]

    # Run all profiles
    profile_gates(circuit, device)
    profile_gates_batch(circuit, device, batch_sizes)
    profile_svd(args.rank, device)
    profile_measurement(circuit, ham, device)
    profile_gradient(circuit, ham, device, [2, 8, 24])
    profile_kernel_overhead(device)
    profile_compile(circuit, device)
    profile_cuda_graphs(circuit, device)

    print("\n" + "=" * 60)
    print("DONE. Look for:")
    print("  - Which phase dominates (build vs measure vs gradient)?")
    print("  - Does torch.compile help? (kernel fusion for free)")
    print("  - Does CUDA Graphs work? (eliminates launch overhead)")
    print("  - How much time is in SVD vs matmul?")
    print("  - Is kernel launch overhead significant? (many small vs few big)")
    print("=" * 60)


if __name__ == "__main__":
    main()
