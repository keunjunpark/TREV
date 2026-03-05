"""Generic circuit transpiler for routing non-adjacent 2-qubit gates.

Supports ring (tensor ring) and chain (MPS) topologies.
Processes gates in order, inserting SWAPs to make all 2-qubit gates adjacent,
then appends bubble-sort SWAPs to restore the identity permutation.
"""
from typing import List, Optional, Tuple, Literal

import torch

from .circuit import Circuit
from .gates.non_parameter_gates import (
    NonParameterOneQubitGate,
    NonParameterTwoQubitsGate,
)
from .gates.parameter_gates import (
    ParameterOneQubitGate,
    ParameterMultiOneQubitGate,
    ParameterTwoQubitGate,
)
from .gates.info import (
    I as I_fun, H as H_fun, X as X_fun, Y as Y_fun, Z as Z_fun,
    RX as RX_fun, RY as RY_fun, RZ as RZ_fun, U3 as U3_fun,
    CNOT as CNOT_fun, SWAP as SWAP_fun, ZZ as ZZ_fun, ZZ_SWAP as ZZ_SWAP_fun,
)


# ── topology helpers ──────────────────────────────────────────────────────


def _ring_adjacent(p0: int, p1: int, N: int) -> bool:
    d = abs(p0 - p1)
    return d == 1 or d == N - 1


def _chain_adjacent(p0: int, p1: int, N: int) -> bool:
    return abs(p0 - p1) == 1


def _ring_route(src: int, dst: int, N: int) -> List[Tuple[int, int]]:
    """Swap path to move *src* adjacent to *dst* on a ring of N nodes."""
    fwd = (dst - src) % N
    bwd = (src - dst) % N
    path: List[Tuple[int, int]] = []
    pos = src
    if fwd <= bwd:
        while (pos + 1) % N != dst:
            nxt = (pos + 1) % N
            path.append((pos, nxt))
            pos = nxt
    else:
        while (pos - 1 + N) % N != dst:
            nxt = (pos - 1 + N) % N
            path.append((nxt, pos))
            pos = nxt
    return path


def _chain_route(src: int, dst: int, N: int) -> List[Tuple[int, int]]:
    """Swap path to move *src* adjacent to *dst* on a chain of N nodes."""
    path: List[Tuple[int, int]] = []
    pos = src
    step = 1 if dst > src else -1
    while abs(pos - dst) > 1:
        nxt = pos + step
        path.append((min(pos, nxt), max(pos, nxt)))
        pos = nxt
    return path


def _swap_perm(perm: List[int], a: int, b: int):
    """Update logical-to-physical permutation after swapping physical sites *a*, *b*."""
    la = perm.index(a)
    lb = perm.index(b)
    perm[la], perm[lb] = perm[lb], perm[la]


def _unroute_bubble(out: Circuit, perm: List[int], N: int):
    """Bubble-sort SWAPs to restore identity permutation.

    Uses only (pos, pos+1) adjacent swaps, which are valid bonds
    on both ring and chain topologies.
    """
    inv = [0] * N
    for logical, physical in enumerate(perm):
        inv[physical] = logical
    changed = True
    while changed:
        changed = False
        for pos in range(N - 1):
            if inv[pos] > inv[pos + 1]:
                out.swap(pos, pos + 1)
                inv[pos], inv[pos + 1] = inv[pos + 1], inv[pos]
                changed = True


# ── gate emission ─────────────────────────────────────────────────────────


def _emit_1q(out: Circuit, gate, phys: int):
    """Emit a copy of a 1-qubit gate at physical qubit *phys*."""
    mf = gate.matrix_fun
    if   mf is I_fun:  out.id(phys)
    elif mf is H_fun:  out.h(phys)
    elif mf is X_fun:  out.x(phys)
    elif mf is Y_fun:  out.y(phys)
    elif mf is Z_fun:  out.z(phys)
    elif mf is RX_fun: out.rx(phys)
    elif mf is RY_fun: out.ry(phys)
    elif mf is RZ_fun: out.rz(phys)
    elif mf is U3_fun: out.u3(phys)
    else:
        raise ValueError(f"Unknown 1q gate: {mf.__name__}")


# ── main entry point ─────────────────────────────────────────────────────


def transpile(
    circuit: Circuit,
    topology: Literal['ring', 'chain'] = 'ring',
    fuse_zz_swap: bool = False,
) -> Circuit:
    """Transpile a circuit so all 2-qubit gates act on adjacent qubits.

    Parameters
    ----------
    circuit : Circuit
        Source circuit (may contain non-adjacent 2-qubit gates).
    topology : 'ring' or 'chain'
        Target qubit connectivity.
        Ring: bonds ``(i, i+1)`` for ``i=0..N-2`` and ``(N-1, 0)``.
        Chain: bonds ``(i, i+1)`` for ``i=0..N-2`` only.
    fuse_zz_swap : bool
        If True, every ZZ gate is replaced with ZZ_SWAP (same parameter,
        one SVD) and the implicit SWAP updates the permutation.

    Returns
    -------
    Circuit
        New circuit with identical ``params_size``.
        The same theta vector works for both the original and transpiled circuits.
    """
    N = circuit.num_qubit
    out = Circuit(N, rank=circuit.rank, device=circuit.device)

    perm = list(range(N))  # perm[logical] = physical
    adj_fn = _ring_adjacent if topology == 'ring' else _chain_adjacent
    route_fn = _ring_route if topology == 'ring' else _chain_route

    for gate in circuit.gates:
        # ── 1-qubit gates ──
        if isinstance(gate, (NonParameterOneQubitGate, ParameterOneQubitGate,
                             ParameterMultiOneQubitGate)):
            _emit_1q(out, gate, perm[gate.qubit])
            continue

        # ── 2-qubit gates ──
        q0, q1 = gate.qubits          # logical qubits
        p0, p1 = perm[q0], perm[q1]   # current physical positions

        # route if not adjacent
        if not adj_fn(p0, p1, N):
            for a, b in route_fn(p0, p1, N):
                out.swap(a, b)
                _swap_perm(perm, a, b)
            p0, p1 = perm[q0], perm[q1]

        # emit the gate
        mf = gate.matrix_fun

        if fuse_zz_swap and mf is ZZ_fun:
            # replace ZZ with ZZ_SWAP
            out.zz_swap(p0, p1)
            _swap_perm(perm, p0, p1)
        elif mf is ZZ_SWAP_fun:
            out.zz_swap(p0, p1)
            _swap_perm(perm, p0, p1)
        elif mf is SWAP_fun:
            out.swap(p0, p1)
            _swap_perm(perm, p0, p1)
        elif mf is ZZ_fun:
            out.zz(p0, p1)
        elif mf is CNOT_fun:
            out.cx(p0, p1)
        else:
            raise ValueError(f"Unknown 2q gate: {mf.__name__}")

    # restore identity permutation for correct measurement
    _unroute_bubble(out, perm, N)

    return out


# ── Qiskit converter ────────────────────────────────────────────────────


# Gate name → (TREV method name, number of params, is_2q)
_QISKIT_1Q_MAP = {
    'h':    'h',
    'x':    'x',
    'y':    'y',
    'z':    'z',
    'id':   'id',
    'i':    'id',
    'rx':   'rx',
    'ry':   'ry',
    'rz':   'rz',
}

_QISKIT_1Q_PARAM_COUNT = {
    'rx': 1, 'ry': 1, 'rz': 1,
}

_QISKIT_U3_NAMES = {'u', 'u3'}


def _fuse_zzswap_pass(gate_list):
    """Fuse ``rzz(a,b) + swap(a,b)`` pairs into ``zz_swap(a,b)``.

    Scans backward from each swap, commuting past single-qubit gates
    on ``{a, b}`` (flipping ``a<->b`` after fusion) and disjoint gates.
    Stops at any 2-qubit gate touching ``{a, b}``; fuses if it is
    ``rzz`` on the same qubits.
    """
    result = list(gate_list)
    i = len(result) - 1
    while i >= 0:
        name, qargs, params = result[i]
        if name == 'swap':
            a, b = qargs[0], qargs[1]
            target = {a, b}
            j = i - 1
            to_flip = []
            found_rzz = False
            while j >= 0:
                jname, jqargs, jparams = result[j]
                jset = set(jqargs)
                if len(jqargs) == 1:
                    if jset <= target:
                        to_flip.append(j)
                        j -= 1
                    elif jset.isdisjoint(target):
                        j -= 1
                    else:
                        break
                else:  # multi-qubit gate
                    if jset.isdisjoint(target):
                        j -= 1
                    elif jname == 'rzz' and jset == target:
                        found_rzz = True
                        break
                    else:
                        break
            if found_rzz:
                # Replace rzz with zz_swap
                result[j] = ('zz_swap', result[j][1], result[j][2])
                # Flip 1q gates between rzz and swap that act on {a,b}
                for fi in to_flip:
                    fn, fq, fp = result[fi]
                    q = fq[0]
                    result[fi] = (fn, [b if q == a else a], fp)
                # Remove the swap gate
                del result[i]
        i -= 1
    return result


def from_qiskit(
    qc,
    fuse_zz_swap: bool = False,
    rank: int = 10,
    device: str = 'cpu',
) -> Tuple['Circuit', Optional[torch.Tensor]]:
    """Convert a Qiskit ``QuantumCircuit`` into a TREV :class:`Circuit`.

    The Qiskit circuit is assumed to be already routed (physical qubit
    indices).  Gates are emitted at their Qiskit qubit indices directly.

    Parameters
    ----------
    qc : qiskit.circuit.QuantumCircuit
        Source Qiskit circuit (typically already transpiled/routed).
    fuse_zz_swap : bool
        If True, adjacent ``rzz + swap`` pairs on the same qubits are
        fused into a single ``zz_swap`` gate (reduces gate count without
        changing the qubit permutation).
    rank : int
        Bond dimension for the TREV circuit.
    device : str
        Torch device for the TREV circuit.

    Returns
    -------
    (Circuit, Tensor | None)
        The TREV circuit and a tensor of collected parameter values (or None
        if any parameter is symbolic / unbound).
    """
    N = qc.num_qubits
    out = Circuit(N, rank=rank, device=device)

    param_values: List[float] = []
    all_numeric = True

    # Collect gates as (name, qargs, gate_params) tuples
    raw_gates: List[Tuple[str, List[int], List[float]]] = []
    for instruction in qc.data:
        op = instruction.operation
        name = op.name.lower()
        qubits = [qc.find_bit(q).index for q in instruction.qubits]

        gate_params: List[float] = []
        for p in op.params:
            try:
                gate_params.append(float(p))
            except (TypeError, ValueError):
                all_numeric = False

        raw_gates.append((name, qubits, gate_params))

    # Fuse rzz+swap pairs if requested
    if fuse_zz_swap:
        raw_gates = _fuse_zzswap_pass(raw_gates)

    # Emit gates to TREV circuit
    for name, qubits, gate_params in raw_gates:
        # 1-qubit gates
        if name in _QISKIT_1Q_MAP:
            method = getattr(out, _QISKIT_1Q_MAP[name])
            method(qubits[0])
            n_params = _QISKIT_1Q_PARAM_COUNT.get(name, 0)
            param_values.extend(gate_params[:n_params])
            continue

        if name in _QISKIT_U3_NAMES:
            out.u3(qubits[0])
            param_values.extend(gate_params[:3])
            continue

        # 2-qubit gates
        q0, q1 = qubits[0], qubits[1]

        if name == 'rzz':
            out.zz(q0, q1)
            param_values.extend(gate_params[:1])
            continue

        if name == 'zz_swap':
            out.zz_swap(q0, q1)
            param_values.extend(gate_params[:1])
            continue

        if name == 'swap':
            out.swap(q0, q1)
            continue

        if name in ('cx', 'cnot'):
            out.cx(q0, q1)
            continue

        raise ValueError(f"Unsupported Qiskit gate: {name}")

    # Save Qiskit layout permutation (logical → physical) if available
    if hasattr(qc, 'layout') and qc.layout is not None:
        try:
            qubit_perm = qc.layout.final_index_layout(filter_ancillas=True)
            if qubit_perm != list(range(N)):
                out.qubit_perm = qubit_perm
        except Exception:
            pass

    theta = torch.tensor(param_values, dtype=torch.float32) if all_numeric else None
    return out, theta


def build_parameter_mapping(
    qc,
    fuse_zz_swap: bool = False,
    rank: int = 10,
    device: str = 'cpu',
) -> Tuple['Circuit', torch.Tensor, torch.Tensor, List[str]]:
    """Build linear mapping from Qiskit logical params to TREV theta.

    The parameter mapping is linear::

        full_theta = param_base + jacobian @ logical_params

    Works for any parameterized circuit (QAOA, UCCSD, hardware-efficient
    ansatz, etc.).  The Jacobian is computed numerically by probing
    :func:`from_qiskit` with zero and unit vectors.

    Parameters
    ----------
    qc : qiskit.circuit.QuantumCircuit
        Parameterized Qiskit circuit (already routed, NOT bound).
    fuse_zz_swap : bool
        Whether to fuse ``rzz + swap`` into ``zz_swap``.
    rank : int
        TREV bond dimension.
    device : str
        Torch device for the TREV circuit.

    Returns
    -------
    (circuit, param_base, jacobian, param_names)
        circuit : Circuit — TREV circuit (structure only).
        param_base : Tensor (P,) — TREV theta when all logical params = 0.
        jacobian : Tensor (P, K) — maps K logical params to P TREV params.
        param_names : list of K parameter name strings.
    """
    params = sorted(qc.parameters, key=lambda p: p.name)
    param_names = [p.name for p in params]
    K = len(params)

    # Bind all params to zero → get theta_base
    zero_bind = {p: 0.0 for p in params}
    qc_zero = qc.assign_parameters(zero_bind)
    circuit, param_base = from_qiskit(
        qc_zero, fuse_zz_swap=fuse_zz_swap, rank=rank, device=device,
    )
    P = param_base.shape[0]

    # Compute Jacobian columns by probing with unit vectors
    jacobian = torch.zeros(P, K, dtype=param_base.dtype)
    for i, param in enumerate(params):
        unit_bind = {p: 0.0 for p in params}
        unit_bind[param] = 1.0
        qc_unit = qc.assign_parameters(unit_bind)
        _, theta_unit = from_qiskit(
            qc_unit, fuse_zz_swap=fuse_zz_swap, rank=rank, device=device,
        )
        jacobian[:, i] = (theta_unit - param_base).cpu()

    return circuit, param_base.cpu(), jacobian, param_names
