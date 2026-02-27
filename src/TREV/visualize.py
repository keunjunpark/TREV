"""Circuit visualization tools.

Usage:
    from TREV.circuit import Circuit
    from TREV.visualize import draw_connectivity, draw_circuit

    c = Circuit(num_qubit=6, rank=4)
    # ... add gates ...
    draw_connectivity(c)   # qubit graph (nodes + edges)
    draw_circuit(c)        # standard circuit diagram (wires + gates)
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from .circuit import Circuit


def draw_connectivity(
    circuit: "Circuit",
    *,
    layout: str = "circular",
    figsize: tuple[float, float] | None = None,
    node_size: int = 800,
    min_width: float = 1.0,
    max_width: float = 12.0,
    edge_color: str = "#4a90d9",
    node_color: str = "#ffffff",
    node_edge_color: str = "#333333",
    font_size: int = 12,
    show_edge_labels: bool = True,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Draw a qubit connectivity graph for the circuit.

    Each qubit is a node (circle).  Two-qubit gates become edges whose
    thickness scales with the number of gates between that pair.

    Parameters
    ----------
    circuit : Circuit
        A TREV Circuit instance (gates must already be added).
    layout : str
        Graph layout algorithm — "circular", "spring", "shell", or "kamada_kawai".
    figsize : tuple, optional
        Matplotlib figure size.  Auto-computed from qubit count if *None*.
    node_size : int
        Size of qubit nodes.
    min_width / max_width : float
        Edge width range.  Width is linearly interpolated between these based
        on gate count.
    edge_color : str
        Colour for edges.
    node_color / node_edge_color : str
        Fill and border colour for qubit nodes.
    font_size : int
        Font size for qubit labels.
    show_edge_labels : bool
        If *True*, annotate each edge with the gate count.
    title : str, optional
        Figure title.  Defaults to "Circuit Connectivity (N qubits)".
    ax : matplotlib Axes, optional
        Draw onto an existing Axes.  A new figure is created if *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .gates.parameter_gates import ParameterTwoQubitGate
    from .gates.non_parameter_gates import NonParameterTwoQubitsGate

    N = circuit.num_qubit

    # --- count 1-qubit and 2-qubit gates per qubit / pair ----------------
    single_count: dict[int, int] = defaultdict(int)
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    edge_types: dict[tuple[int, int], list[str]] = defaultdict(list)

    for gate in circuit.gates:
        if hasattr(gate, "qubits"):  # 2-qubit gate
            q0, q1 = gate.qubits
            key = (min(q0, q1), max(q0, q1))
            edge_count[key] += 1
            name = gate.matrix_fun.__name__
            if name not in edge_types[key]:
                edge_types[key].append(name)
        elif hasattr(gate, "qubit"):  # 1-qubit gate
            single_count[gate.qubit] += 1

    # --- build graph -----------------------------------------------------
    G = nx.Graph()
    for q in range(N):
        G.add_node(q)

    for (q0, q1), cnt in edge_count.items():
        G.add_edge(q0, q1, weight=cnt)

    # --- layout ----------------------------------------------------------
    layout_funcs = {
        "circular": nx.circular_layout,
        "spring": nx.spring_layout,
        "shell": nx.shell_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
    }
    pos = layout_funcs.get(layout, nx.circular_layout)(G)

    # --- figure ----------------------------------------------------------
    if ax is None:
        if figsize is None:
            side = max(5, 1.2 * N)
            figsize = (side, side)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # --- draw edges with variable width ----------------------------------
    if edge_count:
        counts = list(edge_count.values())
        lo, hi = min(counts), max(counts)

        for (q0, q1), cnt in edge_count.items():
            if hi == lo:
                w = (min_width + max_width) / 2
            else:
                w = min_width + (max_width - min_width) * (cnt - lo) / (hi - lo)
            nx.draw_networkx_edges(
                G, pos, edgelist=[(q0, q1)], width=w,
                edge_color=edge_color, alpha=0.7, ax=ax,
            )

    # --- draw nodes ------------------------------------------------------
    # Node size can optionally scale with single-qubit gate count
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_size,
        node_color=node_color,
        edgecolors=node_edge_color,
        linewidths=2,
        ax=ax,
    )

    # --- labels inside nodes: "q0", "q1", ... ----------------------------
    labels = {q: f"q{q}" for q in range(N)}
    nx.draw_networkx_labels(G, pos, labels, font_size=font_size, font_weight="bold", ax=ax)

    # --- edge labels (gate count) ----------------------------------------
    if show_edge_labels and edge_count:
        elabels = {}
        for (q0, q1), cnt in edge_count.items():
            types_str = ",".join(edge_types[(q0, q1)])
            elabels[(q0, q1)] = f"{cnt} ({types_str})"
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=elabels, font_size=font_size - 2, ax=ax,
        )

    # --- title & cleanup -------------------------------------------------
    if title is None:
        title = f"Circuit Connectivity ({N} qubits, {len(circuit.gates)} gates)"
    ax.set_title(title, fontsize=font_size + 2, pad=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ── gate display config ────────────────────────────────────────────────
# Map matrix_fun.__name__ → (label, color)
_GATE_STYLE = {
    "I":       ("I",       "#d0d0d0"),
    "H":       ("H",       "#f4d03f"),
    "X":       ("X",       "#e74c3c"),
    "Y":       ("Y",       "#2ecc71"),
    "Z":       ("Z",       "#3498db"),
    "RX":      ("RX",      "#e67e22"),
    "RY":      ("RY",      "#1abc9c"),
    "RZ":      ("RZ",      "#9b59b6"),
    "CNOT":    ("CX",      "#5dade2"),
    "SWAP":    ("SW",      "#af7ac5"),
    "ZZ":      ("ZZ",      "#f1948a"),
    "ZZ_SWAP": ("ZZ·SW",   "#eb984e"),
}


def _gate_label_color(gate) -> tuple[str, str]:
    name = gate.matrix_fun.__name__
    label, color = _GATE_STYLE.get(name, (name[:4], "#bdc3c7"))
    return label, color


def draw_circuit(
    circuit: "Circuit",
    *,
    figsize: tuple[float, float] | None = None,
    gate_width: float = 0.7,
    gate_height: float = 0.6,
    wire_color: str = "#555555",
    font_size: int = 9,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Draw a standard circuit diagram (horizontal qubit wires with gates).

    Parameters
    ----------
    circuit : Circuit
        A TREV Circuit instance.
    figsize : tuple, optional
        Figure size.  Auto-computed if *None*.
    gate_width / gate_height : float
        Size of single-qubit gate boxes.
    wire_color : str
        Colour of qubit wires.
    font_size : int
        Font size for gate labels.
    title : str, optional
        Figure title.
    ax : matplotlib Axes, optional
        Draw onto existing Axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.patches as mpatches

    N = circuit.num_qubit
    gates = circuit.gates

    # ── assign each gate a time slot (column) ───────────────────────
    # Greedy: each gate goes in the earliest slot where its qubits are free.
    slot_for_gate = []
    qubit_next_slot = [0] * N  # next free slot per qubit

    for gate in gates:
        if hasattr(gate, "qubits"):
            q0, q1 = gate.qubits
            t = max(qubit_next_slot[q0], qubit_next_slot[q1])
            slot_for_gate.append(t)
            qubit_next_slot[q0] = t + 1
            qubit_next_slot[q1] = t + 1
        else:
            q = gate.qubit
            t = qubit_next_slot[q]
            slot_for_gate.append(t)
            qubit_next_slot[q] = t + 1

    depth = max(slot_for_gate) + 1 if slot_for_gate else 1

    # ── figure setup ────────────────────────────────────────────────
    x_spacing = gate_width + 0.3
    y_spacing = 1.0
    margin_left = 1.2
    margin_right = 0.5

    total_w = margin_left + depth * x_spacing + margin_right
    total_h = (N - 1) * y_spacing + 2.0

    if ax is None:
        if figsize is None:
            figsize = (max(6, total_w), max(3, total_h))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # ── draw wires ──────────────────────────────────────────────────
    x_start = margin_left - 0.5
    x_end = margin_left + (depth - 0.5) * x_spacing + 0.5

    for q in range(N):
        y = -q * y_spacing
        ax.plot([x_start, x_end], [y, y], color=wire_color, linewidth=1.0, zorder=0)
        ax.text(
            x_start - 0.3, y, f"q{q}",
            ha="right", va="center", fontsize=font_size + 1, fontweight="bold",
        )

    # ── draw gates ──────────────────────────────────────────────────
    for i, gate in enumerate(gates):
        t = slot_for_gate[i]
        cx = margin_left + t * x_spacing
        label, color = _gate_label_color(gate)

        if hasattr(gate, "qubits"):
            # two-qubit gate
            q0, q1 = gate.qubits
            y0 = -q0 * y_spacing
            y1 = -q1 * y_spacing
            y_lo = min(y0, y1)
            y_hi = max(y0, y1)

            # vertical line connecting the two qubits
            ax.plot(
                [cx, cx], [y_lo, y_hi],
                color=color, linewidth=2.5, zorder=1,
            )

            # control dot for CNOT
            if gate.matrix_fun.__name__ == "CNOT":
                ax.plot(cx, y0, "o", color=color, markersize=8, zorder=3)
                ax.plot(cx, y1, "o", color=color, markersize=12, zorder=2,
                        markerfacecolor="white", markeredgewidth=2.0, markeredgecolor=color)
                # plus sign inside target
                r = 0.15
                ax.plot([cx - r, cx + r], [y1, y1], color=color, linewidth=1.5, zorder=3)
                ax.plot([cx, cx], [y1 - r, y1 + r], color=color, linewidth=1.5, zorder=3)
            else:
                # generic two-qubit: draw box spanning both qubits
                bh = y_hi - y_lo + gate_height
                rect = mpatches.FancyBboxPatch(
                    (cx - gate_width / 2, y_lo - gate_height / 2),
                    gate_width, bh,
                    boxstyle="round,pad=0.05",
                    facecolor=color, edgecolor="#333333", linewidth=1.2,
                    alpha=0.85, zorder=2,
                )
                ax.add_patch(rect)
                ax.text(
                    cx, (y0 + y1) / 2, label,
                    ha="center", va="center", fontsize=font_size - 1,
                    fontweight="bold", color="white", zorder=3,
                )
        else:
            # single-qubit gate
            q = gate.qubit
            y = -q * y_spacing
            rect = mpatches.FancyBboxPatch(
                (cx - gate_width / 2, y - gate_height / 2),
                gate_width, gate_height,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="#333333", linewidth=1.2,
                alpha=0.85, zorder=2,
            )
            ax.add_patch(rect)
            ax.text(
                cx, y, label,
                ha="center", va="center", fontsize=font_size,
                fontweight="bold", color="white", zorder=3,
            )

    # ── title & cleanup ─────────────────────────────────────────────
    if title is None:
        title = f"Circuit Diagram ({N} qubits, depth {depth})"
    ax.set_title(title, fontsize=font_size + 3, pad=10)
    ax.set_xlim(x_start - 0.8, x_end + 0.3)
    ax.set_ylim(-(N - 1) * y_spacing - 0.8, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig
