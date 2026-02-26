"""Circuit visualization as a qubit connectivity graph.

Usage:
    from TREV.circuit import Circuit
    from TREV.visualize import draw_connectivity

    c = Circuit(num_qubit=6, rank=4)
    # ... add gates ...
    fig = draw_connectivity(c)
    fig.savefig("circuit_graph.png")
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
