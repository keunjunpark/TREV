import random
from typing import Tuple, List, Optional

from qiskit.quantum_info import SparsePauliOp


def gengraph(n: int) -> List[Tuple[Tuple[int, int], int]]:
    NUM_NODES = n
    NUM_EDGES_PER_NODE = 3

    assert (NUM_EDGES_PER_NODE * NUM_NODES) % 2 == 0
    NUM_EDGES = NUM_EDGES_PER_NODE * NUM_NODES // 2

    node_set = [NUM_EDGES_PER_NODE for _ in range(NUM_NODES)]
    edges = []

    for i in range(NUM_NODES):
        if node_set[i] > 0:
            num_edges_to_add = node_set[i]
            node_set[i] = 0  # Mark this node as processed

            nodes_to_add = [j for j in range(NUM_NODES) if node_set[j] > 0 and j != i]
            random.shuffle(nodes_to_add)  # Shuffle to get random connections

            selected_nodes = nodes_to_add[:num_edges_to_add]  # Pick needed number of nodes

            for n in selected_nodes:
                weight = random.randint(1, 10)
                edges.append(((i, n), weight))
                node_set[n] -= 1

    assert len(edges) == NUM_EDGES, f"Expected {NUM_EDGES} edges, but got {len(edges)}"

    return edges

def create_hamiltonian(num_qubits: int, edge_weights: List[Tuple[Tuple[int, int], float]]):
    def _make_edge_tuple(edge: Optional[Tuple[int, int]] = None, /) -> Tuple[bool, ...]:
        list_form = [False] * num_qubits

        if edge is not None:
            n1, n2 = edge
            if 0 <= n1 < num_qubits:
                list_form[n1] = True
            if 0 <= n2 < num_qubits:
                list_form[n2] = True

        return tuple(list_form)

    total_edge_cost = sum(cost for _, cost in edge_weights)
    return [
        # +1 so energy is always > 0
        (total_edge_cost / 2.0 + 1.0, _make_edge_tuple()),
        *[(cost / 2.0, _make_edge_tuple(edge)) for edge, cost in edge_weights]
    ]

def make_hamiltonian(ham):
    """
    Constructs a Hamiltonian as a SparsePauliOp from the given list of terms.

    Args:
        ham (list): List of tuples (coefficient, tuple_of_bools) describing each Pauli term.
                    For example: [(coeff, (bool, bool, ..., bool)), ...]

    Returns:
        SparsePauliOp: The Hamiltonian represented as a SparsePauliOp.
    """
    pauli_strings = []
    coefficients = []

    for coeff, bool_list in ham:
        # Convert booleans to 'Z' or 'I'
        z_paulis = ['Z' if b else 'I' for b in bool_list]

        # Reverse the order to match Qiskit's qubit convention (like the second function)
        pauli_string = ''.join(z_paulis[::-1])

        pauli_strings.append(pauli_string)
        coefficients.append(coeff)

    # Create the SparsePauliOp from the list of (string, coeff)
    hamiltonian = SparsePauliOp.from_list(zip(pauli_strings, coefficients))
    return hamiltonian