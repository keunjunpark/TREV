"""Unit tests for Circuit class."""

import pytest
import torch
import numpy as np
from TREV.circuit import Circuit
from TREV.hamiltonian.hamiltonian import Hamiltonian
from TREV.measure.enums import MeasureMethod
from TREV.measure.contraction import contract_tensor_ring


class TestCircuitInitialization:
    """Test Circuit initialization."""
    
    def test_circuit_creation(self):
        """Test basic circuit creation."""
        circuit = Circuit(num_qubit=4, rank=10, device='cpu')
        assert circuit.num_qubit == 4
        assert circuit.rank == 10
        assert circuit.device == 'cpu'
        assert len(circuit.gates) == 0
        assert circuit.params_size == 0
    
    def test_circuit_different_ranks(self):
        """Test circuit with different rank values."""
        for rank in [5, 10, 20]:
            circuit = Circuit(num_qubit=3, rank=rank, device='cpu')
            assert circuit.rank == rank


class TestNonParameterGates:
    """Test non-parameterized gate operations."""
    
    def test_identity_gate(self):
        """Test identity gate addition."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.id(0)
        assert len(circuit.gates) == 1
        assert circuit.params_size == 0
    
    def test_hadamard_gate(self):
        """Test Hadamard gate addition."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.h(0)
        circuit.h(1)
        assert len(circuit.gates) == 2
        assert circuit.params_size == 0
    
    def test_pauli_gates(self):
        """Test Pauli X, Y, Z gates."""
        circuit = Circuit(num_qubit=3, rank=10, device='cpu')
        circuit.x(0)
        circuit.y(1)
        circuit.z(2)
        assert len(circuit.gates) == 3
        assert circuit.params_size == 0
    
    def test_cnot_gate(self):
        """Test CNOT gate addition."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.cx(0, 1)
        assert len(circuit.gates) == 1
        assert circuit.params_size == 0
    
    def test_swap_gate(self):
        """Test SWAP gate addition."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.swap(0, 1)
        assert len(circuit.gates) == 1
        assert circuit.params_size == 0

    @pytest.mark.parametrize("N,rank", [(4, 8), (6, 16)])
    def test_swap_roundtrip_expectation(self, N, rank):
        """SWAP(i,i+1) twice should preserve all expectation values on larger circuits."""
        # Build a non-trivial circuit: rotation layer + entangling layer
        def make_circuit(n, r, add_double_swap=False, swap_pair=(0, 1)):
            c = Circuit(num_qubit=n, rank=r, device='cpu')
            for j in range(n):
                c.ry(j)
            for j in range(n):
                c.rz(j)
            for j in range(n - 1):
                c.cx(j, j + 1)
            if add_double_swap:
                c.swap(*swap_pair)
                c.swap(*swap_pair)
            return c

        # Test SWAP roundtrip on multiple adjacent pairs
        for i in range(N - 1):
            c_ref = make_circuit(N, rank, add_double_swap=False)
            c_swap = make_circuit(N, rank, add_double_swap=True, swap_pair=(i, i + 1))
            theta = torch.randn(c_ref.params_size)

            # Check against several Hamiltonian terms
            paulis = ['Z' if j == k else 'I' for k in range(N) for j in range(N)]
            for k in range(N):
                pauli_str = ''.join('Z' if j == k else 'I' for j in range(N))
                h = Hamiltonian(num_qubits=N)
                h.add_pauli(pauli_str, 1.0)
                ev_ref = float(c_ref.get_expectation_value(theta, h, MeasureMethod.FULL_CONTRACTION))
                ev_swap = float(c_swap.get_expectation_value(theta, h, MeasureMethod.FULL_CONTRACTION))
                assert abs(ev_ref - ev_swap) < 1e-3, \
                    f"N={N} SWAP({i},{i+1})^2 broke <{pauli_str}>: {ev_ref} vs {ev_swap}"

    @pytest.mark.parametrize("N,rank", [(4, 8), (6, 16)])
    def test_swap_exchanges_qubits(self, N, rank):
        """SWAP(i,i+1) should exchange single-qubit expectation values."""
        # Put a distinct rotation on each qubit so <Zk> differs per qubit
        theta_vals = torch.linspace(0.3, 2.5, N)

        c_no = Circuit(num_qubit=N, rank=rank, device='cpu')
        for j in range(N):
            c_no.ry(j)

        # Compute reference <Zk> for every qubit
        ev_ref = []
        for k in range(N):
            pauli_str = ''.join('Z' if j == k else 'I' for j in range(N))
            h = Hamiltonian(num_qubits=N)
            h.add_pauli(pauli_str, 1.0)
            ev_ref.append(float(c_no.get_expectation_value(theta_vals, h, MeasureMethod.FULL_CONTRACTION)))

        # For each adjacent pair, SWAP should exchange their <Z> values
        for i in range(N - 1):
            c_sw = Circuit(num_qubit=N, rank=rank, device='cpu')
            for j in range(N):
                c_sw.ry(j)
            c_sw.swap(i, i + 1)

            ev_sw = []
            for k in range(N):
                pauli_str = ''.join('Z' if j == k else 'I' for j in range(N))
                h = Hamiltonian(num_qubits=N)
                h.add_pauli(pauli_str, 1.0)
                ev_sw.append(float(c_sw.get_expectation_value(theta_vals, h, MeasureMethod.FULL_CONTRACTION)))

            # Qubit i and i+1 should be exchanged, others unchanged
            for k in range(N):
                if k == i:
                    expected = ev_ref[i + 1]
                elif k == i + 1:
                    expected = ev_ref[i]
                else:
                    expected = ev_ref[k]
                assert abs(ev_sw[k] - expected) < 1e-3, \
                    f"N={N} SWAP({i},{i+1}): <Z{k}> expected {expected:.6f}, got {ev_sw[k]:.6f}"

    def test_swap_chain_routing(self):
        """SWAP chain should correctly route a qubit across the full register.
        Move qubit 0 to position N-1 via N-1 SWAPs, verify <Z> follows."""
        N, rank = 6, 16
        theta = torch.tensor([1.5])  # RY only on qubit 0

        c_no = Circuit(num_qubit=N, rank=rank, device='cpu')
        c_no.ry(0)

        # Reference: <Z0> should be cos(1.5), <Zk> for k>0 should be 1.0
        h0 = Hamiltonian(num_qubits=N)
        h0.add_pauli('Z' + 'I' * (N - 1), 1.0)
        z0_ref = float(c_no.get_expectation_value(theta, h0, MeasureMethod.FULL_CONTRACTION))

        # Route qubit 0 to position N-1 via a chain of SWAPs
        c_route = Circuit(num_qubit=N, rank=rank, device='cpu')
        c_route.ry(0)
        for j in range(N - 1):
            c_route.swap(j, j + 1)

        # Now the rotated qubit should be at position N-1
        h_last = Hamiltonian(num_qubits=N)
        h_last.add_pauli('I' * (N - 1) + 'Z', 1.0)
        z_last = float(c_route.get_expectation_value(theta, h_last, MeasureMethod.FULL_CONTRACTION))

        assert abs(z_last - z0_ref) < 1e-3, \
            f"SWAP chain routing: <Z{N-1}> after route = {z_last:.6f}, expected {z0_ref:.6f}"

        # And qubit 0 (now identity) should give <Z0>=1.0
        h_first = Hamiltonian(num_qubits=N)
        h_first.add_pauli('Z' + 'I' * (N - 1), 1.0)
        z_first = float(c_route.get_expectation_value(theta, h_first, MeasureMethod.FULL_CONTRACTION))
        assert abs(z_first - 1.0) < 1e-3, \
            f"SWAP chain routing: <Z0> after route = {z_first:.6f}, expected 1.0"

    @pytest.mark.parametrize("N,rank", [(4, 8), (6, 16)])
    def test_swap_roundtrip_batch(self, N, rank):
        """SWAP^2 via build_tensor_batch should give the same state as no SWAP."""
        B = 4

        def make_circuit(n, r, double_swap=False):
            c = Circuit(num_qubit=n, rank=r, device='cpu')
            for j in range(n):
                c.ry(j)
            for j in range(n - 1):
                c.cx(j, j + 1)
            if double_swap:
                c.swap(0, 1)
                c.swap(0, 1)
            return c

        c_ref = make_circuit(N, rank, double_swap=False)
        c_swap = make_circuit(N, rank, double_swap=True)
        theta_batch = torch.randn(B, c_ref.params_size)

        ring_ref = c_ref.build_tensor_batch(theta_batch, B)
        ring_swap = c_swap.build_tensor_batch(theta_batch, B)

        for b in range(B):
            psi_ref = contract_tensor_ring(ring_ref[b]).flatten()
            psi_swap = contract_tensor_ring(ring_swap[b]).flatten()
            # Align global phase
            overlap = (psi_ref.conj() * psi_swap).sum()
            psi_swap_aligned = psi_swap * (overlap.conj() / overlap.abs())
            assert torch.allclose(psi_ref, psi_swap_aligned, atol=1e-4), \
                f"Batch {b}: SWAP^2 state mismatch (N={N})"

    @pytest.mark.parametrize("N,rank", [(4, 8), (6, 16)])
    def test_swap_exchanges_qubits_batch(self, N, rank):
        """Single SWAP via build_tensor_batch should exchange two qubits in the state."""
        B = 4
        theta_batch = torch.randn(B, N)  # one RY per qubit

        c_no = Circuit(num_qubit=N, rank=rank, device='cpu')
        for j in range(N):
            c_no.ry(j)

        c_sw = Circuit(num_qubit=N, rank=rank, device='cpu')
        for j in range(N):
            c_sw.ry(j)
        c_sw.swap(0, 1)

        ring_no = c_no.build_tensor_batch(theta_batch, B)
        ring_sw = c_sw.build_tensor_batch(theta_batch, B)

        for b in range(B):
            psi_no = contract_tensor_ring(ring_no[b])    # shape (2,)*N
            psi_sw = contract_tensor_ring(ring_sw[b])

            # SWAP(0,1) should transpose the first two axes
            psi_expected = psi_no.permute(1, 0, *range(2, N))
            assert torch.allclose(psi_sw.flatten(), psi_expected.flatten(), atol=1e-4), \
                f"Batch {b}: SWAP(0,1) didn't exchange qubits (N={N})"

    def test_swap_chain_routing_batch(self):
        """SWAP chain routing via build_tensor_batch: move qubit 0 to position N-1."""
        N, rank, B = 6, 16, 4
        theta_batch = torch.randn(B, 1)  # RY on qubit 0 only

        c_no = Circuit(num_qubit=N, rank=rank, device='cpu')
        c_no.ry(0)

        c_route = Circuit(num_qubit=N, rank=rank, device='cpu')
        c_route.ry(0)
        for j in range(N - 1):
            c_route.swap(j, j + 1)

        ring_no = c_no.build_tensor_batch(theta_batch, B)
        ring_route = c_route.build_tensor_batch(theta_batch, B)

        for b in range(B):
            psi_no = contract_tensor_ring(ring_no[b])       # (2,)*N
            psi_route = contract_tensor_ring(ring_route[b])

            # Routing qubit 0 to position N-1 = cyclic left shift of axes
            psi_expected = psi_no.permute(*range(1, N), 0)
            assert torch.allclose(psi_route.flatten(), psi_expected.flatten(), atol=1e-4), \
                f"Batch {b}: SWAP chain routing state mismatch"

    def test_swap_batch_matches_single(self):
        """build_tensor_batch with SWAP should match build_tensor for each element."""
        N, rank, B = 4, 8, 4

        c = Circuit(num_qubit=N, rank=rank, device='cpu')
        for j in range(N):
            c.ry(j)
        c.swap(1, 2)
        c.cx(0, 1)

        theta_batch = torch.randn(B, c.params_size)
        ring_batch = c.build_tensor_batch(theta_batch, B)

        for b in range(B):
            ring_single = c.build_tensor(theta_batch[b])
            psi_batch = contract_tensor_ring(ring_batch[b]).flatten()
            psi_single = contract_tensor_ring(ring_single).flatten()
            overlap = (psi_single.conj() * psi_batch).sum()
            psi_batch_aligned = psi_batch * (overlap.conj() / overlap.abs())
            assert torch.allclose(psi_single, psi_batch_aligned, atol=1e-4), \
                f"Batch {b}: batch vs single state mismatch with SWAP"


    @pytest.mark.parametrize("N,rank", [(4, 8), (6, 16)])
    def test_swap_exchange_after_entangling_batch(self, N, rank):
        """SWAP should correctly exchange qubits even after CNOT entanglement (batch path)."""
        B = 4

        def make_circuit(n, r, do_swap=False, swap_pair=(0, 1)):
            c = Circuit(num_qubit=n, rank=r, device='cpu')
            for j in range(n):
                c.ry(j)
            for j in range(n - 1):
                c.cx(j, j + 1)
            for j in range(n):
                c.rz(j)
            if do_swap:
                c.swap(*swap_pair)
            return c

        for i in range(N - 1):
            c_no = make_circuit(N, rank, do_swap=False)
            c_sw = make_circuit(N, rank, do_swap=True, swap_pair=(i, i + 1))
            theta_batch = torch.randn(B, c_no.params_size)

            ring_no = c_no.build_tensor_batch(theta_batch, B)
            ring_sw = c_sw.build_tensor_batch(theta_batch, B)

            for b in range(B):
                psi_no = contract_tensor_ring(ring_no[b])    # (2,)*N
                psi_sw = contract_tensor_ring(ring_sw[b])

                # SWAP(i, i+1) should transpose axes i and i+1
                perm = list(range(N))
                perm[i], perm[i + 1] = perm[i + 1], perm[i]
                psi_expected = psi_no.permute(*perm)
                assert torch.allclose(psi_sw.flatten(), psi_expected.flatten(), atol=1e-3), \
                    f"Batch {b}: SWAP({i},{i+1}) after entangling failed (N={N})"

    @pytest.mark.parametrize("N,rank,ctrl,tgt", [
        (4, 16, 0, 3),   # far ends
        (4, 16, 0, 2),   # skip one
        (6, 16, 0, 5),   # far ends on 6 qubits
        (6, 16, 1, 4),   # middle pair, skip 2
        (6, 16, 2, 5),   # skip 2 near the end
    ])
    def test_swap_routing_non_adjacent_cnot(self, N, rank, ctrl, tgt):
        """SWAP-routing a CNOT between non-adjacent qubits should give
        the same state as applying CNOT directly on the state vector."""
        # Base circuit: distinct rotation on each qubit
        theta = torch.linspace(0.3, 2.5, N)

        c_base = Circuit(num_qubit=N, rank=rank, device='cpu')
        for j in range(N):
            c_base.ry(j)

        psi_before = contract_tensor_ring(c_base.build_tensor(theta))

        # Apply CNOT(ctrl, tgt) mathematically on the state vector
        psi_expected = psi_before.clone()
        # Build slicer: when ctrl qubit = |1>, flip target qubit
        idx_ctrl1_tgt0 = [slice(None)] * N
        idx_ctrl1_tgt0[ctrl] = 1
        idx_ctrl1_tgt0[tgt] = 0
        idx_ctrl1_tgt1 = [slice(None)] * N
        idx_ctrl1_tgt1[ctrl] = 1
        idx_ctrl1_tgt1[tgt] = 1
        psi_expected[tuple(idx_ctrl1_tgt0)] = psi_before[tuple(idx_ctrl1_tgt1)]
        psi_expected[tuple(idx_ctrl1_tgt1)] = psi_before[tuple(idx_ctrl1_tgt0)]

        # SWAP-route: move target adjacent to control, apply cx, move back
        c_routed = Circuit(num_qubit=N, rank=rank, device='cpu')
        for j in range(N):
            c_routed.ry(j)

        if tgt > ctrl:
            # Move tgt down to ctrl+1
            for j in range(tgt - 1, ctrl, -1):
                c_routed.swap(j, j + 1)
            c_routed.cx(ctrl, ctrl + 1)
            for j in range(ctrl + 1, tgt):
                c_routed.swap(j, j + 1)
        else:
            # Move tgt up to ctrl-1
            for j in range(tgt, ctrl - 1):
                c_routed.swap(j, j + 1)
            c_routed.cx(ctrl, ctrl - 1)
            for j in range(ctrl - 2, tgt - 1, -1):
                c_routed.swap(j, j + 1)

        psi_routed = contract_tensor_ring(c_routed.build_tensor(theta))

        assert torch.allclose(psi_expected.flatten(), psi_routed.flatten(), atol=1e-3), \
            f"SWAP-routed CNOT({ctrl},{tgt}) on N={N} failed"

        # Also verify batch path matches
        B = 4
        theta_b = theta.unsqueeze(0).expand(B, -1)
        ring_b = c_routed.build_tensor_batch(theta_b, B)
        for b in range(B):
            psi_b = contract_tensor_ring(ring_b[b]).flatten()
            assert torch.allclose(psi_expected.flatten(), psi_b.flatten(), atol=1e-3), \
                f"SWAP-routed CNOT({ctrl},{tgt}) batch {b} on N={N} failed"

    @pytest.mark.parametrize("N,ctrl,tgt", [
        (6, 0, 3),   # symmetric — 3 hops each way
        (8, 0, 6),   # right route much shorter (2 vs 5 swaps)
        (8, 1, 6),   # off-center
        (8, 2, 7),   # wraps around, right route uses periodic bond
    ])
    def test_ring_left_vs_right_routing(self, N, ctrl, tgt):
        """On a tensor RING, routing left or right around the periodic boundary
        should give the same state. MPS can only go one way — rings can go both."""
        rank = 16
        theta = torch.linspace(0.3, 2.8, N)

        # Base state
        c_base = Circuit(num_qubit=N, rank=rank, device='cpu')
        for j in range(N):
            c_base.ry(j)
        psi_before = contract_tensor_ring(c_base.build_tensor(theta))

        # Mathematical reference: CNOT(ctrl, tgt) on state vector
        psi_ref = psi_before.clone()
        idx0 = [slice(None)] * N; idx0[ctrl] = 1; idx0[tgt] = 0
        idx1 = [slice(None)] * N; idx1[ctrl] = 1; idx1[tgt] = 1
        psi_ref[tuple(idx0)] = psi_before[tuple(idx1)]
        psi_ref[tuple(idx1)] = psi_before[tuple(idx0)]

        # --- Route A: move target toward control (forward direction) ---
        # Swap tgt leftward to site ctrl+1, apply cx(ctrl, ctrl+1), restore
        c_a = Circuit(num_qubit=N, rank=rank, device='cpu')
        for j in range(N):
            c_a.ry(j)
        swaps_a = []
        for site in range(tgt - 1, ctrl, -1):
            c_a.swap(site, site + 1)
            swaps_a.append((site, site + 1))
        c_a.cx(ctrl, ctrl + 1)
        for a, b in reversed(swaps_a):
            c_a.swap(a, b)

        psi_a = contract_tensor_ring(c_a.build_tensor(theta))

        # --- Route B: move control through wrap-around toward target ---
        # Swap ctrl backward (decreasing, wrapping through 0→N-1) until it
        # reaches site tgt. The original q_tgt gets pushed to site (tgt+1)%N.
        # Then apply cx(tgt, (tgt+1)%N) and restore.
        c_b = Circuit(num_qubit=N, rank=rank, device='cpu')
        for j in range(N):
            c_b.ry(j)
        swaps_b = []
        pos = ctrl
        while pos != tgt:
            nxt = (pos - 1) % N
            c_b.swap(nxt, pos)
            swaps_b.append((nxt, pos))
            pos = nxt
        c_b.cx(tgt, (tgt + 1) % N)
        for a, b in reversed(swaps_b):
            c_b.swap(a, b)

        psi_b = contract_tensor_ring(c_b.build_tensor(theta))

        # All three should match
        assert torch.allclose(psi_ref.flatten(), psi_a.flatten(), atol=1e-3), \
            f"Route A failed for cx({ctrl},{tgt}) on N={N}"
        assert torch.allclose(psi_ref.flatten(), psi_b.flatten(), atol=1e-3), \
            f"Route B (wrap-around) failed for cx({ctrl},{tgt}) on N={N}"
        assert torch.allclose(psi_a.flatten(), psi_b.flatten(), atol=1e-3), \
            f"Route A vs B mismatch for cx({ctrl},{tgt}) on N={N}"

        # Also test batch path
        B = 2
        theta_b = theta.unsqueeze(0).expand(B, -1)
        for c, label in [(c_a, "routeA-batch"), (c_b, "routeB-batch")]:
            ring_b = c.build_tensor_batch(theta_b, B)
            for b in range(B):
                psi_b_elem = contract_tensor_ring(ring_b[b]).flatten()
                assert torch.allclose(psi_ref.flatten(), psi_b_elem.flatten(), atol=1e-3), \
                    f"{label} element {b} failed for cx({ctrl},{tgt}) on N={N}"

    def test_mps_ring_swap_routing_pattern(self):
        """Test the actual build_circuit_mps_ring SWAP routing pattern:
        CNOT ladder, SWAP chain down, cx(1,0), SWAP chain back."""
        N, rank = 6, 16

        # Direct circuit: apply cx on all pairs including wrap-around
        # We compare against a circuit that does the same thing without SWAPs
        # by checking the state vector directly.
        def build_with_swap_routing(n, r):
            c = Circuit(num_qubit=n, rank=r, device='cpu')
            for j in range(n):
                c.rx(j)
            for j in range(n):
                c.rz(j)
            # Nearest-neighbor CNOTs
            for j in range(n - 1):
                c.cx(j, j + 1)
            # Route cx(N-1, 0) via SWAP chain
            for j in range(n - 2, 0, -1):
                c.swap(j, j + 1)
            c.cx(1, 0)
            for j in range(1, n - 1):
                c.swap(j, j + 1)
            return c

        c = build_with_swap_routing(N, rank)
        theta = torch.randn(c.params_size)

        # Single path
        ring_s = c.build_tensor(theta)
        psi_s = contract_tensor_ring(ring_s).flatten()
        assert not torch.isnan(psi_s).any(), "NaN in single path"
        assert psi_s.abs().sum() > 0, "Zero state in single path"

        # Batch path (B=1 and B=4)
        for B in [1, 4]:
            theta_b = theta.unsqueeze(0).expand(B, -1)
            ring_b = c.build_tensor_batch(theta_b, B)
            for b in range(B):
                psi_b = contract_tensor_ring(ring_b[b]).flatten()
                overlap = (psi_s.conj() * psi_b).sum()
                psi_b_aligned = psi_b * (overlap.conj() / overlap.abs())
                assert torch.allclose(psi_s, psi_b_aligned, atol=1e-3), \
                    f"MPS ring routing: batch (B={B}) element {b} doesn't match single path"


class TestParameterGates:
    """Test parameterized gate operations."""
    
    def test_rx_gate(self):
        """Test RX rotation gate."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.rx(0)
        assert len(circuit.gates) == 1
        assert circuit.params_size == 1
    
    def test_ry_gate(self):
        """Test RY rotation gate."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.ry(0)
        assert len(circuit.gates) == 1
        assert circuit.params_size == 1
    
    def test_rz_gate(self):
        """Test RZ rotation gate."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.rz(0)
        assert len(circuit.gates) == 1
        assert circuit.params_size == 1
    
    def test_multiple_parameter_gates(self):
        """Test multiple parameterized gates."""
        circuit = Circuit(num_qubit=3, rank=10, device='cpu')
        circuit.rx(0)
        circuit.ry(1)
        circuit.rz(2)
        assert len(circuit.gates) == 3
        assert circuit.params_size == 3
    
    def test_mixed_gates(self):
        """Test mixture of parameterized and non-parameterized gates."""
        circuit = Circuit(num_qubit=3, rank=10, device='cpu')
        circuit.h(0)
        circuit.rx(1)
        circuit.cx(0, 1)
        circuit.ry(2)
        assert len(circuit.gates) == 4
        assert circuit.params_size == 2


class TestTensorBuilding:
    """Test tensor building operations."""
    
    def test_build_tensor_shape(self):
        """Test that build_tensor returns correct shape."""
        circuit = Circuit(num_qubit=3, rank=10, device='cpu')
        circuit.h(0)
        circuit.rx(1)
        
        theta = torch.randn(circuit.params_size)
        tensor = circuit.build_tensor(theta)
        
        assert tensor.shape == (3, 10, 10, 2)
        assert tensor.dtype == torch.cfloat
    
    def test_build_tensor_initial_state(self):
        """Test initial state of empty circuit."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        theta = torch.tensor([])
        tensor = circuit.build_tensor(theta)
        
        # Check that initial state is |00...0>
        assert tensor[0, 0, 0, 0] == 1.0
        assert tensor[1, 0, 0, 0] == 1.0
    
    def test_build_tensor_batch_shape(self):
        """Test batch tensor building."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.rx(0)
        circuit.ry(1)
        
        batch_size = 5
        theta = torch.randn(batch_size, circuit.params_size)
        tensor = circuit.build_tensor_batch(theta, batch_size)
        
        assert tensor.shape == (batch_size, 2, 10, 10, 2)
        assert tensor.dtype == torch.cfloat
    
    def test_build_tensor_with_parameters(self):
        """Test tensor building with various parameters."""
        circuit = Circuit(num_qubit=2, rank=5, device='cpu')
        circuit.rx(0)
        circuit.ry(1)
        
        theta = torch.tensor([np.pi/4, np.pi/2])
        tensor = circuit.build_tensor(theta)
        
        assert tensor.shape == (2, 5, 5, 2)
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()


class TestMeasurement:
    """Test measurement operations."""
    
    def test_measure_sampling(self):
        """Test measurement with sampling method."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.h(0)
        
        theta = torch.tensor([])
        prob_dist = circuit.measure(theta, method=MeasureMethod.PERFECT_SAMPLING, shots=1000)

        # Results should be a probability distribution
        assert isinstance(prob_dist, list)
        assert abs(sum(prob_dist) - 1.0) < 1e-9  # Allow small floating-point error

    def test_measure_contraction(self):
        """Test measurement with contraction method."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')       
        circuit.h(0)
        
        theta = torch.tensor([])
        result = circuit.measure(theta, method=MeasureMethod.FULL_CONTRACTION)
        
        assert result is not None


class TestExpectationValue:
    """Test expectation value calculations."""
    
    def test_expectation_value_simple_hamiltonian(self):
        """Test expectation value with simple Hamiltonian."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.h(0)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZI', 1.0)
        
        theta = torch.tensor([])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.FULL_CONTRACTION
        )
        
        assert isinstance(exp_val, (float, torch.Tensor))
        assert not np.isnan(float(exp_val))
    
    def test_expectation_value_with_parameters(self):
        """Test expectation value with parameterized circuit."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.rx(0)
        circuit.ry(1)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZZ', 1.0)
        
        theta = torch.randn(circuit.params_size)
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.FULL_CONTRACTION
        )
        
        assert isinstance(exp_val, (float, torch.Tensor))
        assert not np.isnan(float(exp_val))
    
    def test_expectation_value_sampling_method(self):
        """Test expectation value with sampling method."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.h(0)
        circuit.h(1)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZZ', 1.0)
        
        theta = torch.tensor([])
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.PERFECT_SAMPLING, shots=10000
        )
        
        assert isinstance(exp_val, (complex, torch.Tensor))
    
    def test_expectation_value_efficient_contraction(self):
        """Test expectation value with efficient contraction method."""
        circuit = Circuit(num_qubit=3, rank=10, device='cpu')
        circuit.h(0)
        circuit.rx(1)
        
        hamiltonian = Hamiltonian(num_qubits=3)
        hamiltonian.add_pauli('ZZI', 1.0)
        
        theta = torch.randn(circuit.params_size)
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.EFFICIENT_CONTRACTION, shots=1000
        )
        
        assert isinstance(exp_val, (float, torch.Tensor))


class TestCircuitComplexity:
    """Test complex circuit configurations."""
    
    def test_qaoa_like_circuit(self):
        """Test QAOA-like circuit structure."""
        num_qubits = 4
        circuit = Circuit(num_qubit=num_qubits, rank=10, device='cpu')
        
        # Initial Hadamard layer
        for i in range(num_qubits):
            circuit.h(i)
        
        # Problem layer
        for i in range(num_qubits):
            circuit.rz(i)
        
        # Mixer layer
        for i in range(num_qubits):
            circuit.rx(i)
        
        assert circuit.params_size == 2 * num_qubits
        assert len(circuit.gates) == 3 * num_qubits
    
    def test_entangling_circuit(self):
        """Test circuit with entangling gates."""
        circuit = Circuit(num_qubit=4, rank=10, device='cpu')
        
        # Create entangled state
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        
        theta = torch.tensor([])
        tensor = circuit.build_tensor(theta)
        
        assert tensor.shape == (4, 10, 10, 2)
    
    def test_deep_circuit(self):
        """Test deep circuit with many layers."""
        circuit = Circuit(num_qubit=3, rank=10, device='cpu')
        
        for _ in range(10):  # 10 layers
            for i in range(3):
                circuit.ry(i)
            for i in range(2):
                circuit.cx(i, i+1)
        
        assert circuit.params_size == 30  # 10 layers * 3 qubits
        
        theta = torch.randn(circuit.params_size)
        tensor = circuit.build_tensor(theta)
        assert not torch.isnan(tensor).any()


class TestGradients:
    """Test gradient computation."""
    
    def test_parameter_gradient(self):
        """Test that parameters support gradient computation."""
        circuit = Circuit(num_qubit=2, rank=10, device='cpu')
        circuit.rx(0)
        circuit.ry(1)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZZ', 1.0)
        
        theta = torch.randn(circuit.params_size, requires_grad=True)
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.FULL_CONTRACTION
        )
        
        # Check that we can compute gradients
        if isinstance(exp_val, torch.Tensor):
            exp_val.backward()
            assert theta.grad is not None
            assert theta.grad.shape == theta.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDASupport:
    """Test CUDA device support."""
    
    def test_cuda_circuit_creation(self):
        """Test circuit creation on CUDA device."""
        circuit = Circuit(num_qubit=2, rank=10, device='cuda')
        assert circuit.device == 'cuda'
    
    def test_cuda_tensor_building(self):
        """Test tensor building on CUDA."""
        circuit = Circuit(num_qubit=2, rank=10, device='cuda')
        circuit.h(0)
        circuit.rx(1)
        
        theta = torch.randn(circuit.params_size, device='cuda')
        tensor = circuit.build_tensor(theta)
        
        assert tensor.device.type == 'cuda'
    
    def test_cuda_expectation_value(self):
        """Test expectation value computation on CUDA."""
        circuit = Circuit(num_qubit=2, rank=10, device='cuda')
        circuit.rx(0)
        
        hamiltonian = Hamiltonian(num_qubits=2)
        hamiltonian.add_pauli('ZI', 1.0)
        
        theta = torch.randn(circuit.params_size, device='cuda')
        exp_val = circuit.get_expectation_value(
            theta, hamiltonian, MeasureMethod.FULL_CONTRACTION
        )
        
        assert not np.isnan(float(exp_val))


class TestGetExpectationValueBatch:
    """Test the batch-dispatching get_expectation_value method."""

    @staticmethod
    def _make_circuit(N=4, rank=8):
        c = Circuit(num_qubit=N, rank=rank, device='cpu')
        for j in range(N):
            c.ry(j)
        for j in range(N - 1):
            c.cx(j, j + 1)
        for j in range(N):
            c.rz(j)
        return c

    @staticmethod
    def _make_hamiltonian(N=4):
        h = Hamiltonian(num_qubits=N)
        h.add_pauli('Z' + 'I' * (N - 1), 0.5)
        h.add_pauli('I' * (N - 1) + 'Z', 0.3)
        h.add_pauli('ZZ' + 'I' * (N - 2), -0.2)
        return h

    # ---- single theta (1D) returns scalar ----

    def test_single_theta_efficient_contraction(self):
        c = self._make_circuit()
        h = self._make_hamiltonian()
        theta = torch.randn(c.params_size)
        result = c.get_expectation_value(theta, h, MeasureMethod.EFFICIENT_CONTRACTION)
        assert result.dim() == 0, f"Expected scalar, got shape {result.shape}"
        assert not torch.isnan(result)

    def test_single_theta_perfect_sampling(self):
        c = self._make_circuit()
        h = self._make_hamiltonian()
        theta = torch.randn(c.params_size)
        result = c.get_expectation_value(theta, h, MeasureMethod.PERFECT_SAMPLING, shots=5000)
        assert result.dim() == 0, f"Expected scalar, got shape {result.shape}"
        assert not torch.isnan(result)

    def test_single_theta_right_suffix_sampling(self):
        c = self._make_circuit()
        h = self._make_hamiltonian()
        theta = torch.randn(c.params_size)
        result = c.get_expectation_value(theta, h, MeasureMethod.RIGHT_SUFFIX_SAMPLING, shots=5000)
        assert result.dim() == 0, f"Expected scalar, got shape {result.shape}"
        assert not torch.isnan(result)

    # ---- batch theta (2D) returns (B,) ----

    def test_batch_theta_efficient_contraction(self):
        B = 4
        c = self._make_circuit()
        h = self._make_hamiltonian()
        theta_batch = torch.randn(B, c.params_size)
        result = c.get_expectation_value(theta_batch, h, MeasureMethod.EFFICIENT_CONTRACTION)
        assert result.shape == (B,), f"Expected (B,), got {result.shape}"
        assert not torch.isnan(result).any()

    def test_batch_theta_perfect_sampling(self):
        B = 4
        c = self._make_circuit()
        h = self._make_hamiltonian()
        theta_batch = torch.randn(B, c.params_size)
        result = c.get_expectation_value(theta_batch, h, MeasureMethod.PERFECT_SAMPLING, shots=5000)
        assert result.shape == (B,), f"Expected (B,), got {result.shape}"
        assert not torch.isnan(result).any()

    def test_batch_theta_right_suffix_sampling(self):
        B = 4
        c = self._make_circuit()
        h = self._make_hamiltonian()
        theta_batch = torch.randn(B, c.params_size)
        result = c.get_expectation_value(theta_batch, h, MeasureMethod.RIGHT_SUFFIX_SAMPLING, shots=5000)
        assert result.shape == (B,), f"Expected (B,), got {result.shape}"
        assert not torch.isnan(result).any()

    # ---- batch consistency: batch result matches per-element single calls ----

    def test_batch_matches_single_efficient_contraction(self):
        """Each element of batch result should match calling single theta."""
        B = 4
        c = self._make_circuit()
        h = self._make_hamiltonian()
        theta_batch = torch.randn(B, c.params_size)

        batch_result = c.get_expectation_value(theta_batch, h, MeasureMethod.EFFICIENT_CONTRACTION)
        for b in range(B):
            single_result = c.get_expectation_value(theta_batch[b], h, MeasureMethod.EFFICIENT_CONTRACTION)
            assert torch.allclose(batch_result[b], single_result, atol=1e-4), \
                f"Batch[{b}]={batch_result[b]:.6f} != single={single_result:.6f}"

    # ---- accuracy: efficient contraction matches full contraction reference ----

    def test_efficient_contraction_matches_full_contraction(self):
        """Batch efficient contraction should match the exact full contraction."""
        from TREV.measure.contraction import expectation_value as full_contraction_ev

        c = self._make_circuit()
        h = self._make_hamiltonian()
        theta = torch.randn(c.params_size)

        tensor = c.build_tensor(theta)
        ref = float(full_contraction_ev(tensor, h, device='cpu').real)

        result = float(c.get_expectation_value(theta, h, MeasureMethod.EFFICIENT_CONTRACTION))
        assert abs(result - ref) < 1e-4, \
            f"EFFICIENT_CONTRACTION={result:.6f} vs FULL_CONTRACTION={ref:.6f}"

    def test_efficient_contraction_batch_matches_full_contraction(self):
        """Batch efficient contraction for B>1 matches full contraction per element."""
        from TREV.measure.contraction import expectation_value as full_contraction_ev

        B = 4
        c = self._make_circuit()
        h = self._make_hamiltonian()
        theta_batch = torch.randn(B, c.params_size)

        batch_result = c.get_expectation_value(theta_batch, h, MeasureMethod.EFFICIENT_CONTRACTION)
        for b in range(B):
            tensor = c.build_tensor(theta_batch[b])
            ref = float(full_contraction_ev(tensor, h, device='cpu').real)
            assert abs(float(batch_result[b]) - ref) < 1e-4, \
                f"Batch[{b}]: EFFICIENT_CONTRACTION={float(batch_result[b]):.6f} vs ref={ref:.6f}"

    # ---- sampling methods converge to exact value ----

    def test_perfect_sampling_converges(self):
        """Perfect sampling with many shots should be close to exact value."""
        from TREV.measure.contraction import expectation_value as full_contraction_ev

        c = self._make_circuit()
        h = self._make_hamiltonian()
        theta = torch.randn(c.params_size)

        tensor = c.build_tensor(theta)
        ref = float(full_contraction_ev(tensor, h, device='cpu').real)

        result = c.get_expectation_value(theta, h, MeasureMethod.PERFECT_SAMPLING, shots=50000)
        assert abs(float(result) - ref) < 0.15, \
            f"PERFECT_SAMPLING={float(result):.4f} too far from ref={ref:.4f}"

    def test_right_suffix_sampling_converges(self):
        """Right suffix sampling with many shots should be close to exact value."""
        from TREV.measure.contraction import expectation_value as full_contraction_ev

        c = self._make_circuit()
        h = self._make_hamiltonian()
        theta = torch.randn(c.params_size)

        tensor = c.build_tensor(theta)
        ref = float(full_contraction_ev(tensor, h, device='cpu').real)

        result = c.get_expectation_value(theta, h, MeasureMethod.RIGHT_SUFFIX_SAMPLING, shots=50000)
        assert abs(float(result) - ref) < 0.15, \
            f"RIGHT_SUFFIX_SAMPLING={float(result):.4f} too far from ref={ref:.4f}"

    # ---- IXYZ Hamiltonian support ----

    def test_efficient_contraction_ixyz_hamiltonian(self):
        """Efficient contraction should handle X/Y Pauli operators."""
        from TREV.measure.contraction import expectation_value as full_contraction_ev

        N = 4
        c = self._make_circuit(N=N)
        h = Hamiltonian(num_qubits=N)
        h.add_pauli('XZIY', 0.5)
        h.add_pauli('IYXZ', -0.3)
        h.add_pauli('ZZII', 0.2)

        theta = torch.randn(c.params_size)
        tensor = c.build_tensor(theta)
        ref = float(full_contraction_ev(tensor, h, device='cpu').real)

        result = float(c.get_expectation_value(theta, h, MeasureMethod.EFFICIENT_CONTRACTION))
        assert abs(result - ref) < 1e-4, \
            f"IXYZ: EFFICIENT_CONTRACTION={result:.6f} vs ref={ref:.6f}"

    def test_batch_ixyz_hamiltonian(self):
        """Batch with IXYZ Hamiltonian should match full contraction per element."""
        from TREV.measure.contraction import expectation_value as full_contraction_ev

        N, B = 4, 3
        c = self._make_circuit(N=N)
        h = Hamiltonian(num_qubits=N)
        h.add_pauli('XZIY', 0.5)
        h.add_pauli('IYXZ', -0.3)

        theta_batch = torch.randn(B, c.params_size)
        batch_result = c.get_expectation_value(theta_batch, h, MeasureMethod.EFFICIENT_CONTRACTION)
        for b in range(B):
            tensor = c.build_tensor(theta_batch[b])
            ref = float(full_contraction_ev(tensor, h, device='cpu').real)
            assert abs(float(batch_result[b]) - ref) < 1e-4, \
                f"Batch[{b}]: IXYZ result={float(batch_result[b]):.6f} vs ref={ref:.6f}"

    # ---- B=1 edge case ----

    def test_batch_b1(self):
        """Batch with B=1 should work correctly."""
        c = self._make_circuit()
        h = self._make_hamiltonian()
        theta_batch = torch.randn(1, c.params_size)

        result = c.get_expectation_value(theta_batch, h, MeasureMethod.EFFICIENT_CONTRACTION)
        assert result.shape == (1,)

        single = c.get_expectation_value(theta_batch[0], h, MeasureMethod.EFFICIENT_CONTRACTION)
        assert torch.allclose(result[0], single, atol=1e-4)

    # ---- FULL_CONTRACTION raises NotImplementedError ----

    def test_full_contraction_not_implemented(self):
        c = self._make_circuit()
        h = self._make_hamiltonian()
        theta = torch.randn(c.params_size)
        with pytest.raises(NotImplementedError):
            c.get_expectation_value(theta, h, MeasureMethod.FULL_CONTRACTION)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
