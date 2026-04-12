"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer
for variational quantum circuits.

Population-based, derivative-free optimizer that maintains a multivariate
Gaussian and adapts its covariance matrix to learn parameter correlations.

Optimized for GPU:
  - Vectorized rank-mu covariance update (no Python loop)
  - Cached eigendecomposition of C (recomputed every `eigen_every` steps)
  - Avoids materializing diagonal matrices
  - Skips redundant tensor builds for best-bitstring tracking

Reference:
    Hansen, N. & Ostermeier, A. (2001). "Completely derandomized
    self-adaptation in evolution strategies." Evolutionary Computation 9(2).
"""

import math
import sys
import time
import gc

import torch

from ..circuit import Circuit
from ..hamiltonian.hamiltonian import Hamiltonian
from ..measure.enums import MeasureMethod
from ..measure.contraction import get_value_of_highest_probability, argmax_tr_noinv_BE, contract_tensor_ring
from ..measure.right_suffix_sampling import argmax_bitstring_tr_right_suffix


class CMAES:
    """CMA-ES optimizer state.

    Parameters
    ----------
    sigma : float
        Initial step size (standard deviation). Typical: 0.5-1.0.
    pop_size : int or None
        Population size per generation. None = default 4+floor(3*ln(n)).
    measure_method : MeasureMethod
        How to evaluate expectation values.
    shots : int
        Measurement shots per evaluation (0 = exact contraction).
    eigen_every : int
        Recompute eigendecomposition of C every this many generations.
        C changes gradually, so skipping saves O(n^3) per step.
    """

    def __init__(
        self,
        sigma: float = 0.5,
        pop_size: int | None = None,
        measure_method: MeasureMethod = MeasureMethod.RIGHT_SUFFIX_SAMPLING,
        shots: int = 10000,
        eigen_every: int = 1,
    ):
        self.sigma0 = sigma
        self.pop_size_override = pop_size
        self.measure_method = measure_method
        self.shots = shots
        self.eigen_every = eigen_every

    # ------------------------------------------------------------------ #
    #  Core CMA-ES logic (follows Hansen's tutorial notation)             #
    # ------------------------------------------------------------------ #

    def _init_state(self, n: int, device: torch.device):
        """Initialize all CMA-ES internal state variables."""
        lam = self.pop_size_override or (4 + int(3 * math.log(n)))
        mu = lam // 2  # number of parents

        # Recombination weights (log-linear)
        raw_w = torch.tensor(
            [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)],
            device=device, dtype=torch.float64,
        )
        weights = raw_w / raw_w.sum()

        mu_eff = 1.0 / (weights ** 2).sum().item()

        # Step-size adaptation
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        E_chi = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        # Covariance adaptation
        cc = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(
            1.0 - c1,
            2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff),
        )

        # State vectors / matrices
        mean = None  # set from theta
        p_sigma = torch.zeros(n, device=device, dtype=torch.float64)
        p_c = torch.zeros(n, device=device, dtype=torch.float64)
        C = torch.eye(n, device=device, dtype=torch.float64)
        sigma = self.sigma0

        # Cached eigen decomposition
        BD = torch.eye(n, device=device, dtype=torch.float64)  # B * D
        invsqrtC = torch.eye(n, device=device, dtype=torch.float64)

        return dict(
            n=n, lam=lam, mu=mu, weights=weights, mu_eff=mu_eff,
            c_sigma=c_sigma, d_sigma=d_sigma, E_chi=E_chi,
            cc=cc, c1=c1, c_mu=c_mu,
            mean=mean, p_sigma=p_sigma, p_c=p_c, C=C, sigma=sigma,
            BD=BD, invsqrtC=invsqrtC,
            device=device, gen_count=0,
        )

    def _update_eigen(self, s):
        """Eigendecompose C and cache BD and invsqrtC."""
        D2, B = torch.linalg.eigh(s['C'])
        D = torch.sqrt(torch.clamp(D2, min=1e-20))
        s['BD'] = B * D                        # (n, n) — avoids diag matrix
        s['invsqrtC'] = (B / D) @ B.T          # B @ diag(1/D) @ B^T without materializing diag

    def _step(self, s, evaluate_fn):
        """Run one CMA-ES generation.

        Parameters
        ----------
        s : dict
            CMA-ES state dict.
        evaluate_fn : callable
            Maps (B, n) float tensor -> (B,) float tensor of costs.

        Returns
        -------
        best_cost : float
            Best cost in this generation.
        best_params : torch.Tensor
            Parameters of the best individual, shape (n,).
        """
        n, lam, mu = s['n'], s['lam'], s['mu']
        device = s['device']

        # Recompute eigen decomposition periodically
        if s['gen_count'] % self.eigen_every == 0:
            self._update_eigen(s)
        s['gen_count'] += 1

        # Sample population: x_k = mean + sigma * BD @ z_k
        z = torch.randn(lam, n, device=device, dtype=torch.float64)
        y = z @ s['BD'].T                          # (lam, n)
        population = s['mean'] + s['sigma'] * y     # (lam, n)

        # Evaluate all individuals in one batched call
        costs = evaluate_fn(population.float())     # (lam,)

        # Sort by cost (minimization)
        order = torch.argsort(costs)
        y_sel = y[order[:mu]]                       # (mu, n) — selected steps
        best_params = population[order[0]]           # (n,)

        # Weighted recombination
        y_w = (s['weights'].unsqueeze(1) * y_sel).sum(dim=0)  # (n,)
        s['mean'] = s['mean'] + s['sigma'] * y_w

        # Step-size path
        s['p_sigma'] = (
            (1.0 - s['c_sigma']) * s['p_sigma']
            + math.sqrt(s['c_sigma'] * (2.0 - s['c_sigma']) * s['mu_eff'])
            * (s['invsqrtC'] @ y_w)
        )
        norm_ps = torch.linalg.norm(s['p_sigma']).item()
        s['sigma'] *= math.exp(
            (s['c_sigma'] / s['d_sigma']) * (norm_ps / s['E_chi'] - 1.0)
        )

        # Covariance path
        h_sigma = 1.0 if (
            norm_ps / math.sqrt(1.0 - (1.0 - s['c_sigma']) ** (2 * (s['gen_count'] + 1)))
            < (1.4 + 2.0 / (n + 1.0)) * s['E_chi']
        ) else 0.0

        s['p_c'] = (
            (1.0 - s['cc']) * s['p_c']
            + h_sigma * math.sqrt(s['cc'] * (2.0 - s['cc']) * s['mu_eff'])
            * y_w
        )

        # Covariance matrix update — vectorized rank-mu
        rank_one = s['p_c'].unsqueeze(1) @ s['p_c'].unsqueeze(0)
        # rank_mu = sum_i w_i * y_sel[i] @ y_sel[i]^T
        #         = y_sel^T @ diag(w) @ y_sel
        #         = (sqrt(w) * y_sel)^T @ (sqrt(w) * y_sel)
        w_y = s['weights'].sqrt().unsqueeze(1) * y_sel   # (mu, n)
        rank_mu = w_y.T @ w_y                             # (n, n)

        s['C'] = (
            (1.0 - s['c1'] - s['c_mu']) * s['C']
            + s['c1'] * rank_one
            + s['c_mu'] * rank_mu
        )

        return costs[order[0]].item(), best_params


def minimize_cma_es(
    circuit: Circuit,
    theta: torch.Tensor,
    hamiltonian: Hamiltonian,
    cma: CMAES,
    generations: int,
    best_value_method: str,
    wall_clock_cap: float | None = None,
    bitstring_every: int = 10,
):
    """CMA-ES optimization loop for variational quantum circuits.

    Parameters
    ----------
    bitstring_every : int
        Compute best bitstring every N generations (expensive).
        The last generation always computes it.

    Returns
    -------
    theta : torch.Tensor
        Optimized parameters.
    exp_values : list[float]
        Best expectation value per generation.
    best_result : list[str]
        Best bitstring per generation.
    iteration_times : list[float]
        Wall-clock time per generation.
    """
    device = circuit.device
    theta = theta.clone().to(device)
    n = theta.shape[0]

    s = cma._init_state(n, theta.device)
    s['mean'] = theta.double()

    exp_values = []
    best_result = []
    iteration_times = []
    start = time.time()
    last_bitstring = None

    def evaluate_fn(population: torch.Tensor) -> torch.Tensor:
        """Evaluate a population of parameter vectors."""
        pop = population.to(device)
        return circuit.get_expectation_value(
            pop, hamiltonian, cma.measure_method, cma.shots
        )

    for gen in range(generations):
        it_time = time.time()

        best_cost, best_params = cma._step(s, evaluate_fn)

        if device == 'cuda':
            torch.cuda.synchronize()
        iteration_times.append(time.time() - it_time)

        exp_values.append(best_cost)

        # Update theta to current mean
        theta = s['mean'].float()

        # Best bitstring — only compute every N gens (build_tensor is expensive)
        is_last = (gen == generations - 1)
        if gen % bitstring_every == 0 or is_last:
            with torch.no_grad():
                _tensor = circuit.build_tensor(best_params.float())

            if best_value_method == 'highest_probability':
                last_bitstring = get_value_of_highest_probability(_tensor, device)
            elif best_value_method == 'argmax_tr_noinv_BE':
                last_bitstring = argmax_bitstring_tr_right_suffix(_tensor)
            elif best_value_method == 'full_contraction':
                best_idx = contract_tensor_ring(_tensor).abs().pow(2).argmax().item()
                last_bitstring = format(best_idx, f'0{circuit.num_qubit}b')[::-1]
            else:
                last_bitstring = get_value_of_highest_probability(_tensor, device)

        best_result.append(last_bitstring)

        # Progress
        _progress_bar(gen, generations, start, best_cost)

        if gen % 10 == 0:
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()

        if wall_clock_cap is not None and (time.time() - start) >= wall_clock_cap:
            print(
                f"\n[INFO] Early stop at generation {gen} "
                f"(wall-clock cap {time.time() - start:.1f}s >= {wall_clock_cap:.1f}s)"
            )
            break

    return theta, exp_values, best_result, iteration_times


def _progress_bar(current, total, start_time, loss=None, bar_len=30):
    percent = float(current) / total
    arrow = '=' * int(round(percent * bar_len) - 1) + '>' if current < total else '=' * bar_len
    spaces = ' ' * (bar_len - len(arrow))
    elapsed = time.time() - start_time
    eta = (elapsed / current) * (total - current) if current > 0 else 0
    eta_str = time.strftime("%M:%S", time.gmtime(eta))
    metrics = f" | Loss: {loss:.4f}" if loss is not None else ""
    sys.stdout.write(f'\rCMA-ES: [{arrow}{spaces}] {int(percent * 100)}% | ETA: {eta_str}{metrics}')
    sys.stdout.flush()
