"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer
for variational quantum circuits.

Population-based, derivative-free optimizer that maintains a multivariate
Gaussian and adapts its covariance matrix to learn parameter correlations.

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
    """

    def __init__(
        self,
        sigma: float = 0.5,
        pop_size: int | None = None,
        measure_method: MeasureMethod = MeasureMethod.RIGHT_SUFFIX_SAMPLING,
        shots: int = 10000,
    ):
        self.sigma0 = sigma
        self.pop_size_override = pop_size
        self.measure_method = measure_method
        self.shots = shots

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

        return dict(
            n=n, lam=lam, mu=mu, weights=weights, mu_eff=mu_eff,
            c_sigma=c_sigma, d_sigma=d_sigma, E_chi=E_chi,
            cc=cc, c1=c1, c_mu=c_mu,
            mean=mean, p_sigma=p_sigma, p_c=p_c, C=C, sigma=sigma,
            device=device,
        )

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
        """
        n, lam, mu = s['n'], s['lam'], s['mu']
        device = s['device']

        # Eigendecompose C for sampling (C = B D^2 B^T)
        D2, B = torch.linalg.eigh(s['C'])
        D = torch.sqrt(torch.clamp(D2, min=1e-20))
        invsqrtC = B @ torch.diag(1.0 / D) @ B.T

        # Sample population: x_k = mean + sigma * B D z_k
        z = torch.randn(lam, n, device=device, dtype=torch.float64)
        y = z @ (B * D).T  # (lam, n)
        population = s['mean'] + s['sigma'] * y  # (lam, n)

        # Evaluate
        costs = evaluate_fn(population.float())  # (lam,)

        # Sort by cost (minimization)
        order = torch.argsort(costs)
        y_sel = y[order[:mu]]  # (mu, n) — selected steps

        # Weighted recombination
        y_w = (s['weights'].unsqueeze(1) * y_sel).sum(dim=0)  # (n,)
        s['mean'] = s['mean'] + s['sigma'] * y_w

        # Step-size path
        s['p_sigma'] = (
            (1.0 - s['c_sigma']) * s['p_sigma']
            + math.sqrt(s['c_sigma'] * (2.0 - s['c_sigma']) * s['mu_eff'])
            * (invsqrtC @ y_w)
        )
        norm_ps = torch.linalg.norm(s['p_sigma']).item()
        s['sigma'] *= math.exp(
            (s['c_sigma'] / s['d_sigma']) * (norm_ps / s['E_chi'] - 1.0)
        )

        # Covariance path
        h_sigma = 1.0 if (
            norm_ps / math.sqrt(1.0 - (1.0 - s['c_sigma']) ** (2 * (1 + 1)))
            < (1.4 + 2.0 / (n + 1.0)) * s['E_chi']
        ) else 0.0

        s['p_c'] = (
            (1.0 - s['cc']) * s['p_c']
            + h_sigma * math.sqrt(s['cc'] * (2.0 - s['cc']) * s['mu_eff'])
            * y_w
        )

        # Covariance matrix update
        rank_one = s['p_c'].unsqueeze(1) @ s['p_c'].unsqueeze(0)
        rank_mu = sum(
            s['weights'][i] * y_sel[i].unsqueeze(1) @ y_sel[i].unsqueeze(0)
            for i in range(mu)
        )
        s['C'] = (
            (1.0 - s['c1'] - s['c_mu']) * s['C']
            + s['c1'] * rank_one
            + s['c_mu'] * rank_mu
        )

        return costs[order[0]].item()


def minimize_cma_es(
    circuit: Circuit,
    theta: torch.Tensor,
    hamiltonian: Hamiltonian,
    cma: CMAES,
    generations: int,
    best_value_method: str,
    wall_clock_cap: float | None = None,
):
    """CMA-ES optimization loop for variational quantum circuits.

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

    def evaluate_fn(population: torch.Tensor) -> torch.Tensor:
        """Evaluate a population of parameter vectors."""
        pop = population.to(device)
        return circuit.get_expectation_value(
            pop, hamiltonian, cma.measure_method, cma.shots
        )

    for gen in range(generations):
        it_time = time.time()

        best_cost = cma._step(s, evaluate_fn)

        if device == 'cuda':
            torch.cuda.synchronize()
        iteration_times.append(time.time() - it_time)

        exp_values.append(best_cost)

        # Update theta to current mean
        theta = s['mean'].float()

        # Best bitstring from current mean
        with torch.no_grad():
            _tensor = circuit.build_tensor(theta)

        if best_value_method == 'highest_probability':
            best_result.append(
                get_value_of_highest_probability(_tensor, device)
            )
        elif best_value_method == 'argmax_tr_noinv_BE':
            best_result.append(
                argmax_bitstring_tr_right_suffix(_tensor)
            )
        elif best_value_method == 'full_contraction':
            best_idx = contract_tensor_ring(_tensor).abs().pow(2).argmax().item()
            best_bitstring = format(best_idx, f'0{circuit.num_qubit}b')
            best_result.append(best_bitstring[::-1])
        else:
            best_result.append(
                get_value_of_highest_probability(_tensor, device)
            )

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
