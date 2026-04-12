"""
SPSA (Simultaneous Perturbation Stochastic Approximation) gradient estimator.

Estimates the gradient using only 2 function evaluations per step,
regardless of the number of parameters. Each step perturbs all parameters
simultaneously with random ±δ offsets drawn from a Bernoulli distribution.

Reference:
    Spall, J.C. (1998). "Implementation of the simultaneous perturbation
    algorithm for stochastic optimization." IEEE Trans. Aerosp. Electron. Syst.
"""

import torch

from ...circuit import Circuit
from ...hamiltonian.hamiltonian import Hamiltonian
from ...measure.enums import MeasureMethod
from .gradient import Gradient


class SPSAGradient(Gradient):
    """SPSA stochastic gradient estimator.

    Parameters
    ----------
    measure_method : MeasureMethod
        How to evaluate expectation values (sampling or contraction).
    shots : int
        Number of measurement shots per evaluation (0 = exact contraction).
    c : float
        Perturbation magnitude. Controls the finite-difference step size.
        Larger values reduce variance but increase bias. Typical: 0.1-0.2.
    """

    def __init__(
        self,
        measure_method: MeasureMethod = MeasureMethod.RIGHT_SUFFIX_SAMPLING,
        shots: int = 10000,
        c: float = 0.1,
    ):
        super().__init__(measure_method)
        self.shots = shots
        self.c = c
        self.last_exp_value = None

    def run(
        self, theta: torch.Tensor, circuit: Circuit, hamiltonian: Hamiltonian
    ) -> torch.Tensor:
        """Compute SPSA gradient estimate.

        Cost: exactly 2 expectation-value evaluations per call.

        Returns
        -------
        torch.Tensor
            Gradient estimate with same shape as theta, i.e. (P,).
        """
        device = theta.device
        P = theta.shape[0]

        # Bernoulli ±1 perturbation vector
        delta = 2 * torch.bernoulli(torch.full((P,), 0.5, device=device)) - 1

        theta_plus = theta + self.c * delta
        theta_minus = theta - self.c * delta

        # Stack into (2, P) batch for a single batched evaluation
        theta_batch = torch.stack([theta_plus, theta_minus], dim=0)
        exp_vals = circuit.get_expectation_value(
            theta_batch, hamiltonian, self.measure_method, self.shots
        )

        f_plus = exp_vals[0].item()
        f_minus = exp_vals[1].item()

        # Cache the midpoint estimate for the progress bar
        self.last_exp_value = 0.5 * (f_plus + f_minus)

        # SPSA gradient estimate: g_i = (f+ - f-) / (2 * c * delta_i)
        grad = (f_plus - f_minus) / (2.0 * self.c * delta)

        return grad
