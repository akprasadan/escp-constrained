"""
sinkhorn_functions.py

Given one-dimensional data points q_1, ..., q_n from distribution P_D on D, and u_1, ..., u_m from
distribution P_U on U, we wish to estimate a joint distribution P_{D,U} using Sinkhorn's algorithm.

Suppose we have some map Q(lambda, u). Define the cost function
    c(lambda, q, u) = ||q - Q(lambda, u)||.

With Sinkhorn's algorithm, for a fixed lambda and epsilon, we solve

    min_{Pi} integral_D c(lambda, q, u) dPi(q, u) + epsilon * KL(Pi, P_D x P_U)

where Pi ranges over all joint distributions on (D, U) whose marginals in D and U match P_D and P_U,
respectively, P_D x P_U is the independent coupling of P_D and P_U, KL( , ) is the KL divergence
of two distributions, and epsilon is some penalty parameter.

However, the lambda is drawn from some random distribution of interest. To incorporate this,
we consider as a cost function:

    c(q, u) = integral_{Lambda} ||q - Q(lambda, u)|| p(lambda) dlambda,

where p(lambda) is a prior over lambda. Then we solve the Sinkhorn algorithm using this
cost function instead.
"""

from typing import Callable

import numpy as np
import ot
import ot.plot
import matplotlib.pylab as pl

from src.box_functions import rect_A
from src.sampler_functions import sample_distr
from src.output_functions import apply_map


def uniform_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(0, 4, size=(n, 2))


def compute_cost(
    q: float,
    u: float,
    Q: Callable,
    prior: Callable[[np.random.Generator, int], np.ndarray],
    prior_data: np.ndarray | None,
    MC_sample_size: int,
    lambda_mins: np.ndarray,
    lambda_maxes: np.ndarray,
    seed: int,
) -> float:
    """Compute c(q, u) = integral_{Lambda} |q - Q(lambda, u)|^2 p(lambda) dlambda.

    1; We draw 10000 points lambda_i ~ prior (truncated to Lambda).
    2. Evaluate |q - Q(lambda_i, u)|^2 for each i
    3. Compute the average value of these evaluations, which approximates c(q, u).

    The user can either pass in a prior_sample directly or input a prior distribution.
    """
    # (N, dim(Lambda)) array of prior samples, either using given distribution or
    # direct data input
    prior_sample = (
        sample_distr(n=MC_sample_size, seed=seed, distr=prior)
        if prior_data is None
        else prior_data
    )

    # Constrain prior to lie in Lambda
    lambda_bounds_mask = rect_A(lambda_mins, lambda_maxes)
    prior_mask = lambda_bounds_mask(prior_sample)
    prior_sample = prior_sample[prior_mask]
    len_prior_sample = len(prior_sample)

    # Package together (lambda, u) for (N, dim(Lambda) + 1) array
    Q_inputs = np.hstack((prior_sample, np.full((len_prior_sample, 1), u)))
    Q_outputs = apply_map(Q_inputs, Q)  # (N, 1)

    # broadcasts q to (N, 1), return (N, 1)
    cost_vals = np.square(q - Q_outputs)

    weighted_avg_cost = np.mean(cost_vals, dtype=np.float64)

    return weighted_avg_cost


def run_sinkhorn(
    q_data: np.ndarray,
    u_data: np.ndarray,
    Q: Callable,
    prior: Callable[[np.random.Generator, int], np.ndarray],
    prior_data: np.ndarray | None,
    lambda_mins: np.ndarray,
    lambda_maxes: np.ndarray,
    MC_sample_size: int,
    grid_sample_size: int,
    seed: int,
):

    fixed_cost_args = (
        Q,
        prior,
        prior_data,
        MC_sample_size,
        lambda_mins,
        lambda_maxes,
        seed,
    )
    cost_matrix = np.array(
        [[compute_cost(q, u, *fixed_cost_args) for q in q_data] for u in u_data]
    )
    sinkhorn_sol = ot.sinkhorn(
        a=q_data, b=u_data, M=cost_matrix, reg=0.001, verbose=False
    )

    q_grid = np.linspace(np.min(q_data), np.max(q_data), grid_sample_size)
    u_grid = np.linspace(0.01, 0.99, grid_sample_size)

    pl.figure(4, figsize=(5, 5))
    ot.plot.plot1D_mat(q_data, u_data, sinkhorn_sol, "OT matrix Sinkhorn")

    pl.show()


if __name__ == "__main__":
    q_data = np.random.normal(5, 30, 50)
    u_data = np.random.random(50)
    prior = uniform_sampler
    lambda_mins = np.array([1, 1])
    lambda_maxes = np.array([3, 3])
    MC_sample_size = 1000
    grid_sample_size = 100
    Q = lambda x, y, z: x**2 + y**2 + z

    # run_sinkhorn(
    #     q_data,
    #     u_data,
    #     Q,
    #     prior,
    #     lambda_mins,
    #     lambda_maxes,
    #     MC_sample_size,
    #     grid_sample_size,
    #     seed=1,
    # )
