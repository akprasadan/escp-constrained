"""
sinkhorn_functions.py

Given one-dimensional data points q_1, ..., q_n from distribution P_D on D, and u_1, ..., u_m from
distribution P_U on U, we wish to estimate a joint distribution P_{D,U} using Sinkhorn's algorithm.

Suppose we have some map (lambda, u) --> Q(lambda, u), and let p be a prior on Lambda.
Define the cost functio:

    c(q, u) = integral_{Lambda} |q - Q(lambda, u)|^2 p(lambda) d lambda

With Sinkhorn's algorithm, for a fixed epsilon, we solve

    min_{Pi} integral_{D x U} c(q, u) dPi(q, u) + epsilon * KL(Pi, P_D x P_U)

where Pi ranges over all joint distributions on (D, U) whose marginals in D and U match P_D and P_U,
respectively, P_D x P_U is the independent coupling of P_D and P_U, KL( , ) is the KL divergence
of two distributions, and epsilon is some penalty parameter.
"""

from typing import Callable

import numpy as np
import ot
import ot.plot
import matplotlib.pyplot as plt
import pandas as pd


from src.box_functions import rect_A
from src.estimate_functions import empirical_scp
from src.sampler_functions import sample_distr
from src.output_functions import apply_map, counts_observed, make_partition


def compute_cost(
    q: float,
    u: float,
    Q: Callable[..., np.ndarray],
    prior: Callable[[np.random.Generator, int], np.ndarray],
    lambda_mins: np.ndarray,
    lambda_maxes: np.ndarray,
    MC_sample_size: int,
    seed: int,
) -> float:
    """Compute c(q, u) = integral_{Lambda} |q - Q(lambda, u)|^2 p(lambda) dlambda.

    1; We draw MC_sample_size many points lambda_i ~ prior (truncated to Lambda).
    2. Evaluate |q - Q(lambda_i, u)|^2 for each i
    3. Compute the average value of these evaluations, which approximates c(q, u).

    Parameters
    ----------
    q : float
        Value in D
    u : float
        Value in Q
    Q : Callable[..., np.ndarray]
        Map from Lambda x U --> D. Not augmented version.
    prior : Callable[[np.random.Generator, int], np.ndarray]
        Distribution of lambda in Lambda to weight cost function by
    lambda_mins, lambda_maxes : (dim(Lambda), ) shaped np.ndarray
        Array of minimums/maximums, respectively, specifying hyper-rectangle in Lambda
    MC_sample_size : int
        Number of Monte Carlo samples to use to estimate the integral over Lambda
    seed : int
        For reproducibility in prior samples.

    Returns
    -------
    weighted_avg_cost : float
        The value of the cost function c(q, u) at a point (q, u) in D x U.
    """
    # (N, dim(Lambda)) array of prior samples
    prior_sample = sample_distr(n=MC_sample_size, seed=seed, distr=prior)

    # Constrain prior to lie in Lambda
    lambda_bounds_mask = rect_A(lambda_mins, lambda_maxes)
    prior_mask = lambda_bounds_mask(prior_sample)
    prior_sample = prior_sample[prior_mask]
    prior_sample_size = len(prior_sample)

    # Construct (N, dim(Lambda) + 1) array of (lambda, u) pairs, apply map
    Q_inputs = np.column_stack(
        (prior_sample, np.full((prior_sample_size, 1), u))
    )
    Q_outputs = apply_map(Q_inputs, Q)  # (N, 1)

    cost_vals = np.square(q - Q_outputs)
    weighted_avg_cost = np.mean(cost_vals, dtype=np.float64)

    return weighted_avg_cost


def run_sinkhorn(
    q_data: np.ndarray,
    u_data: np.ndarray,
    Q: Callable[..., np.ndarray],
    prior: Callable[[np.random.Generator, int], np.ndarray],
    reg_term: float,
    lambda_mins: np.ndarray,
    lambda_maxes: np.ndarray,
    MC_sample_size: int,
    grid_res: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the Sinkhorn algorithm given two unpaired empirical samples from D and U. Result
    is an empirical probability distribution over D x U corresponding to a joint distribution.

    Parameters
    ----------
    q_data : (N_1, 1) numpy array
        Array of data in D
    u : (N_2, 1) numpy array
        Array of data in U
    Q : Callable[..., np.ndarray]
        Map from Lambda x U --> D. Not the augmented version.
    prior : Callable[[np.random.Generator, int], np.ndarray]
        Distribution of lambda in Lambda to weight cost function by
    lambda_mins, lambda_maxes : (dim(Lambda), ) shaped numpy array
        Array of minimums/maximums, respectively, specifying hyper-rectangle in Lambda
    MC_sample_size : int
        Number of Monte Carlo samples to use to estimate the integral over Lambda
    grid_res : int
        Number of bins to use along either axis (D or U) to estimate Sinkhorn on
    seed : int
        For reproducibility in prior samples.

    Returns
    --------
    sinkhorn_sol : (grid_res, grid_res) numpy array
        Sinkhorn estimate
    u_grid, q_grid : (grid_res + 1, ) numpy array
        Bin edges to use for  U or D, respectively
    empirical_u, empirical_q :(grid_res, ) numpy array
        Empirical probability distribution for U data and Q data, respectively,
        over the bins in u_grid, q_grid.
    """

    u_grid = make_partition(np.min(u_data), np.max(u_data), grid_res)
    q_grid = make_partition(np.min(q_data), np.max(q_data), grid_res)

    # Estimate empirical distribution
    empirical_u = counts_observed(u_data, bin_edges=u_grid)
    empirical_q = counts_observed(q_data, bin_edges=q_grid)

    # Add small positive number and normalize for stability
    empirical_u = (empirical_u + 0.0001) / empirical_u.sum()
    empirical_q = (empirical_q + 0.0001) / empirical_q.sum()

    fixed_cost_args = (
        Q,
        prior,
        lambda_mins,
        lambda_maxes,
        MC_sample_size,
        seed,
    )
    # Estimate cost at the bottom-left corner of each bin formed by u_grid x q_grid
    cost_matrix = np.array(
        [
            [compute_cost(q, u, *fixed_cost_args) for u in u_grid[:grid_res]]
            for q in q_grid[:grid_res]
        ]
    )
    cost_matrix = cost_matrix / np.sum(cost_matrix)

    sinkhorn_sol = ot.sinkhorn(
        a=empirical_u,
        b=empirical_q,
        M=cost_matrix,
        reg=reg_term,
        verbose=False,
    )

    sinkhorn_sol = sinkhorn_sol / sinkhorn_sol.sum()  # type: ignore

    return sinkhorn_sol, u_grid, q_grid, empirical_u, empirical_q


def sample_joint_pairs(
    sinkhorn_sol: np.ndarray,
    u_grid: np.ndarray,
    q_grid: np.ndarray,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample pairs (q, u) from the joint distribution on D x U estimated by Sinkhorn's algorithm.

    Parameters
    -----------
    sinkhorn_sol : (grid_res, grid_res) numpy array
        Sinkhorn estimate, where grid_res was chosen in Sinkhorn algorithm.
    u_grid, q_grid : (grid_res + 1, ) numpy array
        Bin edges to use for  U or D.
    sample_size : int
        Number of joint pairs (q, u) to draw
    seed : int
        For reproducible sampling from Sinkhorn estimate.

    Returns
    --------
    q_sink_samples, u_sink_samples : (sample_size, ) arrays
        Two paired arrays of q and u samples.
    """
    u_bin_count = len(u_grid) - 1
    q_bin_count = len(q_grid) - 1

    if sinkhorn_sol.shape != (u_bin_count, q_bin_count):
        raise ValueError("Sinkhorn solution of wrong dimensions.")

    rng = np.random.default_rng(seed=seed)

    # sinkhorn_sol.ravel() converts a, for example, 100x100 grid of probabilities
    # into one 100^2 long probability vector, which defines a distribution over
    # [1, 2, ..., 100^2] from which we sample with replacement.
    # Each index chosen is a tile from that 100x100 grid, whose indices we extract
    # using np.unravel_index.
    flat_idx = rng.choice(
        a=sinkhorn_sol.size,
        size=sample_size,
        replace=True,
        p=sinkhorn_sol.ravel(),
    )

    u_sink_idxs, q_sink_idxs = np.unravel_index(
        flat_idx, shape=sinkhorn_sol.shape
    )
    if len(u_sink_idxs) != len(q_sink_idxs):
        raise ValueError(
            "Mismatched number of indices sampled from Sinkhorn solution"
        )

    if len(u_sink_idxs) != sample_size:
        raise ValueError(
            "Invalid quantity of indices sampled from Sinkhorn solution"
        )

    if q_sink_idxs.max() > q_bin_count or u_sink_idxs.max() > u_bin_count:
        raise ValueError("Invalid indices sampled from Sinkhorn solution")

    # If a particular tile is chosen, sample uniformly within that tile
    q_sink_samples = rng.uniform(q_grid[q_sink_idxs], q_grid[q_sink_idxs + 1])
    u_sink_samples = rng.uniform(u_grid[u_sink_idxs], u_grid[u_sink_idxs + 1])

    return q_sink_samples, u_sink_samples


def augmented_scp(
    q_data: np.ndarray,
    u_data: np.ndarray,
    J: int,
    M: int,
    augmented_Q: Callable[..., np.ndarray],
    prior: Callable[[np.random.Generator, int], np.ndarray],
    grid_mins: np.ndarray,
    grid_maxes: np.ndarray,
    input_grid_size: int,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameters
    ----------
    q_data, u_data : (N, 1) numpy array
        Paired arrays of data in D and U, respectively
    J : int
        Number of prior samples to draw on Lambda x U.
    M : int
        Number of bins for partitioning D x U (range of augmented Q).
    augmented_Q : Callable[..., np.ndarray],
        Augmented map Q: Lambda x U -> D x U, called as ``augmented_Q(x, y, z)``.
    prior : Callable[[np.random.Generator, int], np.ndarray]
        Sampler for the prior distribution on Lambda x U
    grid_mins, grid_maxes : (dim(Lambda) + 1, ) np.ndarray
        Array of bounds for Lambda x U
        Require grid_mins[i] < grid_maxes[i] for each i = 0, 1, ..., dim(Lambda)
    input_grid_size : int
        Number of grid bins per axis on Lambda x U
        (total ``input_grid_size**(dim(Lambda) + 1)`` bins).
    seed : int or None
        Seed for reproducibility; used to derive independent seeds for TGD and prior.

    Returns
    -------
    probs_df : pandas.DataFrame
        Estimated probabilities on the (Lambda, U) grid tiles.
    prior_df : pandas.DataFrame
        Prior samples on Lambda x U with `(q, u) = augmented_Q(lambda, u)` evaluations.
    """

    if len(q_data) != len(u_data):
        raise ValueError("D and U data have mismatched lengths.")

    augmented_q = np.column_stack([q_data, u_data])

    if augmented_q.shape != (len(q_data), 2):
        raise ValueError("Unexpected shape for augmented Q datapoints.")

    prob_df, prior_df = empirical_scp(
        augmented_q,
        J,
        M,
        augmented_Q,
        prior,
        grid_mins,
        grid_maxes,
        input_grid_size,
        seed,
    )

    return prob_df, prior_df


def plot_sinkhorn(
    sinkhorn_sol: np.ndarray,
    empirical_u: np.ndarray,
    empirical_q: np.ndarray,
    q_sink_samples: np.ndarray,
    u_sink_samples: np.ndarray,
    u_edges: np.ndarray,
    q_edges: np.ndarray,
) -> None:

    ot.plot.plot1D_mat(
        a=empirical_u,
        b=empirical_q,
        M=sinkhorn_sol,
        title="Cost matrix M",
        plot_style="xy",
        a_label="U",
        b_label="D",
    )

    H, _, _ = np.histogram2d(
        u_sink_samples, q_sink_samples, bins=[u_edges, q_edges]
    )
    H = H / H.sum()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    ax[0].pcolormesh(u_edges, q_edges, H.T, shading="auto", cmap="viridis")

    ax[0].set_title("Sampled from joint")

    ax[1].pcolormesh(
        u_edges, q_edges, sinkhorn_sol.T, shading="auto", cmap="viridis"
    )

    ax[1].set_title("Sinkhorn Solution")
    plt.tight_layout()
    plt.show()
    plt.close()
