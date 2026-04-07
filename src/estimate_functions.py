"""
estimate_functions.py

Run the eSCP algorithm over a 2D set Lambda for a real-valued map Q on Lambda
applied to data drawn from some trial-generating distribution (TGD), using a prior assumption.
The function synthetic_scp() will draw data from a user-chosen TGD and
apply the map Q to generate data, while empirical_scp() will start directly with
Q evaluations.

The functions in this module are used to run our main examples.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.probability_functions import prob_over_grid as prob_over_grid
from src.sampler_functions import sample_distr as sample_distr
from src.output_functions import make_partition as make_partition
from src.output_functions import apply_map as apply_map
from src.box_functions import rect_A as rect_A
from src.box_functions import check_lambda_bounds as check_lambda_bounds


def simplify_low_dimensional_dataframes(
    df, dim: int, column: str, axis_names: list[str]
) -> pd.DataFrame:

    if dim > 3:
        return df
    needed_axis_names = axis_names[:dim]
    df[needed_axis_names] = df[column].to_list()
    df = df.drop(columns=column)

    return df


def synthetic_scp(
    K: int,
    J: int,
    M: int,
    Q: Callable,
    tgd: Callable[[np.random.Generator, int], np.ndarray],
    prior: Callable[[np.random.Generator, int], np.ndarray],
    grid_mins: np.ndarray,
    grid_maxes: np.ndarray,
    lambda_grid_size: int,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Estimate eSCP grid probabilities for synthetic data.

    Procedure:

    1) Draw K samples lambda_i from the TGD on Lambda and compute q_data = Q(lambda_i).
    2) Partition q_data into M bins.
    3) Draw J samples lambda_j from the prior on Lambda and compute q_prior_data = Q(lambda_j).
    4) Estimate probabilities for tiles on a Lambda-grid over [xmin,xmax]x[ymin,ymax].

    Parameters
    ----------
    K : int
        Number of TGD samples (lambda_i).
    J : int
        Number of prior samples (lambda_j).
    M : int
        Number of bins for partitioning D (range of Q).
    Q : Callable[[np.ndarray, np.ndarray], np.ndarray],
        Map Q: Lambda -> D, called as ``Q(x, y)``.
    tgd : Callable[[np.random.Generator, int], np.ndarray]
        Sampler for the trial-generating distribution on Lambda.
    prior : Callable[[np.random.Generator, int], np.ndarray]
        Sampler for the prior distribution on Lambda.
    grid_mins, grid_maxes : (d_L, ) np.ndarray
        Array of bounds for Lambda.
        Require grid_mins[i] < grid_maxes[i] for each i = 0, 1, ..., d_L - 1
    lambda_grid_size : int
        Number of grid bins per axis on Lambda (total ``lambda_grid_size**2`` tiles).
    seed : int or None
        Seed for reproducibility; used to derive independent seeds for TGD and prior.

    Returns
    -------
    probs_df : pandas.DataFrame
        Estimated probabilities on the lambda-grid tiles.
    prior_df : pandas.DataFrame
        Prior samples on Lambda with their `q = Q(lambda)`.
    tgd_df : pandas.DataFrame
        TGD samples on Lambda with their `q = Q(lambda)`.
    """
    check_lambda_bounds(grid_mins, grid_maxes)

    # Obtain independent seeds for TGD and prior
    master_seed = np.random.default_rng(seed)
    tgd_seed = int(master_seed.integers(0, 2**63 - 1))
    prior_seed = int(master_seed.integers(0, 2**63 - 1))

    # 1) Sample from TGD and compute q_data = Q(lambda_i)
    tgd_data = sample_distr(n=K, seed=tgd_seed, distr=tgd)

    # Subset the observations inside the given range
    lambda_bounds_mask = rect_A(grid_mins, grid_maxes)
    tgd_mask = lambda_bounds_mask(tgd_data)
    tgd_data = tgd_data[tgd_mask]

    q_data = apply_map(tgd_data, Q)

    # 2) Partition q_data into M bins
    q_mins, q_maxes = np.min(q_data, axis=0), np.max(q_data, axis=0)
    bin_edges = make_partition(q_mins, q_maxes, M)

    # 3) Sample from prior and compute q_prior_data = Q(lambda_j). Then
    # subset those inside given range
    prior_data = sample_distr(J, seed=prior_seed, distr=prior)
    prior_mask = lambda_bounds_mask(prior_data)
    prior_data = prior_data[prior_mask]

    q_prior_data = apply_map(prior_data, Q)

    # 4) Estimate probabilities on lambda-grid
    probs_df = prob_over_grid(
        q_data=q_data,
        prior_data=prior_data,
        q_prior_data=q_prior_data,
        bin_edges=bin_edges,
        grid_mins=grid_mins,
        grid_maxes=grid_maxes,
        h=lambda_grid_size,
    )
    # Normalize the probability
    probs_df["prob"] /= probs_df["prob"].sum()

    # 5) Package prior and TGD samples with respective Q evaluations into dataframes
    # TODO: Correct for higher dimensions
    prior_df = pd.DataFrame(
        {
            "source": "prior",
            "lambda": list(prior_data),
            "q": list(q_prior_data),
        }
    )
    input_dim = prior_data.shape[1]
    output_dim = q_prior_data.shape[1]
    prior_df = simplify_low_dimensional_dataframes(
        prior_df, input_dim, "lambda", ["x", "y", "z"]
    )
    prior_df = simplify_low_dimensional_dataframes(
        prior_df, output_dim, "q", ["q_x", "q_y", "q_z"]
    )

    tgd_df = pd.DataFrame(
        {
            "source": "tgd",
            "lambda": list(tgd_data),
            "q": list(q_data),
        }
    )
    tgd_df = simplify_low_dimensional_dataframes(
        tgd_df, input_dim, "lambda", ["x", "y", "z"]
    )
    tgd_df = simplify_low_dimensional_dataframes(
        tgd_df, output_dim, "q", ["q_x", "q_y", "q_z"]
    )

    return probs_df, prior_df, tgd_df


def empirical_scp(
    q_data: np.ndarray,
    J: int,
    M: int,
    Q: Callable[[np.ndarray, np.ndarray], np.ndarray],
    prior: Callable[[np.random.Generator, int], np.ndarray],
    grid_mins: np.ndarray,
    grid_maxes: np.ndarray,
    lambda_grid_size: int,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate eSCP grid probabilities for empirical dataset.

    Procedure:

    1) Take K observations q_data from a dataset
    2) Partition q_i into M bins.
    3) Draw J samples lambda_j from the prior on Lambda and compute q_prior_data = Q(lambda_j).
    4) Estimate probabilities for tiles on a Lambda-grid over [xmin,xmax]x[ymin,ymax].

    Parameters
    ----------
    q_data : (K, d_D) numpy array
        The output data from an empirical dataset
    J : int
        Number of prior samples (lambda_j).
    M : int
        Number of bins for partitioning D (range of Q).
    Q : Callable
        Map Q: Lambda -> D, called as ``Q(x, y)``.
    prior : Callable[[np.random.Generator, int], np.ndarray]
        Sampler for the prior distribution on Lambda.
    grid_mins, grid_maxes : (d_L, ) np.ndarray
        Array of bounds for Lambda.
        Require grid_mins[i] < grid_maxes[i] for each i = 0, 1, ..., d_L - 1
    lambda_grid_size : int
        Number of grid bins per axis on Lambda (total ``lambda_grid_size**2`` tiles).
    seed : int or None
        Seed used to initialize an RNG for the prior sampler

    Returns
    -------
    probs_df : pandas.DataFrame
        Estimated probabilities on the lambda-grid tiles.
    prior_df : pandas.DataFrame
        Prior samples on Lambda with their `q = Q(lambda)`.
    """
    check_lambda_bounds(grid_mins, grid_maxes)

    # Prior seed only (TGD is empirical here)
    master = np.random.default_rng(seed)

    # 'Draw' a seed for the TGD even though we don't use it, so prior seeds are comparable
    # relative to synthetic version of this function
    _ = int(master.integers(0, 2**63 - 1))
    prior_seed = int(master.integers(0, 2**63 - 1))

    # 1) Partition q_data into M bins
    q_mins, q_maxes = np.min(q_data, axis=0), np.max(q_data, axis=0)
    bin_edges = make_partition(q_mins, q_maxes, M)

    # 2) Sample from prior and compute q_prior_data = Q(lambda_j)
    # Subset prior points outside our given range
    prior_data = sample_distr(J, seed=prior_seed, distr=prior)
    lambda_bounds_mask = rect_A(grid_mins, grid_maxes)
    prior_mask = lambda_bounds_mask(prior_data)
    prior_data = prior_data[prior_mask]

    q_prior_data = apply_map(prior_data, Q)

    # 3) Estimate probabilities on lambda-grid
    probs_df = prob_over_grid(
        q_data=q_data,
        prior_data=prior_data,
        q_prior_data=q_prior_data,
        bin_edges=bin_edges,
        grid_mins=grid_mins,
        grid_maxes=grid_maxes,
        h=lambda_grid_size,
    )
    # Normalize the probability
    probs_df["prob"] /= probs_df["prob"].sum()

    # 4) Package prior with Q evaluations
    prior_df = pd.DataFrame(
        {
            "source": "prior",
            "lambda": list(prior_data),
            "q": list(q_prior_data),
        }
    )
    input_dim = prior_data.shape[1]
    output_dim = q_prior_data.shape[1]
    prior_df = simplify_low_dimensional_dataframes(
        prior_df, input_dim, "lambda", ["x", "y", "z"]
    )
    prior_df = simplify_low_dimensional_dataframes(
        prior_df, output_dim, "q", ["q_x", "q_y", "q_z"]
    )

    return probs_df, prior_df
