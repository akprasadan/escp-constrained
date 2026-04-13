"""
probability_functions.py

Helper functions to compute the SCP conditional probabilities.
"""

import warnings


import numpy as np
import pandas as pd
from src.output_functions import counts_validation, make_partition
from src.output_functions import counts_observed
from src.box_functions import mask_in_A
from src.box_functions import rect_A
from src.box_functions import check_grid_bounds


def prob_input_validation(
    prior_counts: np.ndarray,
    prior_A_counts: np.ndarray,
    obs_counts: np.ndarray,
    K: int,
) -> None:
    """
    Checks that valid inputs are given into probability_functions.prob_A(). Assume
    that we are working on a d-dimensional output space D, and use M bins per axis,
    to give M^d bins total.

    Parameters
    ----------
    prior_counts : (M, M, ..., M) (d-many) numpy array
        Array of counts of the lambda_j ~ prior whose output lies in each bin.
    prior_A_counts : (M, M, ..., M) (d-many) numpy array
        Array of counts of the (lambda_j ~ prior) intersect A
        whose output lies in each of bin
    obs_counts : (M, M, ..., M) shaped (d-many) numpy array
        Array of counts of the observed q_i lying in each of bins.
    K : int
        Number of q_i drawn in total

    Raises
    --------
    ValueError : If any input is invalid.
    """
    if not (prior_counts.shape == prior_A_counts.shape == obs_counts.shape):
        raise ValueError("Incompatible array shapes.")

    if np.any(prior_A_counts > prior_counts):
        raise ValueError("prior_A_counts > prior_counts.")
    if not isinstance(K, int) or K <= 0:
        raise ValueError("K must be a positive integer.")
    if obs_counts.sum() != K:
        msg = (
            f"\n{K - obs_counts.sum()} / {K} = {(1 - obs_counts.sum()/K) * 100:.3f}% "
            "of your observed q values were not contained in the bin edges. This will "
            "lower your estimated probability slightly. To fix, enlargen bin_edges "
            "in the estimate_functions.py functions."
        )
        warnings.warn(msg, RuntimeWarning)


def prob_A(
    prior_counts: np.ndarray,
    prior_A_counts: np.ndarray,
    obs_counts: np.ndarray,
    K: int,
) -> float:
    """
    Estimate P(A) using the eSCP formula (Equation 4.5 in paper).

    Suppose there are M bins along each of d-dimensions, for M^d bins.
    Assume K datapoints {q_i} are given.
    Draw J points {lambda_j} from the prior. For each bin, compute
    the values prior_A_counts, prior_counts, and obs_counts, where c stands for 'count'. Within a bin,
    prior_counts is the frequency of lambda_j, prior_A_counts is the frequency of lambda_j that
    are also in A, and obs_counts is the frequency of q_i. Summing over the M^d bins,
    our probability estimate is:

        P(A) = sum_{M^d bins} (prior_A_counts / prior_counts) * (obs_counts / K)

    We skip any bins where prior_counts = 0 to avoid division by 0.
    If all of the prior_counts are 0, set P(A) = 0. For non-zero prior_counts,
    clearly prior_A_counts <= prior_counts and obs_counts <= K, so P(A) <= 1.

    Parameters
    ----------
    prior_counts : (M, M, ..., M) shaped (d-many) numpy array
        Array of counts of the lambda_j ~ prior whose output lies in each bin
    prior_A_counts : (M, M, ..., M) shaped (d-many) numpy array
        Array of counts of the (lambda_j ~ prior) intersect A
        whose output lies in each bin.
    obs_counts : (M, M, ..., M) shaped (d-many) numpy array
        Array of counts of the observed q_i lying in each of bin
    K : int
        Number of q_i drawn in total

    Returns
    --------
    prob : float
        The estimate of the SCP solution probability for a particular A
    """
    # Check if inputs are valid
    prob_input_validation(prior_counts, prior_A_counts, obs_counts, K)

    # Avoid division by 0
    mask = prior_counts > 0

    # If no lambda_is are in any of the bins, the probability is 0
    if not np.any(mask):
        return 0
    prob = np.sum(
        (prior_A_counts[mask] / prior_counts[mask]) * (obs_counts[mask] / K)
    )

    return prob


def prob_over_grid(
    q_data: np.ndarray,
    prior_data: np.ndarray,
    q_prior_data: np.ndarray,
    bin_edges: np.ndarray,
    grid_mins: np.ndarray,
    grid_maxes: np.ndarray,
    h: int,
) -> pd.DataFrame:
    """
    Compute per-tile probabilities P(A_ij) over an h x h grid on [xmin, xmax] x [ymin, ymax].
    Each row is a tile A_{ij} with [xmin, xmax) x [ymin, ymax) and its probability.
    Let d_L be dimension of Lambda, and d_D be dimension of D.

    Parameters
    ----------
    q_data : (K, d_D) numpy array
        Array of output values q_i, where q_i = Q(lambda_i), lambda_i ~ TGD
    prior_data : (J, d_L) np.ndarray
        Array of points lambda_i drawn from prior. May be outside the given boundaries.
    q_prior_data : (J, d_D) numpy array
        Evaluation of elements of prior_data by Q
    bin_edges : (M + 1, d_D) numpy array
        Array of M + 1 bin edges defining partition of output space D
    grid_mins, grid_maxes : (d_L, ) np.ndarray
        Array of bounds for box to compute probability over.
        Require grid_mins[i] < grid_maxes[i] for each i = 0, 1, ..., d_L - 1
    h : int
        Number of tiles on any axis of the box

    Returns
    -------
    pandas.DataFrame with 5 columns ['xmin','xmax','ymin','ymax','prob'] and h^2 rows.
    """
    # First some basic input validation
    check_grid_bounds(grid_mins, grid_maxes)
    input_dim = prior_data.shape[1]
    assert len(grid_mins) == input_dim

    if not (isinstance(h, int) and h >= 1):
        raise ValueError("h must be an integer >= 1.")
    if len(q_prior_data) != len(prior_data):
        raise ValueError("q_prior_data and prior_data not same length.")

    obs_counts = counts_observed(q_data, bin_edges)
    prior_counts = counts_observed(q_prior_data, bin_edges)
    counts_validation(obs_counts)
    counts_validation(prior_counts)

    K = len(q_data)

    edges = make_partition(grid_mins, grid_maxes, h)
    # inside np.ndindex, we pass in (h, h, ..., h) in R^d
    # Then we get all indices in the box, i.e., elements of {0, 1, ..., h-1}^d
    # Then we loop over each one. We extract the bin_edges at that index
    rows = []
    for idx in np.ndindex(*(h,) * input_dim):
        idx = np.array(idx)
        tile_mins = edges[idx, np.arange(input_dim)]
        tile_maxes = edges[idx + 1, np.arange(input_dim)]

        A = rect_A(
            tile_mins, tile_maxes
        )  # E.g., [x0,x1) x [y0,y1) x [z_0, z_1)
        mask_A = mask_in_A(prior_data, A)
        q_prior_counts_A = q_prior_data[mask_A]
        prior_A_counts = counts_observed(q_prior_counts_A, bin_edges)
        counts_validation(prior_A_counts)

        p_A = prob_A(prior_counts, prior_A_counts, obs_counts, K)
        rows.append({"mins": tile_mins, "maxes": tile_maxes, "prob": p_A})

    df = pd.DataFrame(rows)

    """For convenience, if the input data is 2 or 3 dimensional,
    let's unwrap the mins and maxes columns to be {xmin, ymin, zmin} and
    {xmax, ymax, zmax}, respectively, omitting the zmin, zmax columns if 2 dimensional.
    In practice, we will only be using these dimensions."""

    if input_dim in [1, 2, 3]:
        # e.g., yields ["x", "y"] if dim 2, or ["x", "y", "z"] if dim 3
        axis_names = ["x", "y", "z"][:input_dim]
        axis_min_names = [f"{ax}min" for ax in axis_names]
        axis_max_names = [f"{ax}max" for ax in axis_names]

        # unwrap the mins/maxes columns
        df[axis_min_names] = df["mins"].tolist()
        df[axis_max_names] = df["maxes"].tolist()

        # remove now redundant columns
        df = df.drop(columns=["mins", "maxes"])

    return df
