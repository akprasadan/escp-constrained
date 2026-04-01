"""
probability_functions.py

Helper functions to compute the SCP conditional probabilities.
"""

import numpy as np
import pandas as pd
from src.output_functions import make_partition as make_partition
from src.output_functions import counts_observed as counts_observed
from src.box_functions import mask_in_A as mask_in_A
from src.box_functions import rect_A as rect_A
from src.box_functions import check_lambda_bounds as check_lambda_bounds


def probability_input_validation(
    prior_counts: np.ndarray, prior_A_counts: np.ndarray, obs_counts: np.ndarray, K: int
) -> None:
    """
    Checks that valid inputs are given into probability_functions.prob_A().

    Parameters
    ----------
    prior_counts : (M, ) numpy array
        Array of counts of the lambda_j ~ prior whose output lies in each of M bins
    prior_A_counts : (M, ) numpy array
        Array of counts of the (lambda_j ~ prior) intersect A
        whose output lies in each of M bins.
    obs_counts : (M, ) numpy array
        Array of counts of the observed q_i lying in each of M bins
    K : int
        Number of q_i drawn in total

    Raises
    --------
    ValueError : If any input is invalid.
    """
    if not (prior_counts.shape == prior_A_counts.shape == obs_counts.shape):
        raise ValueError("Incompatible array lengths.")

    for name, array in (
        ("prior_counts", prior_counts),
        ("prior_A_counts", prior_A_counts),
        ("obs_counts", obs_counts),
    ):
        if array.ndim != 1:
            raise ValueError(f"{name} is not a 1D array.")
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must be integer counts.")
        if np.any(array < 0):
            raise ValueError(f"{name} cannot contain negative values.")

    if np.any(prior_A_counts > prior_counts):
        raise ValueError("prior_A_counts > prior_counts.")
    if not isinstance(K, int) or K <= 0:
        raise ValueError("K must be a positive integer.")
    if obs_counts.sum() != K:
        raise ValueError("sum(obs_counts) != K.")


def prob_A(
    prior_counts: np.ndarray, prior_A_counts: np.ndarray, obs_counts: np.ndarray, K: int
) -> float:
    """
    Estimate P(A) using the eSCP formula (Equation 4.5 in paper).

    Suppose there are M bins. Assume K datapoints {q_i} are given.
    Draw J points {lambda_j} from the prior. For each bin, compute
    the values prior_A_counts, prior_counts, and obs_counts, where c stands for 'count'. Within a bin,
    prior_counts is the frequency of lambda_j, prior_A_counts is the frequency of lambda_j that
    are also in A, and obs_counts is the frequency of q_i. Summing over the M bins,
    our probability estimate is:

        P(A) = sum_{M bins} (prior_A_counts / prior_counts) * (obs_counts / K)

    We skip any bins where prior_counts = 0 to avoid division by 0.
    If all of the prior_counts are 0, set P(A) = 0. For non-zero prior_counts,
    clearly prior_A_counts <= prior_counts and obs_counts <= K, so P(A) <= 1.

    Parameters
    ----------
    prior_counts : (M, ) numpy array
        Array of counts of the lambda_j ~ prior whose output lies in each of M bins
    prior_A_counts : (M, numpy array
        Array of counts of the (lambda_j ~ prior) intersect A
        whose output lies in each of M bins.
    obs_counts : (M, ) numpy array
        Array of counts of the observed q_i lying in each of M bins
    K : int
        Number of q_i drawn in total

    Returns
    --------
    prob : float
        The estimate of the SCP solution probability for a particular A
    """
    # Check if inputs are valid
    probability_input_validation(prior_counts, prior_A_counts, obs_counts, K)

    # Avoid division by 0
    mask = prior_counts > 0

    # If no lambda_is are in any of the bins, the probability is 0
    if not np.any(mask):
        return 0.0

    prob = np.sum((prior_A_counts[mask] / prior_counts[mask]) * (obs_counts[mask] / K))

    return prob


def prob_over_grid(
    q_data: np.ndarray,
    prior_data: np.ndarray,
    q_prior_data: np.ndarray,
    bin_edges: np.ndarray,
    grid_bounds: tuple[float, float, float, float],
    h: int,
) -> pd.DataFrame:
    """
    Compute per-tile probabilities P(A_ij) over an h x h grid on [xmin, xmax] x [ymin, ymax].
    Each row is a tile A_{ij} with [xmin, xmax) x [ymin, ymax) and its probability.

    Parameters
    ----------
    q_data : (K, ) numpy array
        Array of output values q_i, where q_i = Q(lambda_i), lambda_i ~ TGD
    prior_data : (J, ) np.ndarray
        Array of points lambda_i drawn from prior. May be outside the given boundaries.
    q_prior_data : (J, ) numpy array
        Evaluation of elements of prior_data by Q
    bin_edges : (M + 1, ) numpy array
        Array of M + 1 bin edges defining partition of output space D
    xmin, xmax, ymin, ymax : float
        Rectangle bounds; require `xmin < xmax` and `ymin < ymax`.
    h : int
        Number of tiles on either the horizontal or vertical axis of the rectangle.

    Returns
    -------
    pandas.DataFrame with 5 columns ['xmin','xmax','ymin','ymax','prob'] and h^2 rows.
    """
    # First some basic input validation
    xmin, xmax, ymin, ymax = grid_bounds
    check_lambda_bounds(xmin, xmax, ymin, ymax)

    if not (isinstance(h, int) and h >= 1):
        raise ValueError("h must be an integer >= 1.")
    if len(q_prior_data) != len(prior_data):
        raise ValueError("q_prior_data and prior_data not same length.")

    obs_counts = counts_observed(q_data, bin_edges)
    prior_counts = counts_observed(q_prior_data, bin_edges)
    K = len(q_data)

    x_edges = make_partition(float(xmin), float(xmax), h)
    y_edges = make_partition(float(ymin), float(ymax), h)

    rows = []
    # Each row = one half-open box [x0,x1) x [y0,y1), to which we assign probability
    for j in range(h):  # y-rows (bottom -> top)
        y0, y1 = float(y_edges[j]), float(y_edges[j + 1])
        for i in range(h):  # x-cols (left -> right)
            x0, x1 = float(x_edges[i]), float(x_edges[i + 1])
            A = rect_A(x0, x1, y0, y1)  # [x0,x1) x [y0,y1),
            mask_A = mask_in_A(prior_data, A)
            prior_A_counts = counts_observed(q_prior_data[mask_A], bin_edges)
            p_A = prob_A(prior_counts, prior_A_counts, obs_counts, K)
            rows.append({"xmin": x0, "xmax": x1, "ymin": y0, "ymax": y1, "prob": p_A})

    return pd.DataFrame(rows)
