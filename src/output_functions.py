"""
output_functions.py

Helper functions involving d-dimensional output space D, including applying the map Q
on n-dimensional input space and partitioning the output space.
"""

from collections.abc import Callable

import numpy as np

from src.box_functions import check_lambda_bounds


def apply_map(points: np.ndarray, Q: Callable[..., np.ndarray]) -> np.ndarray:
    """
    Apply a possibly vector-valued map Q to a set of points, while performing some additional
    input/output validation. Assume input dimension is d, and output dimension is d_out < d.

    Parameters
    ----------
    points : (N, d) ndarray
        Array of N points with d dimensions.
    Q : Callable
        Function that accepts `points` and returns an array of length N.
        Example: `Q = lambda x, y, z: [x**2 + y**2 - z, x + z]`

    Returns
    -------
    q : (N, d_out) ndarray
        Result of applying `Q` to each point.
    """
    # Input validation
    if points.shape[0] == 0:
        raise ValueError("points must be non-empty.")

    # Make sure (N, ) arrays are treated as (N, 1)
    points_new = points[:, None] if points.ndim == 1 else points

    q = np.asarray(Q(*points_new.T), dtype=float).T

    #  Make sure (N, ) arrays are treated as (N, 1)
    q = q[:, None] if q.ndim == 1 else q

    if len(q) != len(points_new):
        raise RuntimeError("Q(x, y) must have same length as 'points'.")

    return q


def make_partition(
    min_vals: np.ndarray, max_vals: np.ndarray, M: int
) -> np.ndarray:
    """
    Along axis i of d-dimensions, build `M + 1` equal-width bin edges spanning
    [`min_vals[i]`, `max_vals[i]`].

    Parameters
    ----------
    min_vals, max_vals : (d, ) np.ndarray
        Lower and upper bound of the range (inclusive) for each of d dimensions.
        Must satisfy `min_val[i] < max_val[i]` for each dimension i.
    M : int
        Number of bins along an axis (must be >= 1).

    Returns
    -------
    numpy.ndarray
        Array of shape (M + 1, d) with bin edges.
    """
    # Basic input validation
    if not (isinstance(M, int) and M >= 1):
        raise ValueError("M must be an integer >= 1.")

    # Make sure mins/max values of right length, orientation
    # Floats should be packed into arrays
    if isinstance(min_vals, float):
        min_vals = np.array([min_vals])
        max_vals = np.array([max_vals])

    check_lambda_bounds(min_vals, max_vals)

    unscaled_partition = np.linspace(0, 1, M + 1)
    partition_all_dimensions = (
        min_vals[:, None] + (max_vals - min_vals)[:, None] * unscaled_partition
    )
    return partition_all_dimensions.T


def counts_observed(q_data: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Count the number of values falling into each of M^d bins on a d-dimensional rectangle.
    If no points are given, return 0 for each of the bins.

    Parameters
    ----------
    q_data : (K, d) numpy array
        Array of observed values.
    bin_edges : (M + 1, d) numpy array
        Array of M + 1 bin edges defining the intervals for each dimension.

    Returns
    -------
    counts : (M, M, ..., M) shaped (d-many) numpy array of integers (float type possibly).
        Counts for each bin in each dimension. For d = 1, returns (M, ) shaped array.
    """
    M: int = len(bin_edges) - 1

    q_data_reshaped = q_data[:, None] if q_data.ndim == 1 else q_data
    dim: int = q_data_reshaped.shape[1]

    if bin_edges.shape[1] != dim:
        raise ValueError("Bins or data of wrong dimension.")

    if len(q_data_reshaped) == 0:
        return np.zeros((M,) * dim, dtype=np.int64)

    # np.histogramdd needs (d, M + 1) bins array input
    counts, _ = np.histogramdd(q_data_reshaped, bins=bin_edges.T)

    return counts
