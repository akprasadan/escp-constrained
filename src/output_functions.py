"""
output_functions.py

Helper functions involving d-dimensional output space D, including applying the map Q
on n-dimensional input space and partitioning the output space.
"""

from collections.abc import Callable
from inspect import signature

import numpy as np

from src.box_functions import check_lambda_bounds as check_lambda_bounds


def apply_map(points: np.ndarray, Q: Callable) -> np.ndarray:
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

    q = np.asarray(Q(*points.T), dtype=float).T

    # If q is a (N, ) vector, reshape it be (N, 1) for consistency with higher dim cases
    if q.ndim == 1:
        q = q.reshape(len(q), 1)

    if len(q) != len(points):
        raise RuntimeError("Q(x, y) must have same length as 'points'.")

    return q


def make_partition(min_vals: np.ndarray, max_vals: np.ndarray, M: int) -> np.ndarray:
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
    check_lambda_bounds(min_vals, max_vals)

    unscaled_partition = np.linspace(0, 1, M + 1)
    partition_all_dimensions = (
        min_vals[:, None] + (max_vals - min_vals)[:, None] * unscaled_partition
    )
    return partition_all_dimensions.T


def counts_observed(q_data: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Count the number of values falling into each of M*d bins on a d-dimensional rectangle.
    If no points are given, return 0 for each of the bins.

    Parameters
    ----------
    q_data : (K, d) numpy array
        Array of observed values.
    bin_edges : (M + 1, d) numpy array
        Array of M + 1 bin edges defining the intervals for each dimension.

    Returns
    -------
    counts : (M, M, ..., M) shaped (d-many) numpy array of ints
        Counts for each bin in each dimension. For d = 1, returns (M, ) shaped array.
    """
    M = len(bin_edges) - 1
    d = q_data.ndim

    if len(q_data) == 0:
        return np.zeros((M,) * d, dtype=np.int64)

    # np.histogramdd needs (d, M + 1) bins array input
    counts, _ = np.histogramdd(q_data, bins=bin_edges.T)

    # Validate that we got integer values
    if not np.array_equal(counts, np.rint(counts)):
        raise ValueError("Counts must be integers.")

    # Safely convert to integer now
    counts = counts.astype(np.int64)

    return counts
