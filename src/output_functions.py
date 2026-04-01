"""
output_functions.py

Helper functions involving d-dimensional output space D, including applying the map Q
on n-dimensional input space and partitioning the output space.
"""

from collections.abc import Callable, Sequence
from inspect import signature

import numpy as np


def apply_map(points: np.ndarray, Q: Callable) -> np.ndarray:
    """
    Apply a possibly vector-valued map Q to a set of points, while performing some additional
    input/output validation.
    Works in any dimension.

    Parameters
    ----------
    points : (N, d) ndarray
        Array of N points with d dimensions.
    Q : Callable
        Function that accepts `points` and returns an array of length N.
        Example: `Q = lambda x, y, z: [x**2 + y**2 - z, x + z]`

    Returns
    -------
    q : (N, ) ndarray
        Result of applying `Q` to each point.
    """
    # Input validation
    if points.shape[0] == 0:
        raise ValueError("points must be non-empty.")

    q_input_size = len(signature(Q).parameters)
    if points.shape[1] != q_input_size:
        raise ValueError("Q takes in too many or too few arguments.")

    q = np.asarray(Q(*points.T), dtype=float)

    # Output validation
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
        Array of shape (d, M + 1) with bin edges.
    """
    # Basic input validation
    if not (isinstance(M, int) and M >= 1):
        raise ValueError("M must be an integer >= 1.")

    d = len(min_vals)
    if d != len(max_vals):
        raise ValueError("Incompatible lengths (dimension) for min_vals and max_vals.")

    for i in range(d):
        if not (min_vals[i] < max_vals[i]):
            raise ValueError(f"min_vals < max_vals for dimension {i}.")

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
    bin_edges : (d, M + 1) numpy array
        Array of M + 1 bin edges defining the intervals for each dimension.

    Returns
    -------
    counts : (d, M) numpy array of ints
        Counts for each bin in each dimension.
    """
    counts, _ = np.histogramdd(q_data, bins=bin_edges)

    return counts
