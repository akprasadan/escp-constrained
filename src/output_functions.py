"""
output_functions.py

Helper functions involving output space D, including applying the map Q
and partitioning the output space.
"""

from collections.abc import Callable

import numpy as np


def apply_map(points: np.ndarray, Q: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Apply a real-valued map Q to 2D points, while performing some additional
    input/output validation.

    Parameters
    ----------
    points : (N, 2) ndarray
        Array of N points with columns [x, y].
    Q : Callable[[np.ndarray, np.ndarray], np.ndarray]
        Function that accepts two 1D arrays (x, y) and returns a 1D array of length N.
        Example: `Q = lambda x, y: x**2 + y**2`

    Returns
    -------
    q : (N, ) ndarray
        Result of applying `Q` to each point.
    """
    # Input validation
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be 2D array of shape (N, 2).")
    if points.shape[0] == 0:
        raise ValueError("points must be non-empty.")

    x, y = points[:, 0], points[:, 1]
    q = np.asarray(Q(x, y), dtype=float)

    # Output validation
    if q.ndim != 1:
        raise RuntimeError("Q(x, y) must return 1D array.")
    if len(q) != len(points):
        raise RuntimeError("Q(x, y) must have same length as 'points'.")

    return q


def make_partition(min_val: float, max_val: float, M: int) -> np.ndarray:
    """
    Build `M + 1` equal-width bin edges spanning [`min_val`, `max_val`].

    Parameters
    ----------
    min_val, max_val : float
        Lower and upper bound of the range (inclusive). Must satisfy `min_val < max_val`.
    M : int
        Number of bins (must be >= 1).

    Returns
    -------
    numpy.ndarray
        Array of shape (M + 1, ) with bin edges.
    """
    # Basic input validation
    if not (isinstance(M, int) and M >= 1):
        raise ValueError("M must be an integer >= 1.")

    if not (min_val < max_val):
        raise ValueError("min_val < max_val.")

    return np.linspace(float(min_val), float(max_val), M + 1)


def counts_observed(q_data: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Count the number of values falling into each of M bins. If no points are given,
    return 0 for each of the bins.

    Parameters
    ----------
    q_data : (K, ) numpy array
        Array of observed values.
    bin_edges : (M + 1, ) numpy array
        Array of M + 1 bin edges defining the intervals.

    Returns
    -------
    counts : (M, ) numpy array of ints
        Counts for each bin.
    """
    if np.any(np.diff(bin_edges) <= 0):
        raise ValueError("Bin edge vector not monotonic.")

    counts, _ = np.histogram(q_data, bins=bin_edges)
    return counts
