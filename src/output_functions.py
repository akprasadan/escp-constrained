"""
output_functions.py.

Helper functions involving d-dimensional output space D, including applying the
map Q on n-dimensional input space and partitioning the output space.
"""

from collections.abc import Callable

import numpy as np

from src.box_functions import check_grid_bounds


def apply_map(points: np.ndarray, Q: Callable[..., np.ndarray]) -> np.ndarray:
    """
    Apply a computer model.

    It applies a possibly vector-valued map Q to a set of points, while
    performing some additional input/output validation. Assume input dimension
    is d, and output dimension is d_out < d.

    Parameters
    ----------
    points : (N, d) numpy array
        Array of N points with d dimensions.
    Q : Callable
        Function that accepts `points` and returns an numpy array of length N.
        If real-valued, a float can be returned instead. Example: `Q = lambda x,
        y, z: np.array([x**2 + y**2 - z, x + z])` or `Q = lambda x, y: x - y`.

    Returns
    -------
    (N, d_out) numpy array
        Result of applying `Q` to each point.
    """
    # Input validation
    if points.shape[0] == 0:
        raise ValueError("points must be non-empty.")

    # Make sure (N, ) arrays are treated as (N, 1)
    points = points[:, None] if points.ndim == 1 else points

    try:
        # Fast vectorized version
        q = np.array(Q(*points.T), dtype=float).T
    except ValueError:
        print("Using slower version of apply_map()")
        q = np.array([np.ravel(Q(*p)) for p in points], dtype=float)

    #  Always convert (N, ) arrays to (N, 1)
    return q[:, None] if q.ndim == 1 else q


def make_partition(grid_mins: np.ndarray, grid_maxes: np.ndarray, M: int) -> np.ndarray:
    """
    Partition a hyper-rectangle.

    Along axis i of d-dimensions, build `M + 1` equal-width bin edges spanning
    [`grid_mins[i]`, `grid_maxes[i]`].

    Parameters
    ----------
    grid_mins, grid_maxes : (d, ) numpy array
        Lower and upper bound of the range (inclusive) for each of d dimensions.
        Must satisfy `min_val[i] < max_val[i]` for each dimension i.
    M : int
        Number of bins along an axis (must be >= 1).

    Returns
    -------
    (M + 1, d) numpy array
        Array of bin edges.

    Notes
    -----
    The reason we manually write this function and counts_observed() rather than
    simply calling np.histogram is because we often need to fix the same bin
    edges in advance and compute counts for those bin edges. In general, if we
    call np.histogram on two different datasets, e.g., observed q data and q
    from the prior points, the bins will differ due to different supports.
    """
    # Basic input validation
    if not (isinstance(M, int) and M >= 1):
        raise ValueError("M must be an integer >= 1.")

    # Floats should be packed into arrays
    if isinstance(grid_mins, float):
        grid_mins = np.array([grid_mins])
        grid_maxes = np.array([grid_maxes])

    # Make sure mins/max values of right length, are monotonic
    check_grid_bounds(grid_mins, grid_maxes)

    unscaled_partition = np.linspace(0, 1, M + 1)
    partition_all_dimensions = (
        grid_mins[:, None] + (grid_maxes - grid_mins)[:, None] * unscaled_partition
    )
    return partition_all_dimensions.T


def counts_observed(q_data: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Estimate histogram of observed data.

    Counts the number of values falling into each of M^d bins on a d-dimensional
    rectangle. If no points are given, return 0 for each of the bins.

    Parameters
    ----------
    q_data : (K, d) numpy array
        Array of observed values.
    bin_edges : (M + 1, d) numpy array
        Array of M + 1 bin edges defining the intervals for each dimension.

    Returns
    -------
    counts : (M, M, ..., M) shaped (d-many) numpy array of integers (float type
    possibly).
        Counts for each bin in each dimension. For d = 1, returns (M, ) shaped
        array.
    """
    bin_edges_len, bin_edges_dim = bin_edges.shape
    M = bin_edges_len - 1

    q_data = q_data[:, None] if q_data.ndim == 1 else q_data
    q_data_len, q_data_dim = q_data.shape

    if bin_edges_dim != q_data_dim:
        raise ValueError("Bins or data of wrong dimension.")

    if q_data_len == 0:
        return np.zeros((M,) * q_data_dim, dtype=np.int64)

    # np.histogramdd needs (d, M + 1) bins array input
    counts, _ = np.histogramdd(q_data, bins=bin_edges.T)

    return counts


def counts_validation(counts: np.ndarray) -> None:
    """
    Check that the output of counts_observed are non-negative integers.

    Parameters
    ----------
    counts : numpy array
        (M, ) or (M, M, ..., M) d-times shaped array of counts. The dtype can be
        a float, but the float must be an integer, e.g., 5.0.

    Raises
    ------
    ValueError if any entry of counts is negative or not an integer.
    """
    if not np.array_equal(counts, np.rint(counts)):
        raise ValueError("counts array must be integers.")
    if np.any(counts < 0):
        raise ValueError("counts array cannot contain negative values.")
