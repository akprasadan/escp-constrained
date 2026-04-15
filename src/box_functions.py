"""
box_functions.py

Helper functions to grid up the input space into sub-rectangles and subset points by rectangle.
"""

from collections.abc import Callable

import numpy as np
from numba import njit


def check_grid_bounds(grid_mins: np.ndarray, grid_maxes: np.ndarray) -> None:
    """Utility function that validates input bounds of hyper-rectangle, usually
    either Lambda or D.

    Parameters
    ----------
    grid_mins, grid_maxes : np.ndarray
        Rectangle bounds; require grid_mins[i] < grid_maxes[i] for each i.

    Raises
    --------
    ValueError : If bounds are invalid.
    """

    d = len(grid_mins)

    if d != len(grid_maxes):
        raise ValueError(
            "Incompatible lengths (dimension) for grid_mins and grid_maxes."
        )

    if np.any(grid_mins >= grid_maxes):
        raise ValueError(f"grid_mins >= grid_maxes for some dimension.")


def mask_in_A(
    points: np.ndarray, A: Callable[[np.ndarray], np.ndarray] | None = None
) -> np.ndarray:
    """
    Compute a boolean mask indicating which points lie in a subset A of R^d.

    Parameters
    ----------
    points : (N, d) numpy array
        Array of N d-dimensional points to test for membership.
    A : Callable[[np.ndarray], np.ndarray] or None
        A mask defining the subset A. It must accept `points`
        (shape (N, d)) and return a boolean array of shape (N, ) where True
        means the point is in A.
        If A is None, all points are considered inside A.

    Returns
    -------
    mask : (N, ) numpy array of dtype bool
        Boolean mask where True indicates the corresponding point is in A.

    Examples
    --------
    >>> points = np.array([[0.1, 0.2], [0.6, 0.7]])
    >>> A = lambda pts: pts[:, 0] <= 0.5  # select points with x <= 0.5
    >>> mask_in_A(points, A)
    array([True, False])
    """
    if len(points) == 0:
        raise ValueError("points must be non-empty.")

    if A is None:
        return np.ones(points.shape[0], dtype=bool)
    return A(points)


@njit
def rect_A_kernel(
    points: np.ndarray, grid_mins: np.ndarray, grid_maxes: np.ndarray
) -> np.ndarray:
    """Returns a numpy array of boolean values, indicating if each element of points
    is within the hyper-rectangle specified by grid_mins and grid_maxes. It is written
    using for loops to work with numba for 2-3x speed-up versus using, say,
    np.all(mask, axis = 1).

    Parameters
    ----------
    points : (N, d) numpy array
        Array of N d-dimensional points to test for membership.
    grid_mins, grid_maxes : np.ndarray
        Rectangle bounds; require grid_mins[i] < grid_maxes[i] for each i.

    Returns
    -------
    mask : (N, ) numpy array of dtype bool
        Boolean mask where True indicates the corresponding point is in the hyper-rectangle.
    """
    n, d = points.shape
    out = np.ones(n, dtype=np.bool)

    for i in range(n):
        for j in range(d):
            point = points[i, j]
            if point < grid_mins[j] or point >= grid_maxes[j]:
                out[i] = False
                break
    return out


def rect_A(
    grid_mins: np.ndarray, grid_maxes: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Produce d-dimensional rectangular mask with half-open bounds:
    A = [x1_min, x1_max) x [x2_min, x2_max) x ... x [xd_min, xd_max]

    Excluding right/top edges of rectangle prevents double-counting when constructing grid.

    Parameters
    ----------
    grid_mins, grid_maxes : np.ndarray
        Rectangle bounds; require grid_mins[i] < grid_maxes[i] for each i.

    Returns
    -------
    _A : Callable[[np.ndarray], np.ndarray]
        Returns boolean mask that acts on a numpy array of shape (N, )
        and returns a numpy array of bools of shape (N, ), indicating
        whether each point belongs to the rectangle.
    """
    # Check we input valid bounds
    check_grid_bounds(grid_mins, grid_maxes)

    def _A(points):
        # (N, ) points should be (N, 1)
        points = points[:, None] if points.ndim == 1 else points
        # Use numba sped-up version
        return rect_A_kernel(points, grid_mins, grid_maxes)

    return _A
