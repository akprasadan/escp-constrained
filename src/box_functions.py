"""
box_functions.py

Helper functions to grid up the input space into sub-rectangles and subset points by rectangle.
"""

from collections.abc import Callable

import numpy as np


def check_lambda_bounds(min_vals: np.ndarray, max_vals: np.ndarray) -> None:
    """Utility function that validates input bounds in Lambda (or D) space.

    Parameters
    ----------
    mins, maxes : np.ndarray
        Rectangle bounds; require mins[i] < maxes[i] for each i.

    Raises
    --------
    ValueError : If bounds are invalid.
    """
    d = len(min_vals)

    if d != len(max_vals):
        raise ValueError("Incompatible lengths (dimension) for min_vals and max_vals.")

    for i in range(d):
        if not (min_vals[i] < max_vals[i]):
            raise ValueError(f"min_vals >= max_vals for dimension {i}.")


def mask_in_A(
    points: np.ndarray, A: Callable[[np.ndarray], np.ndarray] | None = None
) -> np.ndarray:
    """
    Compute a boolean mask indicating which points lie in a subset A of R^2.

    Parameters
    ----------
    points : (N, 2) numpy array
        Array of N 2D points to test for membership.
    A : Callable[[np.ndarray], np.ndarray] or None
        A mask defining the subset A. It must accept `points`
        (shape (N, 2)) and return a boolean array of shape (N, ) where True
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


def rect_A(mins: np.ndarray, maxes: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Produce d-dimensional rectangular mask with half-open bounds:
    A = [x1_min, x1_max) x [x2_min, x2_max) x ... x [xd_min, xd_max]

    Excluding right/top edges of rectangle prevents double-counting when constructing grid.

    Parameters
    ----------
    mins, maxes : np.ndarray
        Rectangle bounds; require mins[i] < maxes[i] for each i.

    Returns
    -------
    _A : Callable[[np.ndarray], np.ndarray]
        Returns boolean mask that acts on a numpy array of shape (N, )
        and returns a numpy array of bools of shape (N, ), indicating
        whether each point belongs to the rectangle.
    """
    # Check we input valid bounds
    check_lambda_bounds(mins, maxes)

    def _A(points: np.ndarray) -> np.ndarray:
        return np.all((points >= mins) & (points < maxes), axis=1)

    return _A
