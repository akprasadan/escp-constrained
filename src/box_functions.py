"""
box_functions.py

Helper functions to grid up the input space into sub-rectangles and subset points by rectangle.
"""

from collections.abc import Callable

import numpy as np


def check_lambda_bounds(xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    """Utility function that validates input bounds in Lambda space.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Specify bounds of a rectangle [xmin, xmax] X [ymin, ymax]

    Raises
    --------
    ValueError : If bounds are invalid.
    """
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("Require xmin < xmax and ymin < ymax.")


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
    if points.shape[0] == 0:
        raise ValueError("points must be non-empty.")

    if A is None:
        return np.ones(points.shape[0], dtype=bool)
    return A(points)


def rect_A(
    xmin: float, xmax: float, ymin: float, ymax: float
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Produce rectangular mask with half-open bounds:
    A = [xmin, xmax) x [ymin, ymax)

    Excluding right/top edges of rectangle prevents double-counting when constructing grid.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Rectangle bounds; require xmin < xmax and ymin < ymax.

    Returns
    -------
    _A : Callable[[np.ndarray], np.ndarray]
        Returns boolean mask that acts on a numpy array of shape (N, 2)
        and returns a numpy array of bools of shape (N, ), indicating
        whether each point belongs to the rectangle.
    """
    # Check we input valid bounds
    check_lambda_bounds(xmin, xmax, ymin, ymax)

    def _A(points: np.ndarray) -> np.ndarray:
        x_vals, y_vals = points[:, 0], points[:, 1]

        return (xmin <= x_vals) & (x_vals < xmax) & (ymin <= y_vals) & (y_vals < ymax)

    return _A
