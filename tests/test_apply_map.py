"""
Unit tests for `src.apply_map()`.

Tests:
- Evaluation for simple examples of Q
- Output type and shape
- Asymmetric Q (to ensure order of input parameters is correctly handled)
- Input validation: empty, non-2D, wrong column count
- Q output validation: non-1D or wrong length
"""

import numpy as np
import pytest

from src.output_functions import apply_map as apply_map


def Q_add(x, y):
    return x + y


def squared_distance(x, y):
    return x**2 + y**2


def test_apply_map_basic():
    """Computes Q(x,y)=x^2+y^2 for three points; checks shape and values."""
    points = np.array([[0, 0], [1, 2], [3, 4]])
    q_vals = apply_map(points, squared_distance)

    assert q_vals.shape == (3,)
    assert np.allclose(q_vals, [0, 5, 25])


def test_apply_map_vectorized_function():
    """Supports native NumPy functions."""
    points = np.array([[3, 4], [5, 12]])
    Q = np.hypot  # sqrt(x^2 + y^2)
    q_vals = apply_map(points, Q)
    assert np.allclose(q_vals, [5, 13])

    points = np.array([[1, 2], [-1, 4]])
    Q = np.fmax  # max(x, y)
    q_vals = apply_map(points, Q)
    assert np.allclose(q_vals, [2, 4])


def test_apply_map_dtype_and_shape():
    """Returns a 1D float array of length N."""
    rng = np.random.default_rng(0)
    points = rng.random((10, 2))
    q_vals = apply_map(points, Q_add)

    assert q_vals.ndim == 1
    assert q_vals.dtype.kind == "f"
    assert q_vals.shape == (10,)


def test_apply_map_asymmetric_q_small_deterministic():
    """Asymmetric Q catches swapped x/y unpacking."""
    points = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, -5.0], [-1.0, 0.5]])

    def Q(x, y):
        return 2.0 * x + 5.0 * y

    x, y = points[:, 0], points[:, 1]
    expected = 2.0 * x + 5.0 * y
    q_vals = apply_map(points, Q)

    assert q_vals.shape == (len(points),)
    assert np.allclose(q_vals, expected)


def test_apply_map_raises_on_empty_points():
    """Raises when 'points' is empty."""
    empty_points = np.empty((0, 2), dtype=float)

    with pytest.raises(ValueError, match="non-empty"):
        apply_map(empty_points, Q_add)


def test_apply_map_raises_when_not_2d():
    """Raises when 'points' is not a 2D array."""
    not_2d = np.array([1.0, 2.0])  # shape (2,)

    with pytest.raises(ValueError, match="must be 2D array"):
        apply_map(not_2d, Q_add)


def test_apply_map_raises_when_not_two_columns():
    """Raises when `points` does not have shape (N, 2)."""
    points = np.ones((3, 3))
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        apply_map(points, Q_add)


def test_apply_map_raises_when_Q_returns_non_1d():
    """Raises when Q returns a non-1D array."""
    points = np.array([[0.0, 1.0], [2.0, 3.0]])

    def Q_bad(x, y):
        return np.column_stack([x, y])  # shape (N, 2)

    with pytest.raises(RuntimeError, match="must return 1D array"):
        apply_map(points, Q_bad)


def test_apply_map_raises_when_length_mismatch():
    """Raises when Q returns a 1D array of the wrong length."""
    points = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])

    def Q_bad(x, y):
        return x[:-1]  # length N-1

    with pytest.raises(RuntimeError, match="same length as 'points'"):
        apply_map(points, Q_bad)
