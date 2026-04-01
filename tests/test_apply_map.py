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


def Q_add_high_dim(x, y, z):
    return [x + y - z, x + z]


def squared_distance(x, y):
    return x**2 + y**2


def test_apply_map_basic():
    """Computes Q(x,y)=x^2+y^2 for three points; checks shape and values."""
    points = np.array([[0, 0], [1, 2], [3, 4]])
    q_vals = apply_map(points, squared_distance)
    expected = np.array([[0], [5], [25]])
    assert q_vals.shape == (3, 1)
    assert np.allclose(q_vals, expected)


def test_apply_map_3_to_2_dim_q():
    """Computes Q(x,y,z)=(x+y-z, x+z); checks shape and values."""
    points = np.array([[0, 0, 1], [1, 2, 2], [3, 4, 1]])
    q_vals = apply_map(points, Q_add_high_dim)

    assert q_vals.shape == (3, 2)
    assert np.allclose(q_vals, [[-1, 1], [1, 3], [6, 4]])


def test_apply_map_3_to_2_dim_q_again():
    """Computes Q(x,y,z)=(2x+5y-z, z); checks shape and values."""
    points = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, -5.0, 1.0], [-1.0, 0.5, 0]])

    def Q(x, y, z):
        return [2.0 * x + 5.0 * y - z, z]

    expected = np.array([[3, 2], [15, 4], [-18, 1], [0.5, 0]])
    q_vals = apply_map(points, Q)

    assert q_vals.shape == (len(points), 2)
    assert np.allclose(q_vals, expected)


def test_apply_map_3_to_2_dim_q_again_swapped_output():
    """Q(x,y,z)=(2x+5y-z, z) but with swapped output components."""
    points = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, -5.0, 1.0], [-1.0, 0.5, 0]])

    def Q(x, y, z):
        return [z, 2.0 * x + 5.0 * y - z]

    expected = np.array([[2, 3], [4, 15], [1, -18], [0, 0.5]])
    q_vals = apply_map(points, Q)

    assert q_vals.shape == (len(points), 2)
    assert np.allclose(q_vals, expected)


def test_apply_map_vectorized_function():
    """Supports native NumPy functions."""
    points = np.array([[3, 4], [5, 12]])
    Q = np.hypot  # sqrt(x^2 + y^2)
    q_vals = apply_map(points, Q)
    expected = np.array([[5], [13]])
    assert np.allclose(q_vals, expected)

    points = np.array([[1, 2], [-1, 4]])
    Q = np.fmax  # max(x, y)
    q_vals = apply_map(points, Q)
    expected = np.array([[2], [4]])
    assert np.allclose(q_vals, expected)


def test_apply_map_dtype_and_shape():
    """Returns a 1d-dimensional float array of length N."""
    rng = np.random.default_rng(0)
    points = rng.random((10, 2))
    q_vals = apply_map(points, Q_add)

    assert q_vals.dtype.kind == "f"
    assert q_vals.shape == (10, 1)


def test_apply_map_dtype_shape_higher_dim():
    """Returns a 2d-dimensional float array of length N."""

    rng = np.random.default_rng(0)
    points_high_dim = rng.random((10, 3))
    q_vals_high_dim = apply_map(points_high_dim, Q_add_high_dim)

    assert q_vals_high_dim.ndim == 2
    assert q_vals_high_dim.dtype.kind == "f"
    assert q_vals_high_dim.shape == (10, 2)


def test_apply_map_asymmetric_q():
    """Asymmetric Q catches swapped x/y unpacking."""
    points = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, -5.0], [-1.0, 0.5]])

    def Q(x, y):
        return 2.0 * x + 5.0 * y

    x, y = points[:, 0], points[:, 1]
    expected = np.array([[5], [19], [-17], [0.5]])
    q_vals = apply_map(points, Q)

    assert q_vals.shape == (len(points), 1)
    assert np.allclose(q_vals, expected)


def test_apply_map_raises_on_empty_points():
    """Raises when 'points' is empty."""
    empty_points = np.empty((0, 2), dtype=float)

    with pytest.raises(ValueError, match="non-empty"):
        apply_map(empty_points, Q_add)


def test_apply_map_raises_when_length_mismatch():
    """Raises when Q returns a 1D array of the wrong length."""
    points = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])

    def Q_bad(x, y):
        return x[:-1]  # length N-1

    with pytest.raises(RuntimeError, match="same length as 'points'"):
        apply_map(points, Q_bad)
