"""
Unit tests for `src.output_functions.py::apply_map()`.

Tests:

- Evaluation for simple examples of Q
- Output type and shape
- Asymmetric Q (to ensure order of input parameters is correctly handled)
- Input validation: empty, non-2D, wrong column count
- Q output validation: non-1D or wrong length
"""

import numpy as np
import pytest

from src.output_functions import apply_map


def Q_scale(x: np.ndarray) -> np.ndarray:
    """Make simple scale function on R.

    Returns
    -------
    numpy array
    """
    return 3 * x


def Q_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Make simple add function on R^2.

    Returns
    -------
    numpy array
    """
    return x + y


def Q_dim3_to_2(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Make simple R^3 --> R^2 map.

    Returns
    -------
    numpy array
    """
    return np.array([x + y - z, x + z])


def squared_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Squared euclidean distance in R^2.

    Returns
    -------
    numpy array
    """
    return x**2 + y**2


def non_vectorizeable_Q(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Create a Q that cannot be vectorized.

    Returns
    -------
    numpy array
    """
    if x > 0:
        return x
    return y


def test_apply_map_one_D() -> None:
    """Test 1D behavior only using Q_scale(x) = 3x.

    Input should work whether with (N, ) or (N, 1) points. Output should always
    be of form (N, 1) though.
    """
    points = np.array([-2, 0, 1, 4, 0.5])
    q_vals = apply_map(points, Q_scale)

    # Basic shape, type checks
    assert q_vals.shape == (5, 1)
    assert q_vals.dtype.kind == "f"

    expected = np.array([-6, 0, 3, 12, 1.5])[:, None]
    assert np.allclose(q_vals, expected)

    # Check degenerate input
    empty_points = np.array([])
    with pytest.raises(ValueError, match="non-empty"):
        apply_map(empty_points, Q_scale)

    # points but as (N, 1) instead of (N, )
    points_new = np.array([-2, 0, 1, 4, 0.5])[:, None]
    q_vals_new = apply_map(points_new, Q_scale)
    assert np.allclose(q_vals_new, expected)


def test_apply_map_basic() -> None:
    """Computes Q(x,y)=x^2+y^2 for three points; checks shape and values."""
    points = np.array([[0, 0], [1, 2], [3, 4]])
    q_vals = apply_map(points, squared_distance)
    expected = np.array([0, 5, 25])[:, None]
    assert q_vals.shape == (3, 1)
    assert np.allclose(q_vals, expected)


def test_apply_map_3_to_2_dim_q() -> None:
    """Computes Q(x,y,z)=(x+y-z, x+z); checks shape and values."""
    points = np.array([[0, 0, 1], [1, 2, 2], [3, 4, 1]])
    q_vals = apply_map(points, Q_dim3_to_2)

    assert q_vals.shape == (3, 2)
    assert np.allclose(q_vals, [[-1, 1], [1, 3], [6, 4]])


def test_apply_map_3_to_2_dim_q_again() -> None:
    """Computes Q(x,y,z)=(2x+5y-z, z); checks shape and values."""
    points = np.array([[0, 1, 2], [2, 3, 4], [4, -5, 1], [-1, 0.5, 0]])

    def Q(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.array([2 * x + 5 * y - z, z])

    expected = np.array([[3, 2], [15, 4], [-18, 1], [0.5, 0]])
    q_vals = apply_map(points, Q)

    assert q_vals.shape == (len(points), 2)
    assert np.allclose(q_vals, expected)


def test_apply_map_3_to_2_dim_q_again_swapped_output() -> None:
    """Q(x,y,z)=(2x+5y-z, z) but with swapped output components."""
    points = np.array([[0, 1, 2], [2, 3, 4], [4, -5, 1], [-1, 0.5, 0]])

    def Q(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.array([z, 2 * x + 5 * y - z])

    expected = np.array([[2, 3], [4, 15], [1, -18], [0, 0.5]])
    q_vals = apply_map(points, Q)

    assert q_vals.shape == (len(points), 2)
    assert np.allclose(q_vals, expected)


def test_apply_map_vectorized_function() -> None:
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


def test_apply_map_dtype_and_shape() -> None:
    """Q_add returns a 1d-dimensional float array of length N."""
    rng = np.random.default_rng(0)
    points = rng.random((10, 2))
    q_vals = apply_map(points, Q_add)

    assert q_vals.dtype.kind == "f"
    assert q_vals.shape == (10, 1)


def test_apply_map_dtype_shape_higher_dim() -> None:
    """Q_dim3_to_2 returns a 2d-dimensional float array of length N."""
    rng = np.random.default_rng(0)
    points_high_dim = rng.random((10, 3))
    q_vals_high_dim = apply_map(points_high_dim, Q_dim3_to_2)

    assert q_vals_high_dim.ndim == 2
    assert q_vals_high_dim.dtype.kind == "f"
    assert q_vals_high_dim.shape == (10, 2)


def test_apply_map_asymmetric_q() -> None:
    """Asymmetric Q catches swapped x/y unpacking."""
    points = np.array([[0, 1], [2, 3], [4, -5], [-1, 0.5]])

    def Q(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2 * x + 5 * y

    expected = np.array([[5], [19], [-17], [0.5]])
    q_vals = apply_map(points, Q)

    assert q_vals.shape == (len(points), 1)
    assert np.allclose(q_vals, expected)


def test_apply_map_raises_on_empty_points() -> None:
    """Raises when 'points' is empty."""
    empty_points = np.empty((0, 2), dtype=float)

    with pytest.raises(ValueError, match="non-empty"):
        apply_map(empty_points, Q_add)


def test_apply_map_non_vectorizeable_map(capfd: pytest.CaptureFixture[str]) -> None:
    """Test a Q that cannot act on vector inputs.

    The previous iteration of apply_map used:

        q = np.array(Q(*points.T), dtype=float).T

    This would fail if Q is of the below type, since "x > 0" only works with
    scalars. Now we try the previous line, otherwise use a (slower) list
    comprehension.

    Parameters
    ----------
    capfd : pytest.CaptureFixture[str]
        Pytest fixture to ensure warning of slow map is caught
    """
    points = np.array([[0, 1], [2, 3], [-4, -5], [-1, 0.5]])

    expected = np.array([1, 2, -5, 0.5])[:, None]
    q_vals = apply_map(points, non_vectorizeable_Q)
    out, _ = capfd.readouterr()
    assert "Using slower version of apply_map()" in out

    assert q_vals.shape == (len(points), 1)
    assert np.allclose(q_vals, expected)


def test_apply_map_with_different_output_types() -> None:
    """
    Check returning floats or numpy arrays is fine if Q real-valued.

    Check 2D -> 1D map and 1D -> 1D map.
    """

    def Q_add_but_vector(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.array([x + y])

    def Q_add_but_vector_v2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.atleast_1d(x + y)

    rng = np.random.default_rng(0)
    points = rng.random((10, 2))
    q_vals = apply_map(points, Q_add)  # uses float output
    q_vals_new = apply_map(points, Q_add_but_vector)
    q_vals_new_v2 = apply_map(points, Q_add_but_vector_v2)

    assert np.allclose(q_vals, q_vals_new)
    assert np.allclose(q_vals, q_vals_new_v2)

    def Q_scale_but_vector(x: np.ndarray) -> np.ndarray:
        return np.array([3 * x])

    def Q_scale_but_vector_v2(x: np.ndarray) -> np.ndarray:
        return np.atleast_1d(3 * x)

    rng = np.random.default_rng(0)
    points = rng.random((10, 1))
    q_vals = apply_map(points, Q_scale)  # uses float output
    q_vals_new = apply_map(points, Q_scale_but_vector)
    q_vals_new_v2 = apply_map(points, Q_scale_but_vector_v2)
    assert np.allclose(q_vals, q_vals_new)
    assert np.allclose(q_vals, q_vals_new_v2)


def test_apply_map_augmented_Q(capfd: pytest.CaptureFixture[str]) -> None:
    """Test that augmentation of map behaves as expected.

    The construction in general is:

    def augmented_Q(a, b, ..., y, z) -> np.ndarray:
        q = original_Q(a, b, ..., y) return np.vstack((q, z))

    Parameters
    ----------
    capfd : pytest.CaptureFixture[str]
        Pytest fixture to ensure warning of slow map is caught
    """

    def augmented_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        q = Q_add(x, y)
        return np.vstack((q, y))

    points = np.array([[1, 2], [2, 3], [1, 1], [0, 1], [0, 0]])
    q_vals = apply_map(points, augmented_add)
    expected = np.array([[3, 2], [5, 3], [2, 1], [1, 1], [0, 0]])
    assert np.allclose(q_vals, expected)

    def augmented_higher_dim(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        # recall Q_dim3_to_2: (x, y, z) --> (x + y - z, x + z)
        q = Q_dim3_to_2(x, y, z)
        return np.vstack((q, z))

    points = np.array([[1, 2, 1], [2, 3, 3], [1, 1, 0], [0, 1, 1], [0, 0, 0]])
    q_vals = apply_map(points, augmented_higher_dim)
    expected = np.array([[2, 2, 1], [2, 5, 3], [2, 1, 0], [0, 1, 1], [0, 0, 0]])
    assert np.allclose(q_vals, expected)

    # Check this this works for vector output version of real-valued Q_add
    def Q_add_but_vector(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.array([x + y])

    def augmented_add_but_vector(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        q = Q_add_but_vector(x, y)
        return np.vstack((q, y))

    points = np.array([[1, 2], [2, 3], [1, 1], [0, 1], [0, 0]])
    q_vals = apply_map(points, augmented_add_but_vector)
    expected = np.array([[3, 2], [5, 3], [2, 1], [1, 1], [0, 0]])
    assert np.allclose(q_vals, expected)

    # Do a one with intentionally un-vectorizeable format
    def augmented_non_vectorizeable(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        q = non_vectorizeable_Q(x, y)
        return np.vstack((q, y))

    points = np.array([[0, 1], [2, 3], [-4, -5], [-1, 0.5]])

    # A warning should be flagged for using slow evaluation
    q_vals = apply_map(points, augmented_non_vectorizeable)

    out, _ = capfd.readouterr()
    assert "Using slower version of apply_map()" in out

    expected = np.array([[1, 1], [2, 3], [-5, -5], [0.5, 0.5]])
    assert np.allclose(q_vals, expected)
