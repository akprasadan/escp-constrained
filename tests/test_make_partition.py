"""
Unit tests for `output_functions.make_partition()`.

Tests:
- Basic properties: correct number of edges, monotonicity, and bounds
- Equal-width spacing over known intervals
- Behavior with min/max derived from unsorted input
- Single-bin case (M=1)
- Sum of bin widths equals full range
- Input validation for invalid bounds and invalid M
"""

from typing import cast

import numpy as np
import pytest

from src.output_functions import make_partition as make_partition


def test_make_partition_basic():
    """Edges span [min_vals, max_vals], are monotonic, and count = M+1."""
    min_vals, max_vals = np.array([0.2]), np.array([3.9])
    M = 4
    edges = make_partition(min_vals, max_vals, M)
    assert edges.shape == (M + 1, 1)
    assert np.all(np.diff(edges) > 0)
    assert edges[0] == min_vals
    assert edges[-1] == max_vals


def test_make_partition_basic_higher_dim():
    """Higher dim: Edges span [min_vals, max_vals], are monotonic, and count = M+1."""
    min_vals, max_vals = np.array([0.2, 0.1]), np.array([3.9, 4.1])
    M = 4
    edges = make_partition(min_vals, max_vals, M)

    assert edges.shape == (M + 1, 2)
    assert np.all(np.diff(edges, axis=0) > 0)

    assert np.allclose(edges[0], min_vals)
    assert np.allclose(edges[-1], max_vals)


def test_make_partition_equal_width():
    """Edges match np.linspace for a simple interval."""
    min_vals, max_vals = np.array([0.0]), np.array([10.0])
    M = 5
    edges = make_partition(min_vals, max_vals, M)
    expected = np.linspace(min_vals, max_vals, M + 1)
    assert np.allclose(edges, expected)


def test_make_partition_equal_width_higher_dim():
    """Edges match np.linspace for a simple interval."""
    min_vals, max_vals = np.array([0.2, 0.1]), np.array([3.9, 4.1])
    M = 5
    edges = make_partition(min_vals, max_vals, M)
    expected_one = np.linspace(0.2, 3.9, M + 1)
    expected_two = np.linspace(0.1, 4.1, M + 1)
    expected = np.vstack((expected_one, expected_two)).T
    assert np.allclose(edges, expected)


def test_make_partition_with_minmax_from_unsorted_input():
    """Edges computed from min/max of unsorted data are correct."""
    q_data = np.array([2.1, 0.5, 3.9, 1.3, 0.2])
    min_vals, max_vals = float(q_data.min()), float(q_data.max())
    M = 3
    edges = make_partition(np.array([min_vals]), np.array([max_vals]), M)
    assert edges[0] == min_vals
    assert edges[-1] == max_vals
    assert np.all(np.diff(edges) > 0)


def test_make_partition_with_minmax_from_unsorted_input_higher_dim():
    """Higher dim: Edges computed from min/max of unsorted data are correct."""
    q_data = np.array([[2.1, 3], [0.5, 1], [3.9, 5], [1.3, -1], [0.2, 0]])
    min_vals = np.array([q_data[:, 0].min(), q_data[:, 1].min()])
    max_vals = np.array([q_data[:, 0].max(), q_data[:, 1].max()])

    assert np.allclose(min_vals, np.array([0.2, -1]))
    assert np.allclose(max_vals, np.array([3.9, 5]))

    M = 3
    edges = make_partition(min_vals, max_vals, M)
    assert np.all(np.diff(edges, axis=0) > 0)
    assert np.allclose(edges[0], min_vals)
    assert np.allclose(edges[-1], max_vals)


def test_make_partition_single_bin():
    """M=1 yields exactly two edges: [min_vals, max_vals]."""
    min_vals, max_vals = np.array([1.0]), np.array([5.0])
    M = 1
    edges = make_partition(min_vals, max_vals, M)
    assert edges.shape == (2, 1)
    assert np.allclose(edges, np.array([[1.0], [5.0]]))


def test_make_partition_single_bin_higher_dim():
    """Higher dim: M=1 yields exactly two edges: [min_vals[i], max_vals[i]] in each dimension."""
    min_vals, max_vals = np.array([1.0, 0.0]), np.array([5.0, 10.0])
    M = 1
    edges = make_partition(min_vals, max_vals, M)
    assert edges.shape == (2, 2)
    assert np.allclose(edges, np.array([[1.0, 0.0], [5.0, 10.0]]))


def test_make_partition_widths_sum_to_range():
    """Sum of bin widths equals (max_vals - min_vals)."""
    rng = np.random.default_rng(0)
    q_data = rng.uniform(-2.5, 7.5, size=100)
    min_vals, max_vals = float(q_data.min()), float(q_data.max())
    M = 7
    edges = make_partition(np.array([min_vals]), np.array([max_vals]), M)
    total_width = float(np.sum(np.diff(edges, axis=0)))
    assert np.isclose(total_width, max_vals - min_vals)


def test_make_partition_raises_when_min_not_less_than_max_equal():
    """Raises when min_vals == max_vals."""
    with pytest.raises(ValueError, match="min_vals < max_vals"):
        make_partition(np.array([1.0]), np.array([1.0]), 3)


def test_make_partition_raises_when_min_not_less_than_max_equal_higher_dim():
    """Raises when min_vals == max_vals."""
    with pytest.raises(ValueError, match="min_vals < max_vals"):
        make_partition(np.array([1.0, 2.0]), np.array([1.0, 3.0]), 3)

    # Same thing but swap components with error
    with pytest.raises(ValueError, match="min_vals < max_vals"):
        make_partition(np.array([1.0, 2.0]), np.array([2.0, 2.0]), 3)


def test_make_partition_raises_when_min_not_less_than_max_reversed():
    """Raises when min_vals > max_vals."""
    with pytest.raises(ValueError, match="min_vals < max_vals"):
        make_partition(np.array([5.0]), np.array([1.0]), 3)


def test_make_partition_raises_when_min_not_less_than_max_reversed_higher_dim():
    """Raises when min_vals > max_vals."""
    with pytest.raises(ValueError, match="min_vals < max_vals"):
        make_partition(np.array([5.0, 7.0]), np.array([1.0, 8.0]), 3)

    # Same thing but swap which component draws the error
    with pytest.raises(ValueError, match="min_vals < max_vals"):
        make_partition(np.array([5.0, 7.0]), np.array([7.0, 6.0]), 5)


def test_make_partition_raises_when_M_less_than_1():
    """Raises when M < 1."""
    with pytest.raises(ValueError, match="must be an integer >= 1"):
        make_partition(np.array([0.0]), np.array([1.0]), 0)


def test_make_partition_raises_when_M_not_int():
    """Raises when M is not an integer."""
    with pytest.raises(ValueError, match="must be an integer >= 1"):
        # Use cast to pretend 0.5 is an integer, otherwise type checker will flag warning
        make_partition(np.array([0.0]), np.array([1.0]), cast(int, 0.5))
