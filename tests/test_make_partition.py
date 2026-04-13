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

from src.output_functions import make_partition


def test_make_partition_basic():
    """Edges span [grid_mins, grid_maxes], are monotonic, and count = M+1."""
    grid_mins, grid_maxes = np.array([0.2]), np.array([3.9])
    M = 4
    edges = make_partition(grid_mins, grid_maxes, M)
    assert edges.shape == (M + 1, 1)
    assert np.all(np.diff(edges) > 0)
    assert edges[0] == grid_mins
    assert edges[-1] == grid_maxes


def test_make_partition_basic_higher_dim():
    """Higher dim: Edges span [grid_mins, grid_maxes], are monotonic, and count = M+1."""
    grid_mins, grid_maxes = np.array([0.2, 0.1]), np.array([3.9, 4.1])
    M = 4
    edges = make_partition(grid_mins, grid_maxes, M)

    assert edges.shape == (M + 1, 2)
    assert np.all(np.diff(edges, axis=0) > 0)

    assert np.allclose(edges[0], grid_mins)
    assert np.allclose(edges[-1], grid_maxes)


def test_make_partition_equal_width():
    """Edges match np.linspace for a simple interval."""
    grid_mins, grid_maxes = np.array([0]), np.array([10])
    M = 5
    edges = make_partition(grid_mins, grid_maxes, M)
    expected = np.linspace(grid_mins, grid_maxes, M + 1)
    assert np.allclose(edges, expected)


def test_make_partition_equal_width_higher_dim():
    """Edges match np.linspace for a simple interval."""
    grid_mins, grid_maxes = np.array([0.2, 0.1]), np.array([3.9, 4.1])
    M = 5
    edges = make_partition(grid_mins, grid_maxes, M)
    expected_one = np.linspace(0.2, 3.9, M + 1)
    expected_two = np.linspace(0.1, 4.1, M + 1)
    expected = np.vstack((expected_one, expected_two)).T
    assert np.allclose(edges, expected)


def test_make_partition_with_minmax_from_unsorted_input():
    """Edges computed from min/max of unsorted data are correct."""
    q_data = np.array([2.1, 0.5, 3.9, 1.3, 0.2])
    grid_mins, grid_maxes = float(q_data.min()), float(q_data.max())
    M = 3
    edges = make_partition(np.array([grid_mins]), np.array([grid_maxes]), M)
    assert edges[0] == grid_mins
    assert edges[-1] == grid_maxes
    assert np.all(np.diff(edges) > 0)


def test_make_partition_with_minmax_from_unsorted_input_higher_dim():
    """Higher dim: Edges computed from min/max of unsorted data are correct."""
    q_data = np.array([[2.1, 3], [0.5, 1], [3.9, 5], [1.3, -1], [0.2, 0]])
    grid_mins = np.array([q_data[:, 0].min(), q_data[:, 1].min()])
    grid_maxes = np.array([q_data[:, 0].max(), q_data[:, 1].max()])

    assert np.allclose(grid_mins, np.array([0.2, -1]))
    assert np.allclose(grid_maxes, np.array([3.9, 5]))

    M = 3
    edges = make_partition(grid_mins, grid_maxes, M)
    assert np.all(np.diff(edges, axis=0) > 0)
    assert np.allclose(edges[0], grid_mins)
    assert np.allclose(edges[-1], grid_maxes)


def test_make_partition_single_bin():
    """M=1 yields exactly two edges: [grid_mins, grid_maxes]."""
    grid_mins, grid_maxes = np.array([1]), np.array([5])
    M = 1
    edges = make_partition(grid_mins, grid_maxes, M)
    assert edges.shape == (2, 1)
    assert np.allclose(edges, np.array([[1], [5]]))


def test_make_partition_single_bin_higher_dim():
    """Higher dim: M=1 yields exactly two edges: [grid_mins[i], grid_maxes[i]] in each dimension."""
    grid_mins, grid_maxes = np.array([1, 0]), np.array([5, 10])
    M = 1
    edges = make_partition(grid_mins, grid_maxes, M)
    assert edges.shape == (2, 2)
    assert np.allclose(edges, np.array([[1, 0], [5, 10]]))


def test_make_partition_widths_sum_to_range():
    """Sum of bin widths equals (grid_maxes - grid_mins)."""
    rng = np.random.default_rng(0)
    q_data = rng.uniform(-2.5, 7.5, size=100)
    grid_mins, grid_maxes = float(q_data.min()), float(q_data.max())
    M = 7
    edges = make_partition(np.array([grid_mins]), np.array([grid_maxes]), M)
    total_width = float(np.sum(np.diff(edges, axis=0)))
    assert np.isclose(total_width, grid_maxes - grid_mins)


def test_make_partition_raises_when_min_not_less_than_max_equal():
    """Raises when grid_mins == grid_maxes."""
    with pytest.raises(ValueError, match="grid_mins >= grid_maxes"):
        make_partition(np.array([1]), np.array([1]), 3)


def test_make_partition_raises_when_min_not_less_than_max_equal_higher_dim():
    """Raises when grid_mins == grid_maxes."""
    with pytest.raises(ValueError, match="grid_mins >= grid_maxes"):
        make_partition(np.array([1, 2]), np.array([1, 3]), 3)

    # Same thing but swap components with error
    with pytest.raises(ValueError, match="grid_mins >= grid_maxes"):
        make_partition(np.array([1, 2]), np.array([2, 2]), 3)


def test_make_partition_raises_when_min_not_less_than_max_reversed():
    """Raises when grid_mins > grid_maxes."""
    with pytest.raises(ValueError, match="grid_mins >= grid_maxes"):
        make_partition(np.array([5]), np.array([1]), 3)


def test_make_partition_raises_when_min_not_less_than_max_reversed_higher_dim():
    """Raises when grid_mins > grid_maxes."""
    with pytest.raises(ValueError, match="grid_mins >= grid_maxes"):
        make_partition(np.array([5, 7]), np.array([1, 8]), 3)

    # Same thing but swap which component draws the error
    with pytest.raises(ValueError, match="grid_mins >= grid_maxes"):
        make_partition(np.array([5, 7]), np.array([7, 6]), 5)


def test_make_partition_raises_when_M_less_than_1():
    """Raises when M < 1."""
    with pytest.raises(ValueError, match="must be an integer >= 1"):
        make_partition(np.array([0]), np.array([1]), 0)


def test_make_partition_raises_when_M_not_int():
    """Raises when M is not an integer."""
    with pytest.raises(ValueError, match="must be an integer >= 1"):
        # Use cast to pretend 0.5 is an integer, otherwise type checker will flag warning
        make_partition(np.array([0]), np.array([1]), cast(int, 0.5))
