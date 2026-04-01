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
    """Edges span [min_val, max_val], are monotonic, and count = M+1."""
    min_val, max_val = 0.2, 3.9
    M = 4
    edges = make_partition(min_val, max_val, M)
    assert edges.shape == (M + 1,)
    assert np.all(np.diff(edges) > 0)
    assert edges[0] == min_val
    assert edges[-1] == max_val


def test_make_partition_equal_width():
    """Edges match np.linspace for a simple interval."""
    min_val, max_val = 0.0, 10.0
    M = 5
    edges = make_partition(min_val, max_val, M)
    expected = np.linspace(min_val, max_val, M + 1)
    assert np.allclose(edges, expected)


def test_make_partition_with_minmax_from_unsorted_input():
    """Edges computed from min/max of unsorted data are correct."""
    q_data = np.array([2.1, 0.5, 3.9, 1.3, 0.2])
    min_val, max_val = float(q_data.min()), float(q_data.max())
    M = 3
    edges = make_partition(min_val, max_val, M)
    assert edges[0] == min_val
    assert edges[-1] == max_val
    assert np.all(np.diff(edges) > 0)


def test_make_partition_single_bin():
    """M=1 yields exactly two edges: [min_val, max_val]."""
    min_val, max_val = 1.0, 5.0
    M = 1
    edges = make_partition(min_val, max_val, M)
    assert edges.shape == (2,)
    assert np.allclose(edges, [min_val, max_val])


def test_make_partition_widths_sum_to_range():
    """Sum of bin widths equals (max_val - min_val)."""
    rng = np.random.default_rng(0)
    q_data = rng.uniform(-2.5, 7.5, size=100)
    min_val, max_val = float(q_data.min()), float(q_data.max())
    M = 7
    edges = make_partition(min_val, max_val, M)
    total_width = float(np.sum(np.diff(edges)))
    assert np.isclose(total_width, max_val - min_val)


def test_make_partition_raises_when_min_not_less_than_max_equal():
    """Raises when min_val == max_val."""
    with pytest.raises(ValueError, match="min_val < max_val"):
        make_partition(1.0, 1.0, 3)


def test_make_partition_raises_when_min_not_less_than_max_reversed():
    """Raises when min_val > max_val."""
    with pytest.raises(ValueError, match="min_val < max_val"):
        make_partition(5.0, 1.0, 3)


def test_make_partition_raises_when_M_less_than_1():
    """Raises when M < 1."""
    with pytest.raises(ValueError, match="must be an integer >= 1"):
        make_partition(0.0, 1.0, 0)


def test_make_partition_raises_when_M_not_int():
    """Raises when M is not an integer."""
    with pytest.raises(ValueError, match="must be an integer >= 1"):
        # Use cast to pretend 0.5 is an integer, otherwise type checker will flag warning
        make_partition(0.0, 1.0, cast(int, 0.5))
