"""
Unit tests for `src.counts_observed()`.

Tests:
- Bin counting for simple cases
- Empty bins or empty input
- Check behavior at bin edges or corners
- When all values fall in the last bin
- Output is of correct shape
"""

import numpy as np

from src.output_functions import counts_observed as counts_observed


def test_counts_observed_basic():
    """Values are counted into bins correctly; last bin includes right edge."""
    q_data = np.array([0.1, 0.5, 1.2, 1.8, 2])
    bin_edges = np.array([[0], [1], [2], [3]])  # bins: [0,1), [1,2), [2,3]
    counts = counts_observed(q_data, bin_edges)
    assert counts.tolist() == [2, 2, 1]


def test_counts_observed_basic_higher_dim():
    """Higher dim: Values are counted into bins correctly; last bin includes right edge."""
    q_data = np.array([[0.1, 10], [0.5, 8], [1.2, 6], [1.8, 4], [2, 1.1]])

    # Create 3 x 3 grid of bins
    # bins: [0, 1) x [1, 2.5), [0, 1) x [2.5, 5), [0, 1) x [5, 10]
    # bins: [1, 2) x [1, 2.5), [1, 2) x [2.5, 5), [1, 2) x [5, 10]
    # bins: [2, 3] x [1, 2.5), [2, 3] x [2.5, 5), [2, 3] x [5, 10]
    bin_edges = np.array([[0, 1], [1, 2.5], [2, 5], [3, 10]])

    # Should get 3 x 3 array
    counts = counts_observed(q_data, bin_edges)
    expected = np.array([[0, 0, 2], [0, 1, 1], [1, 0, 0]])
    assert np.allclose(counts, expected)


def test_counts_observed_empty_bin():
    """Bins with no values report zero counts."""
    q_data = np.array([0.2, 0.3])  # both in first bin
    bin_edges = np.array([[0], [0.5], [1]])
    counts = counts_observed(q_data, bin_edges)
    assert counts.tolist() == [2, 0]


def test_counts_observed_empty_bin_higher_dim():
    """Higher dim: Bins with no values report zero counts."""
    q_data = np.array([[0.2, 5], [0.3, 5], [0.1, 4.5]])  # All data in the [0, 0.5) x [4, 7] bin

    # Create 3 x 3 grid of bins
    # bins: [0, 0.5) x [0, 4), [0, 0.5) x [4, 7]
    # bins: [0.5, 1] x [0, 4), [0.5, 1] x [4, 7]
    bin_edges = np.array([[0, 0], [0.5, 4], [1, 7]])

    counts = counts_observed(q_data, bin_edges)
    expected = np.array([[0, 3], [0, 0]])
    assert np.allclose(counts, expected)


def test_counts_observed_empty_list_returns_zeros():
    """Empty input returns zeros for all bins."""
    bin_edges = np.array([[-1], [0], [0.5], [2]])  # 3 bins
    counts = counts_observed(np.array([]), bin_edges)
    M = len(bin_edges) - 1
    assert counts.shape == (M,)
    assert counts.tolist() == [0] * M


def test_counts_observed_empty_list_returns_zeros_higher_dim():
    """Higher dim: Empty input returns zeros for all bins."""
    bin_edges = np.array([[0, 0], [0.5, 4], [1, 7]])
    counts = counts_observed(np.empty((5, 2)), bin_edges)

    expected = np.zeros((2, 2))
    assert np.allclose(counts, expected)


def test_counts_observed_edges_inclusive():
    """Last bin includes its right edge; others are half-open."""
    q_data = np.array([0, 1, 2])
    bin_edges = np.array([[0], [1], [2]])
    counts = counts_observed(q_data, bin_edges)
    assert counts.tolist() == [1, 2]


def test_counts_observed_all_in_last_bin():
    """All values at or above last bin's left edge go into last bin."""
    q_data = np.array([1, 5, 10])
    bin_edges = np.array([[0], [1], [10]])
    counts = counts_observed(q_data, bin_edges)
    assert counts.tolist() == [0, 3]


def test_counts_observed_bottom_left_corner():
    """Higher dim: check behavior near bottom-left most corner."""

    # Create 3 x 3 grid of bins
    # bins: [0, 1) x [0, 2), [0, 1) x [2, 4), [0, 1) x [4, 6]
    # bins: [1, 2) x [0, 2), [1, 2) x [2, 4), [1, 2) x [4, 6]
    # bins: [2, 3] x [0, 2), [2, 3] x [2, 4), [2, 3] x [4, 6]
    bin_edges = np.array([[0, 0], [1, 2], [2, 4], [3, 6]])

    # First, put points in bottom-left-most corner (first row, first col)
    base_q_data = np.array([[0, 0], [0.1, 0.1], [0, 1.9], [0.1, 1.9], [0.9, 1.9]])
    counts = counts_observed(base_q_data, bin_edges)
    assert np.allclose(counts, np.array([[5, 0, 0], [0, 0, 0], [0, 0, 0]]))

    # Add point (0, 2). Adds to first row, second col
    q_data = np.vstack([base_q_data, [0, 2]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([[5, 1, 0], [0, 0, 0], [0, 0, 0]]))

    # Add point (1, 2). Adds to second row, second column
    q_data = np.vstack([base_q_data, [1, 2]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([[5, 0, 0], [0, 1, 0], [0, 0, 0]]))

    # Add point (1, 1.99). Adds to second row, first column.
    q_data = np.vstack([base_q_data, [1, 1.99]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([[5, 0, 0], [1, 0, 0], [0, 0, 0]]))

    # Add point (0.99, 1.99). Adds to original bottom-left corner (first row, first col)
    q_data = np.vstack([base_q_data, [0.99, 1.99]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([[6, 0, 0], [0, 0, 0], [0, 0, 0]]))


def test_counts_observed_bottom_right_corner():
    """Higher dim: check behavior near bottom-right most corner."""

    # Create 3 x 3 grid of bins
    # bins: [0, 1) x [0, 2), [0, 1) x [2, 4), [0, 1) x [4, 6]
    # bins: [1, 2) x [0, 2), [1, 2) x [2, 4), [1, 2) x [4, 6]
    # bins: [2, 3] x [0, 2), [2, 3] x [2, 4), [2, 3] x [4, 6]
    bin_edges = np.array([[0, 0], [1, 2], [2, 4], [3, 6]])

    # First, put points in bottom-right-most corner (first row, third col)
    base_q_data = np.array([[0, 4], [0.1, 4.1], [0, 5.9], [0.1, 5.9], [0.9, 5.9], [0, 6]])
    counts = counts_observed(base_q_data, bin_edges)
    assert np.allclose(counts, np.array([[0, 0, 6], [0, 0, 0], [0, 0, 0]]))

    # Add point (1, 4). Adds to second row, third col
    q_data = np.vstack([base_q_data, [1, 4]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([[0, 0, 6], [0, 0, 1], [0, 0, 0]]))

    # Add point (0.99, 3.99). Adds to first row, second column
    q_data = np.vstack([base_q_data, [0.99, 3.99]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([[0, 1, 6], [0, 0, 0], [0, 0, 0]]))

    # Add point (1.99, 1.99). Adds to second row, first column.
    q_data = np.vstack([base_q_data, [1.99, 1.99]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([[0, 0, 6], [1, 0, 0], [0, 0, 0]]))

    # Add point (0.99, 6). Adds to original bottom-right corner (first row, third col)
    q_data = np.vstack([base_q_data, [0.99, 6]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([[0, 0, 7], [0, 0, 0], [0, 0, 0]]))


def test_counts_observed_top_right_corner():
    """Higher dim: check behavior near top-right corner."""

    # Create 3 x 3 grid of bins dividing up [0, 3] x [0, 6]
    # bins: [0, 1) x [0, 2), [0, 1) x [2, 4), [0, 1) x [4, 6]
    # bins: [1, 2) x [0, 2), [1, 2) x [2, 4), [1, 2) x [4, 6]
    # bins: [2, 3] x [0, 2), [2, 3] x [2, 4), [2, 3] x [4, 6]
    bin_edges = np.array([[0, 0], [1, 2], [2, 4], [3, 6]])

    # First, put points in top right corner (third row, third column)
    base_q_data = np.array([[2, 4], [2.1, 4.1], [3, 4], [3, 6], [3, 5.99], [2, 6]])
    counts = counts_observed(base_q_data, bin_edges)
    assert np.allclose(counts, np.array([[0, 0, 0], [0, 0, 0], [0, 0, 6]]))

    # Now add one at [2, 3.99]. Adds to third row, second column.
    q_data = np.vstack([base_q_data, [2, 3.99]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([[0, 0, 0], [0, 0, 0], [0, 1, 6]]))

    # Now add one at [1.99, 3.99]. Adds to second row, second column.
    q_data = np.vstack([base_q_data, [1.99, 3.99]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 6]]))

    # Now add one at [1.99, 4]. Adds to second row, third column.
    q_data = np.vstack([base_q_data, [1.99, 4]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([[0, 0, 0], [0, 0, 1], [0, 0, 6]]))


def test_counts_observed_top_left_corner():
    """Higher dim: check behavior near top-left most corner."""

    # Create 3 x 3 grid of bins
    # bins: [0, 1) x [0, 2), [0, 1) x [2, 4), [0, 1) x [4, 6]
    # bins: [1, 2) x [0, 2), [1, 2) x [2, 4), [1, 2) x [4, 6]
    # bins: [2, 3] x [0, 2), [2, 3] x [2, 4), [2, 3] x [4, 6]
    bin_edges = np.array([[0, 0], [1, 2], [2, 4], [3, 6]])

    # First, put points in top-left corner (third row, first col)
    base_q_data = np.array([[2, 0], [2.1, 0.1], [2, 1.9], [2.1, 1.9], [2.9, 1.9], [3, 0]])
    counts = counts_observed(base_q_data, bin_edges)
    assert np.allclose(counts, np.array([[0, 0, 0], [0, 0, 0], [6, 0, 0]]))

    # Add point (0, 2). Adds to first row, second col
    q_data = np.vstack([base_q_data, [0, 2]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([[0, 1, 0], [0, 0, 0], [6, 0, 0]]))

    # Add point (1, 2). Adds to second row, second column
    q_data = np.vstack([base_q_data, [1, 2]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([[0, 0, 0], [0, 1, 0], [6, 0, 0]]))

    # Add point (1, 1.99). Adds to second row, first column.
    q_data = np.vstack([base_q_data, [1, 1.99]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([[0, 0, 0], [1, 0, 0], [6, 0, 0]]))

    # Add point (0.99, 1.99). Adds to original bottom-left corner (first row, first col)
    q_data = np.vstack([base_q_data, [0.99, 1.99]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([[1, 0, 0], [0, 0, 0], [6, 0, 0]]))


def test_counts_observed_length_matches_bins():
    """Output length equals number of bins (len(bin_edges) - 1)."""
    q_data = np.array([0.1, 0.5, 0.9, 1.5])
    bin_edges = np.array([[0], [0.5], [1], [2]])  # 3 bins
    obs_counts = counts_observed(q_data, bin_edges)
    assert len(obs_counts) == len(bin_edges) - 1


def test_higher_dim_bin_shape():
    """Output counts of correct shape."""
    # Create 3 x 3 grid of bins dividing up [0, 3] x [0, 6]
    # bins: [0, 1) x [0, 2), [0, 1) x [2, 4), [0, 1) x [4, 6]
    # bins: [1, 2) x [0, 2), [1, 2) x [2, 4), [1, 2) x [4, 6]
    # bins: [2, 3] x [0, 2), [2, 3] x [2, 4), [2, 3] x [4, 6]
    bin_edges = np.array([[0, 0], [1, 2], [2, 4], [3, 6]])
    implied_bin_per_axis = len(bin_edges) - 1
    q_data = np.array([[2, 4], [2.1, 4.1], [3, 4], [3, 6], [3, 5.99], [2, 6]])
    counts = counts_observed(q_data, bin_edges)

    assert counts.shape == (implied_bin_per_axis, implied_bin_per_axis)
