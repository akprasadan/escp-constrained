"""
Unit tests for `src.counts_observed()`.

Tests:
- Bin counting for simple cases
- Empty bins or empty input
- Check behavior at bin edges or corners
- When all values fall in the last bin
- Output is of correct shape
- Adding tile by tile increases counts correctly
- Our counts validator works correctly
"""

import numpy as np
import pytest

from src.output_functions import counts_observed, counts_validation


def test_counts_observed_basic():
    """Values are counted into bins correctly; last bin includes right edge."""
    bin_edges = np.array([[0], [1], [2], [3]])  # bins: [0,1), [1,2), [2,3]

    q_data = np.array([0.1, 0.5, 1.2, 1.8, 2])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([2, 2, 1]))

    q_data = np.array([0.5, 1.2, 1.8, 2])  # Remove the 0.1 point
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([1, 2, 1]))

    q_data = np.array([0.5, 1.0, 1.2, 1.8, 1.99, 2])  # add to middle bin
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([1, 4, 1]))


def test_counts_observed_under_permutation():
    """Permutations should have no impact on the answer."""
    rng = np.random.default_rng(seed=20251028)
    bin_edges = np.array([[0], [1], [2], [3]])

    q_data = np.array([-1, 0.1, 0.3, 0.5, 1.2, 1.8, 2])
    q_data_permuted = rng.permutation(q_data)
    assert not np.allclose(q_data, q_data_permuted)

    counts = counts_observed(q_data, bin_edges)
    counts_permuted = counts_observed(q_data_permuted, bin_edges)
    assert np.allclose(counts, counts_permuted)


def test_counts_observed_higher_dim_on_diagonal():
    """Higher dim: Values are counted into bins correctly if placed along diagonal
    from (0, 5) to (6, 20). Test each diagonal element one at a time, then together.
    """
    bottom_left = np.array(
        [[0, 5], [0, 9.9], [1, 5], [1, 6]]
    )  # Bottom left, 4x
    center = np.array(
        [[2, 10], [2, 14.9], [3, 12], [3.9, 14.9], [3.9, 10]]
    )  # Middle, 5x
    top_right = np.array(
        [[4, 15], [4, 20], [5, 17], [5, 20], [6, 15], [6, 20]]
    )  # Top right, 6x

    # Create 3 x 3 grid of bins
    # bins: [0, 2) x [5, 10), [0, 2) x [5, 10), [0, 2) x [5, 10]
    # bins: [2, 4) x [10, 15), [2, 4) x [10, 15), [2, 4) x [10, 15]
    # bins: [4, 6] x [15, 20), [4, 6] x [15, 20), [4, 6] x [15, 20]
    bin_edges = np.array([[0, 5], [2, 10], [4, 15], [6, 20]])

    # Note bottom of grid = top of array in bin_edges.
    counts_bottom_left = counts_observed(bottom_left, bin_edges)
    expected_bottom_left = np.array([[4, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert np.allclose(counts_bottom_left, expected_bottom_left)

    counts_center = counts_observed(center, bin_edges)
    expected_center = np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]])
    assert np.allclose(counts_center, expected_center)

    # Note top of grid = bottom of array in bin edges
    counts_top_right = counts_observed(top_right, bin_edges)
    expected_top_right = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 6]])
    assert np.allclose(counts_top_right, expected_top_right)

    # Ensure if we combine each diagonal element, we get right counts.
    combined_points = np.vstack([bottom_left, center, top_right])
    counts_combined = counts_observed(combined_points, bin_edges)
    expected_combined = np.array([[4, 0, 0], [0, 5, 0], [0, 0, 6]])
    assert np.allclose(counts_combined, expected_combined)


def test_counts_observed_empty_bin():
    """Bins with no values report zero counts (one bin non-empty)."""
    q_data = np.array([0.2, 0.3])  # both in first bin
    bin_edges = np.array([[0], [0.5], [1]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([2, 0]))


def test_counts_observed_fully_empty_bins():
    """All bins empty mean zero counts."""
    q_data = np.array([10, 20])  # both in first bin
    bin_edges = np.array([[0], [0.5], [1]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([0, 0]))


def test_counts_observed_empty_bin_higher_dim():
    """
    Higher dim: Bins with no values report zero counts (with
    at least one bin non-empty).
    """
    # All data in the [0.5, 1) x [4, 7] bin, top right
    q_data = np.array([[0.7, 5], [0.8, 5], [0.9, 4.5], [0.9, 7]])

    # Create 2 x 2 grid of bins on [0, 1] x [0, 7]
    # bins: [0, 0.5) x [0, 4), [0, 0.5) x [4, 7]
    # bins: [0.5, 1] x [0, 4), [0.5, 1] x [4, 7]
    bin_edges = np.array([[0, 0], [0.5, 4], [1, 7]])

    counts = counts_observed(q_data, bin_edges)
    expected = np.array([[0, 0], [0, 4]])
    assert np.allclose(counts, expected)


def test_counts_observed_fully_empty_bins_higher_dim():
    """Higher dim: All bins empty mean zero counts."""

    q_data = np.array([[-0.1, 10], [-0.1, -0.1], [1.1, 7.1], [1.1, -0.1]])

    # Create 2 x 2 grid of bins on [0, 1] x [0, 7]
    # bins: [0, 0.5) x [0, 4), [0, 0.5) x [4, 7]
    # bins: [0.5, 1] x [0, 4), [0.5, 1] x [4, 7]
    bin_edges = np.array([[0, 0], [0.5, 4], [1, 7]])

    counts = counts_observed(q_data, bin_edges)
    expected = np.array([[0, 0], [0, 0]])
    assert np.allclose(counts, expected)


def test_counts_observed_empty_list_returns_zeros():
    """Empty input give all 0 counts."""
    bin_edges = np.array([[-1], [0], [0.5], [2]])  # 3 bins
    counts = counts_observed(np.array([]), bin_edges)
    assert np.allclose(counts, np.zeros((3,)))

    # Also try a (N, 1) shaped array, not just (N, )
    counts = counts_observed(np.array([]), bin_edges)
    assert np.allclose(counts, np.zeros((3, 1)))


def test_counts_observed_empty_list_returns_zeros_higher_dim():
    """Higher dim: Empty input gives all 0 counts."""
    bin_edges = np.array([[0, 0], [0.5, 4], [1, 7]])
    counts = counts_observed(np.empty((0, 2)), bin_edges)
    assert np.allclose(counts, np.zeros((2, 2)))


def test_counts_observed_edges_inclusive():
    """Last bin includes its right edge; others are half-open."""
    q_data = np.array([0, 1, 2])
    bin_edges = np.array([[0], [1], [2]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([1, 2]))


def test_counts_observed_all_in_last_bin():
    """All values at or above last bin's left edge go into last bin."""
    q_data = np.array([1, 5, 10])
    bin_edges = np.array([[0], [1], [10]])
    counts = counts_observed(q_data, bin_edges)
    assert np.allclose(counts, np.array([0, 3]))


def test_counts_observed_higher_dim_randomized_add_one_by_one():
    """
    First, start with a fairly large set of q_data, populating [0, 3] x [0, 6],
    split into 9 tiles. Then, for each of the 9 tiles, add 4-5 datapoints
    just in that tile, and ensure we increased the counts by the expected amount
    and in the right bin.
    """

    # Create 3 x 3 grid of bins
    # bins: [0, 1) x [0, 2), [0, 1) x [2, 4), [0, 1) x [4, 6]
    # bins: [1, 2) x [0, 2), [1, 2) x [2, 4), [1, 2) x [4, 6]
    # bins: [2, 3] x [0, 2), [2, 3] x [2, 4), [2, 3] x [4, 6]
    bin_edges = np.array([[0, 0], [1, 2], [2, 4], [3, 6]])

    n = 30
    rng = np.random.default_rng(seed=20251028)
    base_q_data = np.column_stack((rng.uniform(0, 3, n), rng.uniform(0, 6, n)))
    base_counts = counts_observed(base_q_data, bin_edges)

    def check_single_added_tile(added_points: np.ndarray, expected: np.ndarray):
        q_data = np.vstack([base_q_data, added_points])
        counts = counts_observed(q_data, bin_edges)
        assert np.allclose(counts, expected)

    # Add points to first row, first column (bottom left)
    first_row_first_col = np.array([[0, 0], [0.9, 1.9], [0, 1.9], [0.9, 1.9]])
    expected = base_counts + np.array([[4, 0, 0], [0, 0, 0], [0, 0, 0]])
    check_single_added_tile(first_row_first_col, expected)

    # Add points to first row, second column (bottom center)
    first_row_second_col = np.array([[0, 2], [0, 3.9], [0.9, 2], [0.9, 3.9]])
    expected = base_counts + np.array([[0, 4, 0], [0, 0, 0], [0, 0, 0]])
    check_single_added_tile(first_row_second_col, expected)

    # Add points to first row, third column (bottom right)
    first_row_third_col = np.array([[0, 4], [0, 6], [0.9, 4], [0.9, 6]])
    expected = base_counts + np.array([[0, 0, 4], [0, 0, 0], [0, 0, 0]])
    check_single_added_tile(first_row_third_col, expected)

    # Add points to second row, first column (center left)
    second_row_first_col = np.array([[1, 0], [1, 1.9], [1.9, 0], [1.9, 1.9]])
    expected = base_counts + np.array([[0, 0, 0], [4, 0, 0], [0, 0, 0]])
    check_single_added_tile(second_row_first_col, expected)

    # Add points to second row, second column (center)
    second_row_second_col = np.array([[1, 2], [1, 3.9], [1.9, 2], [1.9, 3.9]])
    expected = base_counts + np.array([[0, 0, 0], [0, 4, 0], [0, 0, 0]])
    check_single_added_tile(second_row_second_col, expected)

    # Add points to second row, third column (center right)
    second_row_third_col = np.array([[1, 4], [1, 5.9], [1.9, 4], [1.9, 5.9]])
    expected = base_counts + np.array([[0, 0, 0], [0, 0, 4], [0, 0, 0]])
    check_single_added_tile(second_row_third_col, expected)

    # Add points to third row, first column (top left)
    third_row_first_col = np.array([[2, 0], [2, 1.9], [3, 0], [3, 1.9]])
    expected = base_counts + np.array([[0, 0, 0], [0, 0, 0], [4, 0, 0]])
    check_single_added_tile(third_row_first_col, expected)

    # Add points to third row, second column (top center)
    third_row_second_col = np.array([[2, 2], [2, 3.9], [3, 2], [3, 3.9]])
    expected = base_counts + np.array([[0, 0, 0], [0, 0, 0], [0, 4, 0]])
    check_single_added_tile(third_row_second_col, expected)

    # Add points to third row, third column (top right)
    third_row_third_col = np.array([[2, 4], [2.1, 4.1], [2, 6], [3, 4], [3, 6]])
    expected = base_counts + np.array([[0, 0, 0], [0, 0, 0], [0, 0, 5]])
    check_single_added_tile(third_row_third_col, expected)


def test_counts_observed_length_matches_bins():
    """1D Output shape is (M, ), where M is number of bins."""
    q_data = np.array([0.1, 0.5, 0.9, 1.5])
    bin_edges = np.array([[0], [0.5], [1], [2]])  # 3 bins
    obs_counts = counts_observed(q_data, bin_edges)
    implied_bin_per_axis = len(bin_edges) - 1
    assert obs_counts.shape == (implied_bin_per_axis,)


def test_higher_dim_bin_shape():
    """2D Output shape is (M, M), where M is number of bins. 3D is (M, M, M)."""
    # Create 3 x 3 grid of bins dividing up [0, 3] x [0, 6]
    # bins: [0, 1) x [0, 2), [0, 1) x [2, 4), [0, 1) x [4, 6]
    # bins: [1, 2) x [0, 2), [1, 2) x [2, 4), [1, 2) x [4, 6]
    # bins: [2, 3] x [0, 2), [2, 3] x [2, 4), [2, 3] x [4, 6]
    bin_edges = np.array([[0, 0], [1, 2], [2, 4], [3, 6]])
    bin_per_axis = len(bin_edges) - 1
    q_data = np.array([[2, 4], [2.1, 4.1], [3, 4], [3, 6], [3, 5.9], [2, 6]])
    counts = counts_observed(q_data, bin_edges)

    assert counts.shape == (bin_per_axis, bin_per_axis)

    # 3D data: (M, M, M)
    bin_edges = np.array([[0, 0, 0], [1, 2, 2], [2, 4, 3], [3, 6, 10]])
    bin_per_axis = len(bin_edges) - 1
    q_data = np.array(
        [
            [2, 4, 1],
            [2.1, 4.1, 2],
            [3, 4, 8],
            [3, 6, 10],
            [3, 5.9, 1],
            [2, 6, 0],
        ]
    )
    counts = counts_observed(q_data, bin_edges)
    assert counts.shape == (bin_per_axis, bin_per_axis, bin_per_axis)


def test_counts_observed_incompatible_shapes():
    """Check that incompatible q_data or bin_edges triggers error."""
    # 2D bin edges with 1D q_data
    bin_edges = np.array([[0, 0], [1, 2], [2, 4], [3, 6]])
    q_data = np.array([0.1, 0.5, 0.9, 1.5])

    with pytest.raises(ValueError, match="Bins or data of wrong dimension"):
        counts_observed(q_data, bin_edges)

    # 1D bin edges with 2D q_data
    bin_edges = np.array([[0], [0.5], [1], [2]])
    q_data = np.array([[2, 4], [2.1, 4.1], [3, 4], [3, 6], [3, 5.9], [2, 6]])
    with pytest.raises(ValueError, match="Bins or data of wrong dimension"):
        counts_observed(q_data, bin_edges)


# This function used to be checked in prob_A, now is checked in estimate_grid_probs_df
def test_counts_validation_bad_counts_raises():
    """Counts cannot be non-integers or negative."""
    counts = np.array([5, 1, 0.2])
    with pytest.raises(ValueError, match=r"array must be integers"):
        counts_validation(counts)

    counts = np.array([5, 1, -2])
    with pytest.raises(ValueError, match=r"cannot contain negative values."):
        counts_validation(counts)

    counts = np.array([[1, 2, 3], [3, -1, 5], [6, 1, 2]])
    with pytest.raises(ValueError, match=r"cannot contain negative values."):
        counts_validation(counts)

    counts = np.array([[1, 2, 3], [3.1, 1, 5], [6, 1, 2]])
    with pytest.raises(ValueError, match=r"array must be integers."):
        counts_validation(counts)
