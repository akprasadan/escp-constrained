"""
Unit tests for `mask_in_A()`.

Tests:
- Full inclusion/exclusion of points by A
- Single-point exclusions via top/right boundaries
- Per-bin consistency of counts with/without masking
- Behavior when A is None (all included) and when A excludes all
- Constructed case where exactly half the points per q-bin are selected
- Return type/shape of masks
- Error guaranteed if empty input given for points
"""

import numpy as np
import pytest

from src.output_functions import apply_map as apply_map
from src.output_functions import counts_observed as counts_observed
from src.box_functions import mask_in_A as mask_in_A


def test_mask_in_A_all_points_in_rect_A_equal_bins():
    """If A includes all points, mask is all True and masked counts match unmasked counts."""
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
    q_prior_data = np.array([0.2, 0.8, 1.2, 1.9, 2.1, 2.7])
    prior_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.2, 0.9], [0.4, 0.1]])

    def A_rect(pts):
        x, y = pts[:, 0], pts[:, 1]
        return (x >= 0.0) & (x <= 0.8) & (y >= 0.1) & (y <= 0.9)

    mask_A = mask_in_A(prior_data, A_rect)
    prior_counts = counts_observed(q_prior_data, bin_edges)
    prior_A_counts = counts_observed(q_prior_data[mask_A], bin_edges)

    assert mask_A.all()
    assert np.array_equal(prior_counts, [2, 2, 2])
    assert np.array_equal(prior_A_counts, prior_counts)


def test_mask_in_A_excludes_one_point_top():
    """
    Exclude exactly one point by the TOP boundary (y too large).
    Expect one fewer count in the highest q-bin (excluded q=2.1 in [2,3)).
    """
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
    q_prior_data = np.array([0.2, 0.8, 1.2, 1.9, 2.1, 2.7])
    prior_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.2, 0.9], [0.4, 0.1]])

    # Half-open-style: exclude the point with y=0.9 (index 4); keep others.
    def A_rect(pts):
        x, y = pts[:, 0], pts[:, 1]
        return (x >= 0.0) & (x <= 0.8) & (y >= 0.1) & (y <= 0.85)

    mask_A = mask_in_A(prior_data, A_rect)
    prior_counts = counts_observed(q_prior_data, bin_edges)  # [2, 2, 2]
    prior_A_counts = counts_observed(q_prior_data[mask_A], bin_edges)  # exclude q=2.1 -> [2, 2, 1]

    assert np.count_nonzero(~mask_A) == 1
    assert np.array_equal(prior_counts, np.array([2, 2, 2]))
    assert np.array_equal(prior_A_counts, np.array([2, 2, 1]))


def test_mask_in_A_excludes_one_point_right():
    """
    Exclude exactly one point by the RIGHT boundary (x too large).
    Expect one fewer count in the middle q-bin (excluded q=1.9 in [1,2)).
    """
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
    q_prior_data = np.array([0.2, 0.8, 1.2, 1.9, 2.1, 2.7])
    prior_data = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.2, 0.9], [0.4, 0.1]])

    # Tighten xmax to exclude the point (0.7, 0.8) (index 3).
    def A_rect(pts):
        x, y = pts[:, 0], pts[:, 1]
        return (x >= 0.0) & (x <= 0.65) & (y >= 0.1) & (y <= 0.95)

    mask_A = mask_in_A(prior_data, A_rect)
    prior_counts = counts_observed(q_prior_data, bin_edges)  # [2, 2, 2]
    prior_A_counts = counts_observed(q_prior_data[mask_A], bin_edges)  # exclude q=1.9 -> [2, 1, 2]

    assert np.count_nonzero(~mask_A) == 1
    assert np.array_equal(prior_counts, np.array([2, 2, 2]))
    assert np.array_equal(prior_A_counts, np.array([2, 1, 2]))


def test_mask_in_A_rect_A_excludes_one_point_first_bin():
    """
    Exclude exactly one point from the FIRST q-bin [0,1).
    Exclude (0.1, 0.2) with q=0.2; expect masked counts [1, 2, 2].
    """
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
    q_prior_data = np.array([0.2, 0.8, 1.2, 1.9, 2.1, 2.7])
    prior_data = np.array(
        [
            [0.1, 0.2],  # q=0.2 in [0,1)  <- will be excluded
            [0.3, 0.4],  # q=0.8 in [0,1)
            [0.5, 0.6],  # q=1.2 in [1,2)
            [0.7, 0.8],  # q=1.9 in [1,2)
            [0.2, 0.9],  # q=2.1 in [2,3)
            [0.4, 0.1],  # q=2.7 in [2,3)
        ]
    )

    def A_rect(pts):
        x, y = pts[:, 0], pts[:, 1]
        return (x >= 0.15) & (x <= 1.0) & (y >= 0.0) & (y <= 1.0)

    mask_A = mask_in_A(prior_data, A_rect)
    prior_counts = counts_observed(q_prior_data, bin_edges)  # [2, 2, 2]
    prior_A_counts = counts_observed(q_prior_data[mask_A], bin_edges)  # exclude q=0.2 -> [1, 2, 2]

    assert np.count_nonzero(~mask_A) == 1
    assert np.array_equal(prior_counts, np.array([2, 2, 2]))
    assert np.array_equal(prior_A_counts, np.array([1, 2, 2]))


def test_mask_in_A_none_behaves_like_all():
    """If A is None, mask is all True and masked counts equal unmasked counts."""
    M, L = 3, 4
    J = M * L
    bin_edges = np.linspace(0.0, 1.0, M + 1)
    q_prior_data = (np.repeat(np.arange(M), L) + 0.5) / M
    prior_data = np.zeros((J, 2))

    mask_A = mask_in_A(prior_data, A=None)
    prior_counts = counts_observed(q_prior_data, bin_edges)
    prior_A_counts = counts_observed(q_prior_data[mask_A], bin_edges)

    assert mask_A.all()
    assert np.all(prior_counts == L)
    assert np.array_equal(prior_A_counts, prior_counts)


def test_mask_in_A_excludes_all_points():
    """If A excludes all points, mask is all False and masked counts are zeros."""
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0])
    q_prior_data = np.array([0.2, 0.8, 1.2, 1.9, 2.1, 2.7])
    prior_data = np.array(
        [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.2, 0.9], [0.4, 0.1]],
    )

    A_none = lambda pts: np.zeros(len(pts), dtype=bool)

    mask_A = mask_in_A(prior_data, A_none)
    prior_counts = counts_observed(q_prior_data, bin_edges)
    prior_A_counts = counts_observed(q_prior_data[mask_A], bin_edges)

    assert not mask_A.any()
    assert np.array_equal(prior_counts, [2, 2, 2])
    assert np.array_equal(prior_A_counts, [0, 0, 0])


def test_mask_in_A_half_in_A_equal_per_bin_with_real_points():
    """
    Three q-bins and three radii r with r^2 in each bin.
    For each r, place 4 points on the circle: two with x >= 0 and two with x < 0.
    A selects exactly half per r -> per-bin unmasked counts [4,4,4], masked [2,2,2].
    """
    bin_edges = np.array([0.0, 1.0, 4.0, 9.0])  # [0,1), [1,4), [4,9)
    radii = [0.5, 1.5, 2.5]  # r^2 = 0.25, 2.25, 6.25

    prior_data = np.array(
        [[[r, 0.0], [0.1, r], [-r, 0.0], [-0.1, -r]] for r in radii],
    ).reshape(-1, 2)

    def Q(x, y):
        return x**2 + y**2

    q_prior_data = apply_map(prior_data, Q)

    A = lambda pts: pts[:, 0] >= 0
    mask_A = mask_in_A(prior_data, A)

    prior_counts = counts_observed(q_prior_data, bin_edges)  # [4, 4, 4]
    prior_A_counts = counts_observed(q_prior_data[mask_A], bin_edges)  # [2, 2, 2]

    assert np.array_equal(prior_counts, np.array([4, 4, 4]))
    assert np.array_equal(prior_A_counts, np.array([2, 2, 2]))


def test_mask_in_A_returns_boolean_mask_of_length_J():
    """mask_in_A returns a boolean mask of length J."""
    J = 5
    prior_data = np.array(
        [
            [-1.0, 0.0],  # False
            [0.2, -0.1],  # True
            [0.0, 0.0],  # True (boundary)
            [1.0, 2.0],  # True
            [-0.3, 0.4],  # False
        ]
    )
    A = lambda pts: pts[:, 0] >= 0.0

    mask_A = mask_in_A(prior_data, A)
    expected = np.array([False, True, True, True, False])

    assert mask_A.dtype == bool
    assert mask_A.shape == (J,)
    assert np.array_equal(mask_A, expected)


def test_mask_in_A_no_input_points():
    """mask_in_A returns error if no points are supplied, for trivial and non-trivial A."""

    prior_data = np.array([])
    with pytest.raises(ValueError, match="points must be non-empty"):
        mask_in_A(points=prior_data, A=None)

    A = lambda pts: pts[:, 0] >= 0.0
    with pytest.raises(ValueError, match="points must be non-empty"):
        mask_in_A(points=prior_data, A=A)
