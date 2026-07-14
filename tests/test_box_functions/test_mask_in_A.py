"""
Unit tests for `src.box_functions.py::mask_in_A()`.

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

from src.box_functions import mask_in_A, rect_A
from src.output_functions import apply_map, counts_observed


def test_mask_in_A_all_points_in_rect_A_equal_bins() -> None:
    """Test when A includes all points.

    If A includes all points, mask is all True and masked counts match unmasked
    counts. Higher dim version below.
    """
    # Note we convert (N, ) array to (N, 1)
    bin_edges = np.array([0, 1, 2, 3])[:, None]
    q_prior_data = np.array([0.2, 0.8, 1.2, 1.9, 2.1, 2.7])
    prior_data = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
        [0.2, 0.9],
        [0.4, 0.1],
    ])

    def A_rect(pts: np.ndarray) -> np.ndarray:
        x, y = pts[:, 0], pts[:, 1]
        return (x >= 0) & (x <= 0.8) & (y >= 0.1) & (y <= 0.9)

    # should be all true, since prior_data in [0, 0.8] x [0.1, 0.9]
    mask_A = mask_in_A(prior_data, A_rect)
    assert mask_A.all()

    # Should just put 2 q points per each bin in bin_edges
    prior_counts = counts_observed(q_prior_data, bin_edges)
    assert np.array_equal(prior_counts, [2, 2, 2])

    # mask on Lambda should filter nothing in D, hence leave unchanged counts
    prior_A_counts = counts_observed(q_prior_data[mask_A], bin_edges)
    assert np.array_equal(prior_A_counts, prior_counts)


def test_mask_in_A_excludes_one_point_top() -> None:
    """
    Test excluding one point.

    Exclude exactly one point by the TOP boundary (y too large). Expect one
    fewer count in the highest q-bin (excluded q=2.1 in [2,3)). Higher dim
    version below.
    """
    bin_edges = np.array([0, 1, 2, 3])[:, None]
    q_prior_data = np.array([0.2, 0.8, 1.2, 1.9, 2.1, 2.7])
    prior_data = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
        [0.2, 0.9],
        [0.4, 0.1],
    ])

    # Half-open-style: exclude the point with y=0.9 (index 4); keep others.
    def A_rect(pts: np.ndarray) -> np.ndarray:
        x, y = pts[:, 0], pts[:, 1]
        return (x >= 0) & (x <= 0.8) & (y >= 0.1) & (y <= 0.85)

    mask_A = mask_in_A(prior_data, A_rect)
    prior_counts = counts_observed(q_prior_data, bin_edges)  # [2, 2, 2]
    prior_A_counts = counts_observed(
        q_prior_data[mask_A], bin_edges
    )  # exclude q=2.1 -> [2, 2, 1]

    assert np.count_nonzero(~mask_A) == 1
    assert np.array_equal(prior_counts, np.array([2, 2, 2]))
    assert np.array_equal(prior_A_counts, np.array([2, 2, 1]))


def test_mask_in_A_excludes_one_point_right() -> None:
    """
    Test excluding one point.

    Exclude exactly one point by the RIGHT boundary (x too large). Expect one
    fewer count in the middle q-bin (excluded q=1.9 in [1,2)). Higher dim
    version below.
    """
    bin_edges = np.array([0, 1, 2, 3])[:, None]
    q_prior_data = np.array([0.2, 0.8, 1.2, 1.9, 2.1, 2.7])
    prior_data = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
        [0.2, 0.9],
        [0.4, 0.1],
    ])

    # Tighten xmax to exclude the point (0.7, 0.8) (index 3).
    def A_rect(pts: np.ndarray) -> np.ndarray:
        x, y = pts[:, 0], pts[:, 1]
        return (x >= 0) & (x <= 0.65) & (y >= 0.1) & (y <= 0.95)

    mask_A = mask_in_A(prior_data, A_rect)
    prior_counts = counts_observed(q_prior_data, bin_edges)  # [2, 2, 2]
    prior_A_counts = counts_observed(
        q_prior_data[mask_A], bin_edges
    )  # exclude q=1.9 -> [2, 1, 2]

    assert np.count_nonzero(~mask_A) == 1
    assert np.array_equal(prior_counts, np.array([2, 2, 2]))
    assert np.array_equal(prior_A_counts, np.array([2, 1, 2]))


def test_mask_in_A_rect_A_excludes_one_point_first_bin() -> None:
    """Test excluding one point from first bin [0, 1).

    Exclude prior point (0.1, 0.2) with q=0.2; expect masked counts [1, 2, 2].
    """
    bin_edges = np.array([0, 1, 2, 3])[:, None]
    q_prior_data = np.array([0.2, 0.8, 1.2, 1.9, 2.1, 2.7])
    prior_data = np.array([
        [0.1, 0.2],  # q=0.2 in [0,1)  <- will be excluded
        [0.3, 0.4],  # q=0.8 in [0,1)
        [0.5, 0.6],  # q=1.2 in [1,2)
        [0.7, 0.8],  # q=1.9 in [1,2)
        [0.2, 0.9],  # q=2.1 in [2,3)
        [0.4, 0.1],  # q=2.7 in [2,3)
    ])

    def A_rect(pts: np.ndarray) -> np.ndarray:
        x, y = pts[:, 0], pts[:, 1]
        return (x >= 0.15) & (x <= 1) & (y >= 0) & (y <= 1)

    mask_A = mask_in_A(prior_data, A_rect)
    prior_counts = counts_observed(q_prior_data, bin_edges)  # [2, 2, 2]
    prior_A_counts = counts_observed(
        q_prior_data[mask_A], bin_edges
    )  # exclude q=0.2 -> [1, 2, 2]

    assert np.count_nonzero(~mask_A) == 1
    assert np.array_equal(prior_counts, np.array([2, 2, 2]))
    assert np.array_equal(prior_A_counts, np.array([1, 2, 2]))


def test_mask_in_A_none_behaves_like_all() -> None:
    """Test when A is none.

    If A is None, mask is all True and masked counts equal unmasked counts.
    Higher dim version below.
    """
    M, L = 3, 4
    J = M * L
    bin_edges = np.linspace(0, 1, M + 1)[:, None]
    q_prior_data = (np.repeat(np.arange(M), L) + 0.5) / M
    prior_data = np.zeros((J, 2))

    mask_A = mask_in_A(prior_data, A=None)
    prior_counts = counts_observed(q_prior_data, bin_edges)
    prior_A_counts = counts_observed(q_prior_data[mask_A], bin_edges)

    assert mask_A.all()
    assert np.all(prior_counts == L)
    assert np.array_equal(prior_A_counts, prior_counts)


def test_mask_in_A_excludes_all_points() -> None:
    """Test A excluding all points.

    If A excludes all points, mask is all False and masked counts are zeros
    Higher dim version below.
    """
    bin_edges = np.array([0, 1, 2, 3])[:, None]
    q_prior_data = np.array([0.2, 0.8, 1.2, 1.9, 2.1, 2.7])[:, None]
    prior_data = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
        [0.2, 0.9],
        [0.4, 0.1],
    ])

    def A_none(pts: np.ndarray) -> np.ndarray:
        return np.zeros(len(pts), dtype=bool)

    mask_A = mask_in_A(prior_data, A_none)
    prior_counts = counts_observed(q_prior_data, bin_edges)
    prior_A_counts = counts_observed(q_prior_data[mask_A], bin_edges)
    assert not mask_A.any()
    assert np.array_equal(prior_counts, [2, 2, 2])
    assert np.array_equal(prior_A_counts, [0, 0, 0])


def test_mask_in_A_half_in_A_equal_per_bin_with_real_points() -> None:
    """Test circular mask.

    Pick three q-bins and three radii r with r^2 in each bin. For each r, place
    4 points on the circle: two with x >= 0 and two with x < 0. "A" selects
    exactly half per r -> per-bin unmasked counts [4,4,4], masked [2,2,2].
    """
    bin_edges = np.array([0, 1, 4, 9])[:, None]  # [0,1), [1,4), [4,9)
    radii = [0.5, 1.5, 2.5]  # r^2 = 0.25, 2.25, 6.25

    prior_data = np.array([
        [[r, 0], [0.1, r], [-r, 0], [-0.1, -r]] for r in radii
    ]).reshape(-1, 2)

    def Q(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x**2 + y**2

    q_prior_data = apply_map(prior_data, Q)

    def A(pts: np.ndarray) -> np.ndarray:
        return pts[:, 0] >= 0

    mask_A = mask_in_A(prior_data, A)

    prior_counts = counts_observed(q_prior_data, bin_edges)  # [4, 4, 4]
    prior_A_counts = counts_observed(q_prior_data[mask_A], bin_edges)  # [2, 2, 2]

    assert np.array_equal(prior_counts, np.array([4, 4, 4]))
    assert np.array_equal(prior_A_counts, np.array([2, 2, 2]))


def test_mask_in_A_returns_boolean_mask_of_length_J() -> None:
    """mask_in_A returns a boolean mask of length J.

    Higher dim version incorporated below (inside other higher dim tests).
    """
    J = 5
    prior_data = np.array([
        [-1, 0],  # False
        [0.2, -0.1],  # True
        [0, 0],  # True (boundary)
        [1, 2],  # True
        [-0.3, 0.4],  # False
    ])

    def A(pts: np.ndarray) -> np.ndarray:
        return pts[:, 0] >= 0

    mask_A = mask_in_A(prior_data, A)
    expected = np.array([False, True, True, True, False])

    assert mask_A.dtype == bool
    assert mask_A.shape == (J,)
    assert np.array_equal(mask_A, expected)


def test_mask_in_A_no_input_points() -> None:
    """Test empty input.

    mask_in_A returns error if no points are supplied, for trivial and
    non-trivial A.
    """
    prior_data = np.array([])
    with pytest.raises(ValueError, match="points must be non-empty"):
        mask_in_A(points=prior_data, A=None)

    def A(pts: np.ndarray) -> np.ndarray:
        return pts[:, 0] >= 0

    with pytest.raises(ValueError, match="points must be non-empty"):
        mask_in_A(points=prior_data, A=A)


def test_mask_in_A_no_input_points_higher_dim() -> None:
    """Higher dim: test empty input.

    mask_in_A returns error if no points are supplied, for trivial and
    non-trivial A.
    """
    prior_data = np.empty((0, 2))
    with pytest.raises(ValueError, match="points must be non-empty"):
        mask_in_A(points=prior_data, A=None)

    A = rect_A(np.array([0, 0.1, 0]), np.array([3, 4, 2]))
    with pytest.raises(ValueError, match="points must be non-empty"):
        mask_in_A(points=prior_data, A=A)


def test_mask_in_A_1d() -> None:
    """Perform a several 1-D examples of mask behavior."""
    # Note we convert (N, ) array to (N, 1)
    bin_edges = np.array([0, 1, 2, 3])[:, None]
    q_prior_data = np.array([0.2, 0.8, 1.2, 1.9, 2.1, 2.7, 2.9])[:, None]
    prior_data = np.array([0, -1, 0.5, 0.6, 2, 0.8, 0.7])[:, None]
    rect = rect_A(np.array([0]), np.array([0.8]))
    mask_A = mask_in_A(prior_data, rect)

    assert mask_A.dtype == bool

    expected = np.array([1, 0, 1, 1, 0, 0, 1])
    assert np.allclose(mask_A, expected)

    # Should just put 2 or 3 q points per each bin in bin_edges
    prior_counts = counts_observed(q_prior_data, bin_edges)
    assert np.array_equal(prior_counts, [2, 2, 3])

    # mask on Lambda should only keep the first, third/fourth, and seventh item
    # this removes 1 from bin 1 and 2 from bin 3
    prior_A_counts = counts_observed(q_prior_data[mask_A], bin_edges)
    expected_A = np.array([1, 2, 1])
    assert np.array_equal(prior_A_counts, expected_A)

    # This should now add 6th point in prior_data
    new_rect = rect_A(np.array([0]), np.array([0.81]))
    new_mask_A = mask_in_A(prior_data, new_rect)
    new_expected = np.array([1, 0, 1, 1, 0, 1, 1])
    assert np.allclose(new_mask_A, new_expected)

    # Now add back one point to prior_A_counts in third bin
    new_prior_A_counts = counts_observed(q_prior_data[new_mask_A], bin_edges)
    new_expected_A = np.array([1, 2, 2])
    assert np.array_equal(new_prior_A_counts, new_expected_A)

    # Now say every point is included
    prior_data = np.linspace(0, 0.7, 7)[:, None]
    rect = rect_A(np.array([0]), np.array([0.8]))
    mask_A = mask_in_A(prior_data, rect)
    assert mask_A.all()

    # Mask should not subset anything in q_prior_data
    prior_A_counts = counts_observed(q_prior_data[mask_A], bin_edges)
    assert np.allclose(prior_A_counts, prior_counts)

    # If A is none
    mask_A = mask_in_A(prior_data, None)
    assert mask_A.all()


def test_mask_in_A_non_rectangular() -> None:
    """
    Test irregular mask on different dimensions.

    We subset 2D points such that the floor function applied to the first
    coordinate is even.
    """
    rng = np.random.default_rng(seed=20251028)
    bin_edges = np.array([0, 10, 20, 30])[:, None]
    q_prior_data = np.array([15, 2, 3, 4, 12, 24, 9, 28])
    prior_data_x = np.array([1.1, 2.2, 0.1, 2.5, 9.8, 10.0, -0.5, 0.0])
    prior_data_y = rng.uniform(low=-10, high=50, size=len(prior_data_x))
    prior_data = np.column_stack([prior_data_x, prior_data_y])

    def A_even(pts: np.ndarray) -> np.ndarray:
        return np.floor(pts[:, 0]) % 2 == 0

    mask_A = mask_in_A(prior_data, A_even)
    expected = np.array([False, True, True, True, False, True, False, True])
    assert np.allclose(mask_A, expected)

    prior_counts = counts_observed(q_prior_data, bin_edges)
    assert np.array_equal(prior_counts, [4, 2, 2])

    # We should remove q_points of 15, 12, and 9, i.e., two from middle bin, one
    # from first bin.
    prior_A_counts = counts_observed(q_prior_data[mask_A], bin_edges)
    expected_counts = np.array([3, 0, 2])
    assert np.array_equal(prior_A_counts, expected_counts)


"""
Begin a set of tests for 3D Lambda, 2D D, where we first include all the points
in D by choosing sufficiently large rectangle mask. Then we test removing one
point at a time by shrinking the rectangle mask along certain dimensions. Then
we close by removing all but a single points by shrinking the rectangle around
one point.

Bins over [0, 3]^2:
    - [0, 1) x [0, 1),    [0, 1) x [1, 2),    [0, 1] x [2, 3] 
    - [1, 2) x [0, 1),    [1, 2) x [1, 2),    [1, 2) x [2, 3] 
    - [2, 3] x [0, 1),    [2, 3] x [1, 2),    [2, 3] x [2, 3] 
"""


BIN_EDGES_2D = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
Q_PRIOR_DATA_2D = np.array([
    [0.2, 0.8],  # We will come back to this point (row 1, col 1)
    [1.2, 0.9],  # we will later only pick this point (row 2, col 1)
    [1.2, 1.5],
    [1.2, 1.9],
    [2.1, 2.7],
    [2.9, 2.1],
    [1.2, 2.5],
    [0.2, 2.9],
    [2.5, 0.1],  # We will come back to this point (row 3, col 1)
])

PRIOR_DATA_3D = np.array([
    [0.01, 0.101, 0.15],  # close to leaving mask if we increase x lower lim
    [0.15, 0.2, 0.15],  # we will later only pick this point.
    [2, 1, 1],
    [2.9, 0.2, 1.5],
    [0.7, 0.8, 0.15],
    [0.2, 0.9, 1.5],
    [1, 3.9, 0.15],
    [2, 3, 1.5],
    [0.5, 0.5, 1.99],  # close to leaving mask if we decrease z upper limit
])


def test_mask_in_A_all_points_in_rect_A_equal_bins_higher_dim() -> None:
    """Higher dim: test A including every point.

    If A includes all points, mask is all True and masked counts match unmasked
    counts. Check types and shape too of mask.
    """
    rect_mask = rect_A(np.array([0, 0.1, 0]), np.array([3, 4, 2]))

    # should be all true, since prior_data in [0, 3] x [0.1, 4] x [0, 2]
    mask_A = mask_in_A(PRIOR_DATA_3D, rect_mask)
    assert mask_A.all()
    assert mask_A.dtype == bool
    assert mask_A.shape == (len(PRIOR_DATA_3D),)

    prior_counts = counts_observed(Q_PRIOR_DATA_2D, BIN_EDGES_2D)
    expected_prior_counts = np.array([[1, 0, 1], [1, 2, 1], [1, 0, 2]])
    assert np.array_equal(prior_counts, expected_prior_counts)

    # mask on Lambda should filter nothing in D, hence leave unchanged counts
    prior_A_counts = counts_observed(Q_PRIOR_DATA_2D[mask_A], BIN_EDGES_2D)
    assert np.array_equal(prior_A_counts, prior_counts)


def test_mask_in_A_if_A_none_higher_dim() -> None:
    """Higher dim: If A includes is none, mask is trivially true."""
    # should be all true
    mask_A = mask_in_A(PRIOR_DATA_3D, None)
    assert mask_A.all()
    assert mask_A.dtype == bool
    assert mask_A.shape == (len(PRIOR_DATA_3D),)

    prior_counts = counts_observed(Q_PRIOR_DATA_2D, BIN_EDGES_2D)
    expected_prior_counts = np.array([[1, 0, 1], [1, 2, 1], [1, 0, 2]])
    assert np.array_equal(prior_counts, expected_prior_counts)

    # mask on Lambda should filter nothing in D, hence leave unchanged counts
    prior_A_counts = counts_observed(Q_PRIOR_DATA_2D[mask_A], BIN_EDGES_2D)
    assert np.array_equal(prior_A_counts, prior_counts)


def test_mask_in_A_excludes_one_point_row3_col1_higher_dim() -> None:
    """Exclude exactly one point in 3D.

    Do so by taking the rectangle A = [0, 3] x [0.1, 4] x [0, 1.95] instead of A
    = [0, 3] x [0.1, 4] x [0, 2]. Check that corresponding q point is removed.
    """
    # Choose box to only remove the last point of prior_data
    rect_mask = rect_A(np.array([0, 0.1, 0]), np.array([3, 4, 1.95]))

    mask_A = mask_in_A(PRIOR_DATA_3D, rect_mask)
    assert mask_A.dtype == bool
    assert mask_A.shape == (len(PRIOR_DATA_3D),)

    # Only one point removed from prior_data
    assert np.count_nonzero(~mask_A) == 1

    prior_counts = counts_observed(Q_PRIOR_DATA_2D, BIN_EDGES_2D)
    expected_prior_counts = np.array([[1, 0, 1], [1, 2, 1], [1, 0, 2]])
    assert np.array_equal(prior_counts, expected_prior_counts)

    prior_A_counts = counts_observed(Q_PRIOR_DATA_2D[mask_A], BIN_EDGES_2D)
    # Remove the row 3, column 1 point
    expected_prior_A_counts = np.array([[1, 0, 1], [1, 2, 1], [0, 0, 2]])
    assert np.array_equal(prior_A_counts, expected_prior_A_counts)


def test_mask_in_A_excludes_one_point_row1_col1_higher_dim() -> None:
    """Exclude one point in 3D.

    Exclude exactly one point by taking the rectangle A = [0.02, 3] x [0.1, 4] x
    [0, 2] instead of A = [0, 3] x [0.1, 4] x [0, 2]. Check that corresponding q
    point is removed.
    """
    # Choose box to only remove the first point of prior_data (x-value of 0.01)
    rect_mask = rect_A(np.array([0.02, 0.1, 0]), np.array([3, 4, 2]))
    mask_A = mask_in_A(PRIOR_DATA_3D, rect_mask)
    assert mask_A.dtype == bool
    assert mask_A.shape == (len(PRIOR_DATA_3D),)

    # Only one point removed again
    assert np.count_nonzero(~mask_A) == 1

    prior_counts = counts_observed(Q_PRIOR_DATA_2D, BIN_EDGES_2D)
    expected_prior_counts = np.array([[1, 0, 1], [1, 2, 1], [1, 0, 2]])
    assert np.array_equal(prior_counts, expected_prior_counts)

    prior_A_counts = counts_observed(Q_PRIOR_DATA_2D[mask_A], BIN_EDGES_2D)
    # Remove a row 1, column 1 point
    expected_prior_A_counts = np.array([[0, 0, 1], [1, 2, 1], [1, 0, 2]])
    assert np.array_equal(prior_A_counts, expected_prior_A_counts)


def test_mask_in_A_excludes_all_but_one_point_higher_dim() -> None:
    """Test excluding one point in 3D.

    Exclude all but one point by taking the rectangle A to be concentrated
    sharply around the second point, [0.15, 0.2, 0.15].
    """
    # Choose box to only remove the last point of prior_data
    rect_mask = rect_A(np.array([0.149, 0.199, 0.149]), np.array([0.151, 0.201, 0.151]))

    mask_A = mask_in_A(PRIOR_DATA_3D, rect_mask)
    # All but one point removed from prior_data
    assert np.count_nonzero(~mask_A) == len(PRIOR_DATA_3D) - 1

    prior_counts = counts_observed(Q_PRIOR_DATA_2D, BIN_EDGES_2D)
    expected_prior_counts = np.array([[1, 0, 1], [1, 2, 1], [1, 0, 2]])
    # As usual, no change to unmasked version
    assert np.array_equal(prior_counts, expected_prior_counts)

    prior_A_counts = counts_observed(Q_PRIOR_DATA_2D[mask_A], BIN_EDGES_2D)
    # Remove all but row 2, col 1 point
    expected_prior_A_counts = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    assert np.array_equal(prior_A_counts, expected_prior_A_counts)


def test_mask_in_A_excludes_all_points_higher_dim() -> None:
    """Higher dim: exclude all points."""
    # Choose box to only remove the last point of prior_data
    rect_mask = rect_A(np.array([5, 5, 5]), np.array([7, 7, 7]))

    mask_A = mask_in_A(PRIOR_DATA_3D, rect_mask)
    assert mask_A.dtype == bool
    assert mask_A.shape == (len(PRIOR_DATA_3D),)

    # All points removed from prior_data
    assert np.count_nonzero(~mask_A) == len(PRIOR_DATA_3D)

    prior_counts = counts_observed(Q_PRIOR_DATA_2D, BIN_EDGES_2D)
    expected_prior_counts = np.array([[1, 0, 1], [1, 2, 1], [1, 0, 2]])
    # As usual, no change to unmasked version
    assert np.array_equal(prior_counts, expected_prior_counts)

    prior_A_counts = counts_observed(Q_PRIOR_DATA_2D[mask_A], BIN_EDGES_2D)
    # Remove all points
    expected_prior_A_counts = np.zeros((3, 3))
    assert np.array_equal(prior_A_counts, expected_prior_A_counts)
