"""
Unit tests for `prob_over_grid()`.

Tests:
- Shape of final dataset (rows = h^2)
- Input validation (bounds, h, length of input data)
- Boundary correctness (min/max match inputs)
- x and y edge construction (start/end at right place, have right spacing)
- Grid contiguity: end of one subinterval is start of next
- Grid has even spacing along either axis
- Grid spacing is non-degenerate and of expected length
- Columns and their types are as expected
- Probability check: h=1 implies prob=1; sum of probs = 1 across many cases
  (assuming prior has support contained in Lambda)
- If prior in a square only is in one half, probabilities should concentrate there.
- If prior fully leaves Lambda, probability should be 0 on entire grid
"""

import numpy as np
from pandas.api.types import is_numeric_dtype
import pytest

from src.probability_functions import prob_over_grid as prob_over_grid


def make_dummy_inputs(seed: int = 0, M_obs: int = 200, N_prior: int = 500):
    """A helper function to generate data.

    - q_prior_data and prior_data share length N.
    - q values in [0, 1], bin_edges cover [0, 1]."""

    rng = np.random.default_rng(seed)
    q_data = rng.random(M_obs)

    # Prior points subset of [0, 1]^2, so if Lambda contains this entire set,
    # probabilities will sum to 1.
    prior_data = rng.random((N_prior, 2))
    q_prior_data = rng.random(N_prior)
    bin_edges = np.linspace(0.0, 1.0, 11)
    return q_data, prior_data, q_prior_data, bin_edges


def test_rows_are_h_squared_loop():
    """Output dataframe should have h^2 rows for each probability."""

    q_data, prior_data, q_prior_data, bin_edges = make_dummy_inputs()
    for h in [1, 2, 3, 5, 10, 100]:
        df = prob_over_grid(
            q_data, prior_data, q_prior_data, bin_edges, grid_bounds=(0, 10, 0, 10), h=h
        )
        assert len(df) == h * h


def test_invalid_inputs_raise():
    """Invalid inputs raise ValueError (bad bounds or invalid h)."""

    q_data, prior_data, q_prior_data, bin_edges = make_dummy_inputs()

    invalid_cases = [
        # Bad X bounds
        dict(grid_bounds=(1, 1, 0, 10), h=2),
        dict(grid_bounds=(2, 1, 0, 10), h=2),
        # Bad Y bounds
        dict(grid_bounds=(0, 0, 10, 5), h=2),
        dict(grid_bounds=(0, 10, 9, 5), h=2),
        # Bad h
        dict(grid_bounds=(0, 10, 0, 10), h=0),
        dict(grid_bounds=(0, 10, 0, 10), h=-3),
        dict(grid_bounds=(0, 10, 0, 10), h=1.5),
        dict(grid_bounds=(0, 10, 0, 10), h="4"),
    ]
    error_messages = r"xmin < xmax and ymin < ymax|h must be an integer"
    for case in invalid_cases:
        with pytest.raises(ValueError, match=error_messages):
            prob_over_grid(q_data, prior_data, q_prior_data, bin_edges, **case)


def test_q_prior_data_prior_data_length_mismatch_raises():
    """q_prior_data and prior_data must have the same length."""

    q_data, prior_data, q_prior_data, bin_edges = make_dummy_inputs()
    prior_data_bad = prior_data[:-1]  # mismatch
    with pytest.raises(ValueError, match="q_prior_data and prior_data not same length"):
        prob_over_grid(
            q_data,
            prior_data_bad,
            q_prior_data,
            bin_edges,
            grid_bounds=(0, 1, 0, 1),
            h=2,
        )


def test_min_max_match_inputs():
    """Min/Max boundaries must match the input box."""

    q_data, prior_data, q_prior_data, bin_edges = make_dummy_inputs(seed=1)
    xmin, xmax, ymin, ymax, h = -3.25, 3.75, 5.5, 10.0, 3
    df = prob_over_grid(
        q_data,
        prior_data,
        q_prior_data,
        bin_edges,
        grid_bounds=(xmin, xmax, ymin, ymax),
        h=h,
    )
    assert df["xmin"].min() == xmin
    assert df["xmax"].max() == xmax
    assert df["ymin"].min() == ymin
    assert df["ymax"].max() == ymax


def test_x_and_y_grids_respect_bounds_and_spacing():
    """X and Y grid edges start, end, and step according to their respective bounds."""
    q_data, prior_data, q_prior_data, bin_edges = make_dummy_inputs(seed=2)
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 10.0, 30.0
    h = 4

    df = prob_over_grid(
        q_data,
        prior_data,
        q_prior_data,
        bin_edges,
        grid_bounds=(xmin, xmax, ymin, ymax),
        h=h,
    )

    xmins = np.sort(df["xmin"].unique())
    ymins = np.sort(df["ymin"].unique())

    dx = (xmax - xmin) / h
    dy = (ymax - ymin) / h

    assert dx != dy

    # x grid starts/ends at the left/right bound
    assert np.isclose(xmins[0], xmin)
    assert np.isclose(xmins[-1] + dx, xmax)

    # x grid edges are uniformly spaced by dx
    assert np.allclose(np.diff(xmins), dx)

    # y grid starts/ends at the left/right bound
    assert np.isclose(ymins[0], ymin)
    assert np.isclose(ymins[-1] + dy, ymax)

    # y grid edges are uniformly spaced by dy
    assert np.allclose(np.diff(ymins), dy)


def test_grid_has_expected_number_of_unique_edges():
    """Grid has h unique xmin/xmax and ymin/ymax edges."""
    q_data, prior_data, q_prior_data, bin_edges = make_dummy_inputs(seed=3)
    xmin, xmax, ymin, ymax, h = -7.0, 5.0, 2.0, 11.0, 7

    df = prob_over_grid(
        q_data,
        prior_data,
        q_prior_data,
        bin_edges,
        grid_bounds=(xmin, xmax, ymin, ymax),
        h=h,
    )

    assert len(df["xmin"].unique()) == h
    assert len(df["xmax"].unique()) == h
    assert len(df["ymin"].unique()) == h
    assert len(df["ymax"].unique()) == h


def test_grid_edges_are_contiguous():
    """Adjacent grid edges touch with no gaps or overlaps."""
    q_data, prior_data, q_prior_data, bin_edges = make_dummy_inputs(seed=3)
    xmin, xmax, ymin, ymax, h = -7.0, 5.0, 2.0, 11.0, 7

    df = prob_over_grid(
        q_data,
        prior_data,
        q_prior_data,
        bin_edges,
        grid_bounds=(xmin, xmax, ymin, ymax),
        h=h,
    )

    xmins = np.sort(df["xmin"].unique())
    xmaxs = np.sort(df["xmax"].unique())
    ymins = np.sort(df["ymin"].unique())
    ymaxs = np.sort(df["ymax"].unique())

    assert np.allclose(xmins[1:], xmaxs[:-1])
    assert np.allclose(ymins[1:], ymaxs[:-1])


def test_grid_edges_have_uniform_spacing():
    """Grid edges are evenly spaced along x and y."""
    q_data, prior_data, q_prior_data, bin_edges = make_dummy_inputs(seed=3)
    xmin, xmax, ymin, ymax, h = -7.0, 5.0, 2.0, 11.0, 7

    df = prob_over_grid(
        q_data,
        prior_data,
        q_prior_data,
        bin_edges,
        grid_bounds=(xmin, xmax, ymin, ymax),
        h=h,
    )

    xmins = np.sort(df["xmin"].unique())
    ymins = np.sort(df["ymin"].unique())

    dx = (xmax - xmin) / h
    dy = (ymax - ymin) / h

    assert np.allclose(np.diff(xmins), dx)
    assert np.allclose(np.diff(ymins), dy)


def test_tiles_have_constant_positive_dimensions():
    """All grid tiles have equal positive width and height."""
    q_data, prior_data, q_prior_data, bin_edges = make_dummy_inputs(seed=4)
    xmin, xmax, ymin, ymax, h = -3.5, 4.5, -10.0, -2.0, 4

    df = prob_over_grid(
        q_data,
        prior_data,
        q_prior_data,
        bin_edges,
        grid_bounds=(xmin, xmax, ymin, ymax),
        h=h,
    )

    dx = (xmax - xmin) / h
    dy = (ymax - ymin) / h

    widths = (df["xmax"] - df["xmin"]).to_numpy()
    heights = (df["ymax"] - df["ymin"]).to_numpy()

    assert np.allclose(widths, dx)
    assert np.allclose(heights, dy)
    assert np.all(widths > 0)
    assert np.all(heights > 0)


def test_columns_and_types():
    """Check columns and types of dataframe."""
    q_data, prior_data, q_prior_data, bin_edges = make_dummy_inputs(seed=5)
    df = prob_over_grid(q_data, prior_data, q_prior_data, bin_edges, grid_bounds=(0, 3, -2, 2), h=3)
    expected_cols = {"xmin", "xmax", "ymin", "ymax", "prob"}
    assert set(df.columns) == expected_cols
    for col in ["xmin", "xmax", "ymin", "ymax", "prob"]:
        assert is_numeric_dtype(df[col])
    for col in ["xmin", "xmax", "ymin", "ymax"]:
        assert not df[col].isna().any()


def test_h_one_prob_is_one_many_random_cases():
    """For h = 1, the single tile's probability is 1 for prior having support as subset of Lambda."""
    boxes = [(0.0, 1.0, 0.0, 1.0), (-0.5, 1.5, -0.5, 1.5)]

    for seed in [0, 1, 2, 7]:
        for M_obs, N_prior in [(50, 80), (200, 500), (1000, 2000)]:

            q_data, prior_data, q_prior_data, bin_edges = make_dummy_inputs(
                seed=seed, M_obs=M_obs, N_prior=N_prior
            )
            for xmin, xmax, ymin, ymax in boxes:
                df = prob_over_grid(
                    q_data,
                    prior_data,
                    q_prior_data,
                    bin_edges,
                    grid_bounds=(xmin, xmax, ymin, ymax),
                    h=1,
                )
                assert len(df) == 1
                assert np.isclose(df["prob"].iloc[0], 1.0)


def test_probs_sum_to_one_many_random_cases():
    """Sum of all probabilities is 1 across many seeds/boxes/h, for prior having subset of Lambda support."""

    boxes = [(0.0, 1.0, 0.0, 1.0), (-0.5, 1.5, -0.5, 1.5), (0.0, 10.0, 0.0, 10.0)]
    for seed in [0, 1, 2, 11]:
        for M_obs, N_prior in [(200, 500), (1000, 2000), (2000, 5000)]:
            q_data, prior_data, q_prior_data, bin_edges = make_dummy_inputs(
                seed=seed, M_obs=M_obs, N_prior=N_prior
            )
            for h in [1, 2, 3, 5, 8, 12]:
                for xmin, xmax, ymin, ymax in boxes:
                    df = prob_over_grid(
                        q_data,
                        prior_data,
                        q_prior_data,
                        bin_edges,
                        grid_bounds=(xmin, xmax, ymin, ymax),
                        h=h,
                    )
                    assert np.isclose(df["prob"].sum(), 1.0, rtol=1e-9, atol=1e-9)


def test_prob_mass_is_assigned_to_correct_x_column():
    """Probability mass falls entirely in the left column when all x_prior < mid-point."""
    h = 2  # a 2 x 2 grid
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    N = 200

    # All points lie strictly in the left x-tile (x < 0.5)
    x_prior = np.full(N, 0.25)
    y_prior = np.random.rand(N)
    prior_data = np.column_stack([x_prior, y_prior])

    q_prior_data = np.zeros(N)
    q_data = np.zeros(10)
    bin_edges = np.array([0.0, 1.0])

    df = prob_over_grid(
        q_data=q_data,
        prior_data=prior_data,
        q_prior_data=q_prior_data,
        bin_edges=bin_edges,
        grid_bounds=(xmin, xmax, ymin, ymax),
        h=h,
    )

    # Group tiles by xmin (each unique xmin is one vertical column),
    # sum the probability values within each column,
    # and return a NumPy array ordered from leftmost to rightmost column
    col_sums = df.groupby("xmin", sort=True)["prob"].sum().to_numpy()
    assert np.allclose(col_sums, [1.0, 0.0], atol=1e-12)


def test_prob_mass_is_assigned_to_correct_y_row():
    """Probability mass falls entirely in the bottom row when all y_prior < mid-point."""
    h = 2
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    N = 200

    # All points with y in [0, 0.5)
    y_prior = np.full(N, 0.25)
    x_prior = np.random.rand(N)
    prior_data = np.column_stack([x_prior, y_prior])

    # Make per-tile prob = fraction of points in tile (single q-bin)
    q_prior_data = np.zeros(N)
    q_data = np.zeros(10)
    bin_edges = np.array([0.0, 1.0])

    df = prob_over_grid(
        q_data=q_data,
        prior_data=prior_data,
        q_prior_data=q_prior_data,
        bin_edges=bin_edges,
        grid_bounds=(xmin, xmax, ymin, ymax),
        h=h,
    )

    # Group tiles by ymin (each unique ymin is one vertical column),
    # sum the probability values within each column,
    # and return a NumPy array ordered from leftmost to rightmost column
    row_sums = df.groupby("ymin", sort=True)["prob"].sum().to_numpy()
    assert np.allclose(row_sums, [1.0, 0.0])


def test_prob_is_zero_when_all_prior_points_outside_box():
    """All tile probabilities are zero when all prior points lie outside the grid."""
    h = 4  # a 2 x 2 grid
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    N = 200

    # All points lie strictly outside the box (to the right and above)
    x_prior = np.full(N, 2.0)  # x > xmax
    y_prior = np.full(N, 2.0)  # y > ymax
    prior_data = np.column_stack([x_prior, y_prior])

    q_prior_data = np.zeros(N)
    q_data = np.zeros(10)
    bin_edges = np.array([0.0, 1.0])

    df = prob_over_grid(
        q_data=q_data,
        prior_data=prior_data,
        q_prior_data=q_prior_data,
        bin_edges=bin_edges,
        grid_bounds=(xmin, xmax, ymin, ymax),
        h=h,
    )

    # Every tile should receive zero probability mass
    assert np.allclose(df["prob"].to_numpy(), 0.0)
