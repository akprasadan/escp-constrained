"""
Unit tests for `prob_over_grid()`.

Tests:
- Shape of final dataset (rows = h^d in d-dimensions)
- Input validation (bounds, h, length of input data)
- Boundary correctness (min/max match inputs)
- x and y edge construction (start/end at right place, have right spacing)
- Grid contiguity: end of one subinterval is start of next
- Grid has even spacing along each axis
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

from src.box_functions import rect_A
from src.output_functions import make_partition as make_partition
from src.probability_functions import prob_over_grid as prob_over_grid


def dummy_data(
    grid_mins: np.ndarray,
    grid_maxes: np.ndarray,
    seed: int = 0,
    M_obs: int = 200,
    N_prior: int = 500,
):
    """A helper function to generate data.

    - prior_data in [0, 1]^2. If Lambda contains this entire set, probabilities will sum to 1.
    - q_prior_data and prior_data share length N_prior.
    - q values in [0, 1], bin_edges cover [0, 1].
    """

    rng = np.random.default_rng(seed)
    q_data = rng.random(M_obs)
    prior_data = rng.random((N_prior, 2))

    # Use [:, None] to make sure (N, ) arrays are treated as (N, 1) arrays.
    q_prior_data = rng.random(N_prior)[:, None]
    bin_edges = np.linspace(0.0, 1.0, 11)[:, None]
    return q_data, prior_data, q_prior_data, bin_edges, grid_mins, grid_maxes


def dummy_data_higher_dim(
    grid_mins: np.ndarray,
    grid_maxes: np.ndarray,
    seed: int = 0,
    M_obs: int = 200,
    N_prior: int = 500,
):
    """A helper function to generate data in higher dimensions.

    - prior_data in [0, 1]^3. If Lambda contains this entire set, probabilities will sum to 1.
    - q_prior_data and prior_data share length N.
    - q values in [0, 1]^2, bin_edges cover [0, 1]^2.
    """

    rng = np.random.default_rng(seed)
    q_data = rng.random((M_obs, 2))
    prior_data = rng.random((N_prior, 3))
    q_prior_data = rng.random((N_prior, 2))
    bin_edges = make_partition(
        min_vals=np.array([-0.1, -0.1]), max_vals=np.array([1.1, 1.1]), M=5
    )
    return q_data, prior_data, q_prior_data, bin_edges, grid_mins, grid_maxes


def test_rows_are_h_squared_loop():
    """Output dataframe should have h^2 rows for each probability,
    and in d-dimensions for Lambda, h^d."""

    dummy_inputs = dummy_data(
        grid_mins=np.array([0, 0]), grid_maxes=np.array([10, 10])
    )
    for h in [1, 2, 3, 5, 10, 100]:
        df = prob_over_grid(*dummy_inputs, h=h)
        assert len(df) == h * h

    # Higher dimension too
    dummy_inputs = dummy_data_higher_dim(
        grid_mins=np.array([0, 0, 0]), grid_maxes=np.array([10, 10, 10])
    )
    for h in [1, 2, 3, 5, 6, 7, 8]:
        df = prob_over_grid(*dummy_inputs, h=h)
        assert len(df) == h**3


def test_invalid_inputs_raise():
    """Invalid inputs raise ValueError (bad bounds or invalid h)."""

    bad_bounds = [
        ([1, 1], [0, 10]),  # bad x
        ([2, 0], [1, 10]),  # bad x
        ([2, 0], [-1, 10]),  # bad x
        ([0, 10], [10, 5]),  # bad y
        ([0, 9], [10, 5]),  # bad y
        ([0, 9], [10, -1]),  # bad y
        ([3, 9], [2, 5]),  # bad x and bad y
    ]

    invalid_dummy_data = [
        dummy_data(grid_mins=np.array(mins), grid_maxes=np.array(maxes))
        for mins, maxes in bad_bounds
    ]

    invalid_hs = [0, -3, 1.5, "4"]

    for bad_dummy in invalid_dummy_data:
        with pytest.raises(ValueError, match="min_vals >= max_vals"):
            prob_over_grid(*bad_dummy, h=2)

    good_dummy_input = dummy_data(
        grid_mins=np.array([0, 0]), grid_maxes=np.array([1, 1])
    )
    for bad_h in invalid_hs:
        with pytest.raises(ValueError, match="h must be an integer"):
            prob_over_grid(*good_dummy_input, h=bad_h)


def test_invalid_inputs_raise_higher_dim():
    """Higher dim: Invalid inputs raise ValueError (bad bounds or invalid h)."""

    bad_bounds = [
        ([1, 1, 0], [0, 10, 1]),  # bad x
        ([2, 0, 5], [1, 10, 15]),  # bad x
        ([0, 10, 0], [10, 5, 1]),  # bad y
        ([0, 9, 2], [10, 5, 4]),  # bad y
        ([0, 0, 5], [10, 5, 1]),  # bad z
        ([0, 2, 2], [10, 5, 0]),  # bad z
    ]

    invalid_dummy_data = [
        dummy_data_higher_dim(
            grid_mins=np.array(mins), grid_maxes=np.array(maxes)
        )
        for mins, maxes in bad_bounds
    ]

    invalid_hs = [0, -3, 1.5, "4"]

    for bad_dummy in invalid_dummy_data:
        with pytest.raises(ValueError, match="min_vals >= max_vals"):
            prob_over_grid(*bad_dummy, h=2)

    good_dummy_input = dummy_data_higher_dim(
        grid_mins=np.array([0, 0, 0]), grid_maxes=np.array([1, 1, 1])
    )
    for bad_h in invalid_hs:
        with pytest.raises(ValueError, match="h must be an integer"):
            prob_over_grid(*good_dummy_input, h=bad_h)


def test_q_prior_data_prior_data_length_mismatch_raises():
    """q_prior_data and prior_data must have the same length. Higher dim tests included too."""

    q_data, prior_data, q_prior_data, bin_edges, grid_mins, grid_maxes = (
        dummy_data(grid_mins=np.array([0, 0]), grid_maxes=np.array([1, 1]))
    )
    prior_data_bad = prior_data[:-1]  # mismatch
    with pytest.raises(
        ValueError, match="q_prior_data and prior_data not same length"
    ):
        prob_over_grid(
            q_data,
            prior_data_bad,
            q_prior_data,
            bin_edges,
            grid_mins=grid_mins,
            grid_maxes=grid_maxes,
            h=2,
        )

    # Higher dimensions:
    q_data, prior_data, q_prior_data, bin_edges, grid_mins, grid_maxes = (
        dummy_data_higher_dim(
            grid_mins=np.array([0, 0, 0]), grid_maxes=np.array([1, 1, 1])
        )
    )
    prior_data_bad = prior_data[:-1]  # mismatch
    with pytest.raises(
        ValueError, match="q_prior_data and prior_data not same length"
    ):
        prob_over_grid(
            q_data,
            prior_data_bad,
            q_prior_data,
            bin_edges,
            grid_mins=grid_mins,
            grid_maxes=grid_maxes,
            h=3,
        )


def test_min_max_match_inputs():
    """Min/Max boundaries must match the input box. In higher dimensions too."""

    xmin, xmax = -3.25, 3.75
    ymin, ymax = 5.5, 10.0
    dummy_inputs = dummy_data(
        grid_mins=np.array([xmin, ymin]),
        grid_maxes=np.array([xmax, ymax]),
        seed=1,
    )
    df = prob_over_grid(*dummy_inputs, h=3)

    assert (df["xmin"].min(), df["xmax"].max()) == (xmin, xmax)
    assert (df["ymin"].min(), df["ymax"].max()) == (ymin, ymax)

    # Higher dimensions

    xmin, xmax = -3, 3
    ymin, ymax = -2, 5
    zmin, zmax = -1, 10

    dummy_inputs = dummy_data_higher_dim(
        grid_mins=np.array([xmin, ymin, zmin]),
        grid_maxes=np.array([xmax, ymax, zmax]),
        seed=1,
    )
    df = prob_over_grid(*dummy_inputs, h=3)
    assert (df["xmin"].min(), df["xmax"].max()) == (xmin, xmax)
    assert (df["ymin"].min(), df["ymax"].max()) == (ymin, ymax)
    assert (df["zmin"].min(), df["zmax"].max()) == (zmin, zmax)


def test_x_and_y_grids_respect_bounds_and_spacing():
    """X and Y grid edges start, end, and step according to their respective bounds."""
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 10.0, 30.0
    dummy_inputs = dummy_data(
        grid_mins=np.array([xmin, ymin]),
        grid_maxes=np.array([xmax, ymax]),
        seed=2,
    )
    h = 4

    df = prob_over_grid(*dummy_inputs, h=h)

    claimed_xmins = np.sort(df["xmin"].unique())
    claimed_ymins = np.sort(df["ymin"].unique())

    dx = (xmax - xmin) / h
    dy = (ymax - ymin) / h

    assert dx != dy

    # x, y grids starts/ends at the left/right bound
    assert np.isclose(claimed_xmins[0], xmin)
    assert np.isclose(claimed_xmins[-1] + dx, xmax)
    assert np.isclose(claimed_ymins[0], ymin)
    assert np.isclose(claimed_ymins[-1] + dy, ymax)

    # x, y grid edges are uniformly spaced by dx, dy
    assert np.allclose(np.diff(claimed_xmins), dx)
    assert np.allclose(np.diff(claimed_ymins), dy)


def test_x_and_y_and_z_grids_respect_bounds_and_spacing():
    """X and Y and Z grid edges start, end, and step according to their respective bounds."""
    xmin, xmax = 0.0, 1.0
    ymin, ymax = 10.0, 30.0
    zmin, zmax = -1, 5
    dummy_inputs = dummy_data_higher_dim(
        grid_mins=np.array([xmin, ymin, zmin]),
        grid_maxes=np.array([xmax, ymax, zmax]),
        seed=2,
    )
    h = 4

    df = prob_over_grid(*dummy_inputs, h=h)

    xmins = np.sort(df["xmin"].unique())
    ymins = np.sort(df["ymin"].unique())
    zmins = np.sort(df["zmin"].unique())

    dx = (xmax - xmin) / h
    dy = (ymax - ymin) / h
    dz = (zmax - zmin) / h

    assert dx != dy
    assert dy != dz
    assert dx != dz

    # x, y, and z grid starts/ends at the left/right bound
    assert np.isclose(xmins[0], xmin)
    assert np.isclose(xmins[-1] + dx, xmax)
    assert np.isclose(ymins[0], ymin)
    assert np.isclose(ymins[-1] + dy, ymax)
    assert np.isclose(zmins[0], zmin)
    assert np.isclose(zmins[-1] + dz, zmax)

    # x, y, z grid edges are uniformly spaced by dx, dy, dz
    assert np.allclose(np.diff(xmins), dx)
    assert np.allclose(np.diff(ymins), dy)
    assert np.allclose(np.diff(zmins), dz)


def test_grid_has_expected_number_of_unique_edges():
    """Grid has h unique xmin/xmax and ymin/ymax edges. In higher dimensions too."""
    h = 7
    xmin, xmax, ymin, ymax = -7.0, 5.0, 2.0, 11.0
    dummy_inputs = dummy_data(
        grid_mins=np.array([xmin, ymin]),
        grid_maxes=np.array([xmax, ymax]),
        seed=3,
    )

    df = prob_over_grid(*dummy_inputs, h=h)
    for edge in ["xmin", "xmax", "ymin", "ymax"]:
        assert len(df[edge].unique()) == h

    # Higher dimensions too
    xmin, xmax, ymin, ymax, zmin, zmax = -7.0, 5.0, 2.0, 11.0, -1, 3
    dummy_inputs = dummy_data_higher_dim(
        grid_mins=np.array([xmin, ymin, zmin]),
        grid_maxes=np.array([xmax, ymax, zmax]),
        seed=3,
    )

    df = prob_over_grid(*dummy_inputs, h=h)

    for edge in ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]:
        assert len(df[edge].unique()) == h


def test_grid_edges_are_contiguous():
    """Adjacent grid edges touch with no gaps or overlaps."""
    h = 7
    xmin, xmax, ymin, ymax = -7.0, 5.0, 2.0, 11.0
    dummy_inputs = dummy_data(
        grid_mins=np.array([xmin, ymin]),
        grid_maxes=np.array([xmax, ymax]),
        seed=3,
    )

    df = prob_over_grid(*dummy_inputs, h=h)

    xmins = np.sort(df["xmin"].unique())
    xmaxs = np.sort(df["xmax"].unique())
    ymins = np.sort(df["ymin"].unique())
    ymaxs = np.sort(df["ymax"].unique())

    # adjacent edges touch
    assert np.allclose(xmins[1:], xmaxs[:-1])
    assert np.allclose(ymins[1:], ymaxs[:-1])

    # In higher dimensions

    xmin, xmax, ymin, ymax, zmin, zmax = -7.0, 5.0, 2.0, 11.0, 0, 3
    dummy_inputs = dummy_data_higher_dim(
        grid_mins=np.array([xmin, ymin, zmin]),
        grid_maxes=np.array([xmax, ymax, zmax]),
        seed=3,
    )

    df = prob_over_grid(*dummy_inputs, h=h)

    xmins = np.sort(df["xmin"].unique())
    xmaxs = np.sort(df["xmax"].unique())
    ymins = np.sort(df["ymin"].unique())
    ymaxs = np.sort(df["ymax"].unique())
    zmins = np.sort(df["zmin"].unique())
    zmaxs = np.sort(df["zmax"].unique())

    # adjacent edges touch
    assert np.allclose(xmins[1:], xmaxs[:-1])
    assert np.allclose(ymins[1:], ymaxs[:-1])
    assert np.allclose(zmins[1:], zmaxs[:-1])


def test_grid_edges_have_uniform_spacing():
    """Grid edges are evenly spaced along x and y."""
    h = 7
    xmin, xmax, ymin, ymax = -7.0, 5.0, 2.0, 11.0
    dummy_inputs = dummy_data(
        grid_mins=np.array([xmin, ymin]),
        grid_maxes=np.array([xmax, ymax]),
        seed=3,
    )

    df = prob_over_grid(*dummy_inputs, h=h)

    xmins = np.sort(df["xmin"].unique())
    ymins = np.sort(df["ymin"].unique())

    dx = (xmax - xmin) / h
    dy = (ymax - ymin) / h

    assert np.allclose(np.diff(xmins), dx)
    assert np.allclose(np.diff(ymins), dy)

    # Higher dimensions
    xmin, xmax, ymin, ymax, zmin, zmax = -7.0, 5.0, 2.0, 11.0, 0, 3
    dummy_inputs = dummy_data_higher_dim(
        grid_mins=np.array([xmin, ymin, zmin]),
        grid_maxes=np.array([xmax, ymax, zmax]),
        seed=3,
    )

    df = prob_over_grid(*dummy_inputs, h=h)

    xmins = np.sort(df["xmin"].unique())
    ymins = np.sort(df["ymin"].unique())
    zmins = np.sort(df["zmin"].unique())

    dx = (xmax - xmin) / h
    dy = (ymax - ymin) / h
    dz = (zmax - zmin) / h

    assert np.allclose(np.diff(xmins), dx)
    assert np.allclose(np.diff(ymins), dy)
    assert np.allclose(np.diff(zmins), dz)


def test_tiles_have_constant_positive_dimensions():
    """All grid tiles have equal positive width and height."""
    h = 4
    xmin, xmax, ymin, ymax = -3.5, 4.5, -10.0, -2.0
    dummy_inputs = dummy_data(
        grid_mins=np.array([xmin, ymin]),
        grid_maxes=np.array([xmax, ymax]),
        seed=4,
    )

    df = prob_over_grid(*dummy_inputs, h=h)

    dx = (xmax - xmin) / h
    dy = (ymax - ymin) / h

    widths = (df["xmax"] - df["xmin"]).to_numpy()
    heights = (df["ymax"] - df["ymin"]).to_numpy()

    assert np.allclose(widths, dx)
    assert np.allclose(heights, dy)
    assert np.all(widths > 0)
    assert np.all(heights > 0)

    # Higher dim
    xmin, xmax, ymin, ymax, zmin, zmax = -3.5, 4.5, -10.0, -2.0, 0, 2

    dummy_inputs = dummy_data_higher_dim(
        grid_mins=np.array([xmin, ymin, zmin]),
        grid_maxes=np.array([xmax, ymax, zmax]),
        seed=4,
    )

    df = prob_over_grid(*dummy_inputs, h=h)

    dx = (xmax - xmin) / h
    dy = (ymax - ymin) / h
    dz = (zmax - zmin) / h

    widths = (df["xmax"] - df["xmin"]).to_numpy()
    heights = (df["ymax"] - df["ymin"]).to_numpy()
    depths = (df["zmax"] - df["zmin"]).to_numpy()

    # Check each width/length/depth is right size and non-zero
    assert np.allclose(widths, dx)
    assert np.allclose(heights, dy)
    assert np.allclose(depths, dz)

    assert np.all(widths > 0)
    assert np.all(heights > 0)
    assert np.all(depths > 0)


def test_columns_and_types():
    """Check columns and types of dataframe."""
    dummy_inputs = dummy_data(
        grid_mins=np.array([0, -2]), grid_maxes=np.array([3, 2]), seed=5
    )
    df = prob_over_grid(*dummy_inputs, h=3)
    expected_cols = {"xmin", "xmax", "ymin", "ymax", "prob"}
    assert set(df.columns) == expected_cols
    for col in expected_cols:
        assert is_numeric_dtype(df[col])
    for col in expected_cols:
        assert not df[col].isna().any()


def test_columns_and_types_higher_dims():
    """Higher dims: Check columns and types of dataframe."""
    dummy_inputs = dummy_data_higher_dim(
        grid_mins=np.array([0, -2, 1]), grid_maxes=np.array([3, 2, 5]), seed=5
    )
    df = prob_over_grid(*dummy_inputs, h=3)
    expected_cols = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "prob"}
    assert set(df.columns) == expected_cols
    for col in expected_cols:
        assert is_numeric_dtype(df[col])
    for col in expected_cols:
        assert not df[col].isna().any()


def test_h_one_prob_is_one_many_random_cases():
    """For h = 1, the single tile's probability is 1 for prior having support as subset of Lambda.
    This requires prior_A_counts = prior_counts on every bin, and no bins
    with 0 prior_counts. If the bin granularity is too high or the N_prior sample size
    too low, this will fail. For example, this test fails if we replace (50, 100) with (50, 10).
    """
    boxes = [(0.0, 1.0, 0.0, 1.0), (-0.5, 1.5, -0.5, 1.5)]

    for seed in [0, 1, 2, 7]:
        for M_obs, N_prior in [(50, 100), (200, 500), (1000, 2000)]:
            for xmin, xmax, ymin, ymax in boxes:
                dummy_inputs = dummy_data(
                    grid_mins=np.array([xmin, ymin]),
                    grid_maxes=np.array([xmax, ymax]),
                    seed=seed,
                    M_obs=M_obs,
                    N_prior=N_prior,
                )
                df = prob_over_grid(*dummy_inputs, h=1)
                assert len(df) == 1
                assert np.isclose(df["prob"].iloc[0], 1.0)


def test_h_one_prob_is_one_many_random_cases_higher_dims():
    """
    Higher dim: For h = 1, the single tile's probability is 1 for prior
    having support as subset of Lambda. This requires prior_A_counts = prior_counts on every bin, and no bins
    with 0 prior_counts. If the bin granularity is too high or the N_prior sample size
    too low, this will fail. For example, this test fails if we replace (50, 400) with (50, 100).
    In higher dimensions, it's harder to fill every bin unless the bin are few in number.
    """
    boxes = [(0, 1, 0, 1, 0, 1), (-0.5, 1.5, -0.5, 1.5, -1, 2)]

    for seed in [0, 1, 2, 7]:
        for M_obs, N_prior in [(50, 400), (200, 500), (1000, 2000)]:
            for xmin, xmax, ymin, ymax, zmin, zmax in boxes:
                dummy_inputs = dummy_data_higher_dim(
                    grid_mins=np.array([xmin, ymin, zmin]),
                    grid_maxes=np.array([xmax, ymax, zmax]),
                    seed=seed,
                    M_obs=M_obs,
                    N_prior=N_prior,
                )
                df = prob_over_grid(*dummy_inputs, h=1)
                assert len(df) == 1

                assert np.isclose(df["prob"].iloc[0], 1.0)


def test_probs_sum_to_one_many_random_cases():
    """
    Sum of all probabilities is 1 across many seeds/boxes/h, for prior having subset of
    Lambda support and prior sample size sufficiently large to fill all bins.
    """

    boxes = [
        (0.0, 1.0, 0.0, 1.0),
        (-0.5, 1.5, -0.5, 1.5),
        (0.0, 10.0, 0.0, 10.0),
    ]
    for seed in [0, 1, 2, 11]:
        for M_obs, N_prior in [(200, 500), (1000, 2000), (2000, 5000)]:
            for h in [1, 2, 3, 5, 6]:
                for xmin, xmax, ymin, ymax in boxes:
                    dummy_inputs = dummy_data(
                        grid_mins=np.array([xmin, ymin]),
                        grid_maxes=np.array([xmax, ymax]),
                        seed=seed,
                        M_obs=M_obs,
                        N_prior=N_prior,
                    )

                    df = prob_over_grid(*dummy_inputs, h=h)
                    assert np.isclose(
                        df["prob"].sum(), 1.0, rtol=1e-9, atol=1e-9
                    )


def test_probs_sum_to_one_many_random_cases_higher_dim():
    """
    Higher dim: Sum of all probabilities is 1 across many seeds/boxes/h,
    for prior having subset of Lambda support, and prior sample size sufficiently large to fill all
    bins.
    """

    boxes = [
        (0.0, 1.0, 0.0, 1.0, 0, 1),
        (-0.5, 1.5, -0.5, 1.5, -1, 2),
        (0.0, 10.0, 0.0, 10.0, -0.1, 11),
    ]
    for seed in [0, 1, 2, 11]:
        for M_obs, N_prior in [(200, 500), (1000, 2000), (2000, 5000)]:
            for h in [1, 2, 3, 5, 6]:  # don't make h large, is expensive
                for xmin, xmax, ymin, ymax, zmin, zmax in boxes:
                    dummy_inputs = dummy_data_higher_dim(
                        grid_mins=np.array([xmin, ymin, zmin]),
                        grid_maxes=np.array([xmax, ymax, zmax]),
                        seed=seed,
                        M_obs=M_obs,
                        N_prior=N_prior,
                    )

                    df = prob_over_grid(*dummy_inputs, h=h)
                    assert np.isclose(
                        df["prob"].sum(), 1.0, rtol=1e-9, atol=1e-9
                    )


def test_prob_mass_is_assigned_to_correct_x_column():
    """Probability mass falls entirely in the left column when all x_prior < mid-point."""
    h = 2  # a 2 x 2 grid
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    N = 200

    rng = np.random.default_rng(seed=20251028)
    # All points lie strictly in the left x-tile (x < 0.5)
    x_prior = np.full(N, 0.25)
    y_prior = rng.random(N)
    prior_data = np.column_stack([x_prior, y_prior])

    q_prior_data = rng.random((N, 1))
    q_data = rng.random((10, 1))
    bin_edges = np.array([[0.0], [1.0]])

    df = prob_over_grid(
        q_data=q_data,
        prior_data=prior_data,
        q_prior_data=q_prior_data,
        bin_edges=bin_edges,
        grid_mins=np.array([xmin, ymin]),
        grid_maxes=np.array([xmax, ymax]),
        h=h,
    )

    # Group tiles by xmin (each unique xmin is one vertical column),
    # sum the probability values within each column,
    # and return a NumPy array ordered from leftmost to rightmost column
    col_sums = df.groupby("xmin", sort=True)["prob"].sum().to_numpy()
    assert np.allclose(col_sums, [1.0, 0.0], atol=1e-12)


def test_prob_mass_is_assigned_to_correct_bin_higher_dim():
    """Probability mass falls entirely in the correct bins when all x_prior > mid-point,
    y_prior < mid_point."""
    h = 2  # a 2 x 2 x 2 grid
    xmin, xmax, ymin, ymax, zmin, zmax = 0.0, 1.0, 0.0, 1.0, 0, 1
    N = 200
    rng = np.random.default_rng(seed=20251028)
    # All points lie strictly in the right x-tiles (x > 0.5) and y-tiles (y < 0.5)
    x_prior = np.full(N, 0.75)
    y_prior = np.full(N, 0.25)
    z_prior = rng.random(N)
    prior_data = np.column_stack([x_prior, y_prior, z_prior])

    q_prior_data = rng.random((N, 2))
    q_data = rng.random((10, 2))
    bin_edges = np.array([[0.0, 0.0], [1.0, 1.0]])

    df = prob_over_grid(
        q_data=q_data,
        prior_data=prior_data,
        q_prior_data=q_prior_data,
        bin_edges=bin_edges,
        grid_mins=np.array([xmin, ymin, zmin]),
        grid_maxes=np.array([xmax, ymax, zmax]),
        h=h,
    )

    # Group tiles by xmin/ymin and sum the probability values within each column
    # The col sums will first get (x_low, y_low), then (xlow, y_high),
    # then (x_high, y_low), and finally (x_high, y_high).
    # So we should see everything in column 3.
    col_sums = df.groupby(["xmin", "ymin"], sort=True)["prob"].sum().to_numpy()
    assert np.allclose(col_sums, [0, 0, 1, 0], atol=1e-12)

    # Now suppose we move a single y prior to be 0.75. This should add 1/200 to
    # the fourth column sum, and subtract 1/200 from the third column sum
    y_prior[0] = 0.75
    prior_data = np.column_stack([x_prior, y_prior, z_prior])
    df = prob_over_grid(
        q_data=q_data,
        prior_data=prior_data,
        q_prior_data=q_prior_data,
        bin_edges=bin_edges,
        grid_mins=np.array([xmin, ymin, zmin]),
        grid_maxes=np.array([xmax, ymax, zmax]),
        h=h,
    )
    # Group tiles by each combo of xmin/ymin as its own column
    # The rows in that column are all the diff zmins, whose prob we sum over
    col_sums = df.groupby(["xmin", "ymin"], sort=True)["prob"].sum().to_numpy()
    assert np.allclose(col_sums, [0, 0, 1 - 1 / 200, 1 / 200], atol=1e-12)


def test_prob_mass_is_assigned_to_correct_y_row():
    """Probability mass falls entirely in the bottom row when all y_prior < mid-point."""
    h = 2
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    N = 200

    rng = np.random.default_rng(seed=20251028)
    # All points with y in [0, 0.5)
    y_prior = np.full(N, 0.25)
    x_prior = rng.random(N)
    prior_data = np.column_stack([x_prior, y_prior])

    # Make per-tile prob = fraction of points in tile (single q-bin)
    q_prior_data = rng.random((N, 1))
    q_data = rng.random((10, 1))
    bin_edges = np.array([[0.0], [1.0]])

    df = prob_over_grid(
        q_data=q_data,
        prior_data=prior_data,
        q_prior_data=q_prior_data,
        bin_edges=bin_edges,
        grid_mins=np.array([xmin, ymin]),
        grid_maxes=np.array([xmax, ymax]),
        h=h,
    )

    # Group tiles by ymin (each unique ymin is one vertical column),
    # sum the probability values within each column,
    # and return a NumPy array ordered from leftmost to rightmost column
    row_sums = df.groupby("ymin", sort=True)["prob"].sum().to_numpy()
    assert np.allclose(row_sums, [1.0, 0.0])


def test_prob_mass_is_assigned_to_correct_single_bin_higher_dim():
    """Higher dim: Probability mass falls entirely in single bin (x, y > 0.5, z < 0.5)"""
    h = 2  # a 2 x 2 x 2 grid
    xmin, xmax, ymin, ymax, zmin, zmax = 0.0, 1.0, 0.0, 1.0, 0, 1
    N = 200
    rng = np.random.default_rng(seed=20251028)
    # All points lie strictly in the left z-tiles (z > 0.5)
    z_prior = np.full(N, 0.25)
    y_prior = np.full(N, 0.75)
    x_prior = np.full(N, 0.75)
    prior_data = np.column_stack([x_prior, y_prior, z_prior])

    q_prior_data = rng.random((N, 2))
    q_data = rng.random((10, 2))
    bin_edges = np.array([[0.0, 0.0], [1.0, 1.0]])

    df = prob_over_grid(
        q_data=q_data,
        prior_data=prior_data,
        q_prior_data=q_prior_data,
        bin_edges=bin_edges,
        grid_mins=np.array([xmin, ymin, zmin]),
        grid_maxes=np.array([xmax, ymax, zmax]),
        h=h,
    )

    # Group tiles by each combo of xmin, ymin, zmin
    # sum the probability values within each column
    # and return a NumPy array ordered from leftmost to rightmost column
    # There should be 8 column sum entries, and it should be the second to last
    # with all the weight (since last two entries are: x_high, y_high, z_low,
    # x_high, y_high, z_high
    col_sums = (
        df.groupby(["xmin", "ymin", "zmin"], sort=True)["prob"].sum().to_numpy()
    )
    assert np.allclose(col_sums, [0, 0, 0, 0, 0, 0, 1, 0], atol=1e-12)


def test_prob_is_zero_when_all_prior_points_outside_box():
    """All tile probabilities are zero when all prior points lie outside the grid."""
    h = 4  # a 2 x 2 grid
    xmin, xmax, ymin, ymax = 0.0, 1.0, 0.0, 1.0
    N = 200

    # All points lie strictly outside the box (to the right and above)
    x_prior = np.full(N, 2.0)  # x > xmax
    y_prior = np.full(N, 2.0)  # y > ymax
    prior_data = np.column_stack([x_prior, y_prior])

    q_prior_data = np.zeros(N)[:, None]
    q_data = np.zeros(10)[:, None]
    bin_edges = np.array([[0.0], [1.0]])

    df = prob_over_grid(
        q_data=q_data,
        prior_data=prior_data,
        q_prior_data=q_prior_data,
        bin_edges=bin_edges,
        grid_mins=np.array([xmin, ymin]),
        grid_maxes=np.array([xmax, ymax]),
        h=h,
    )

    # Every tile should receive zero probability mass
    assert np.allclose(df["prob"].to_numpy(), 0.0)


def test_prob_is_zero_when_all_prior_points_outside_box_higher_dim():
    """Higher dim: All tile probabilities are zero when all prior points lie outside the grid."""
    h = 4  # a 4 x 4 x 4 grid
    xmin, xmax, ymin, ymax, zmin, zmax = 0, 1, 0, 1, 0, 1
    N = 200

    # All points lie strictly outside the box (to the right and above)
    x_prior = np.full(N, 2)  # x > xmax
    y_prior = np.full(N, 1.2)  # y > ymax
    z_prior = np.full(N, -0.2)  # z < zmin
    prior_data = np.column_stack([x_prior, y_prior, z_prior])

    q_prior_data = np.zeros((N, 2))
    q_data = np.zeros((10, 2))
    bin_edges = np.array([[0.0, 0.0], [1.0, 1.0]])

    df = prob_over_grid(
        q_data=q_data,
        prior_data=prior_data,
        q_prior_data=q_prior_data,
        bin_edges=bin_edges,
        grid_mins=np.array([xmin, ymin, zmin]),
        grid_maxes=np.array([xmax, ymax, zmax]),
        h=h,
    )

    # Every tile should receive zero probability mass
    assert np.allclose(df["prob"].to_numpy(), 0.0)
