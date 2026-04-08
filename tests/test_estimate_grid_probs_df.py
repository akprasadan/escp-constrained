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

from src.output_functions import make_partition
from src.probability_functions import prob_over_grid


def dummy_data(
    grid_mins: np.ndarray,
    grid_maxes: np.ndarray,
    seed: int = 0,
    M_obs: int = 200,
    N_prior: int = 500,
    input_dim=2,
    output_dim=1,
):
    """A helper function to generate 1D data.

    - prior_data in [0, 1]^input_dim. If Lambda contains this entire set, probabilities will sum to 1.
    - q_prior_data and prior_data share length N_prior.
    - q values in [0, 1]^output_dim, bin_edges cover [0, 1]^output_dim.
    """

    rng = np.random.default_rng(seed)
    q_data = rng.random((M_obs, output_dim))  # Over [0, 1]^output_dim
    prior_data = rng.random((N_prior, input_dim))  # Over [0, 1]^input_dim

    q_prior_data = rng.random((N_prior, output_dim))  # Over [0, 1]^output_dim
    # Use bin edges on [-0.1, 1.1]^output_dim
    bin_edges = make_partition(
        min_vals=-0.1 * np.ones(output_dim),
        max_vals=1.1 * np.ones(output_dim),
        M=5,
    )

    return q_data, prior_data, q_prior_data, bin_edges, grid_mins, grid_maxes


def test_rows_are_h_squared_loop():
    """Output dataframe should have h^2 rows for each probability,
    and in general d-dimensions for Lambda, h^d."""
    input_output_dim_options = [(1, 1), (2, 1), (3, 2)]
    h_options = [1, 2, 3, 5, 6, 7, 8]

    for input_dim, output_dim in input_output_dim_options:
        for h in h_options:
            dummy_inputs = dummy_data(
                grid_mins=np.zeros(input_dim),  # [0, 0], e.g.
                grid_maxes=10 * np.ones(input_dim),  # [10, 10], e.g.
                input_dim=input_dim,
                output_dim=output_dim,
            )
            df = prob_over_grid(*dummy_inputs, h=h)
            assert len(df) == h**input_dim


def test_invalid_inputs_raise():
    """Invalid inputs raise ValueError (bad bounds or invalid h)."""

    bad_bounds = [
        ([1], [0]),  # bad x in 1d
        ([1, 1], [0, 10]),  # bad x in 2d
        ([2, 0], [1, 10]),  # bad x in 2d
        ([2, 0], [-1, 10]),  # bad x in 2d
        ([0, 10], [10, 5]),  # bad y in 2d
        ([0, 9], [10, 5]),  # bad y in 2d
        ([0, 9], [10, -1]),  # bad y in 2d
        ([3, 9], [2, 5]),  # bad x and bad y in in 2d
        ([1, 1, 0], [0, 10, 1]),  # bad x in 3d
        ([2, 0, 5], [1, 10, 15]),  # bad x in 3d
        ([0, 10, 0], [10, 5, 1]),  # bad y in 3d
        ([0, 9, 2], [10, 5, 4]),  # bad y in 3d
        ([0, 0, 5], [10, 5, 1]),  # bad z in 3d
        ([0, 2, 2], [10, 5, 0]),  # bad z in 2d
    ]
    output_dims = [1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1]

    invalid_dummy_data = [
        dummy_data(
            grid_mins=np.array(mins),
            grid_maxes=np.array(maxes),
            input_dim=len(mins),
            output_dim=output_dim,
        )
        for (mins, maxes), output_dim in zip(bad_bounds, output_dims)
    ]

    for bad_dummy in invalid_dummy_data:
        with pytest.raises(ValueError, match="min_vals >= max_vals"):
            prob_over_grid(*bad_dummy, h=2)

    invalid_hs = [0, -3, 1.5, "4"]

    good_bounds = [([0], [1]), ([0, 0], [1, 1]), ([0, 0, 0], [1, 1, 1])]

    good_dummy_inputs = [
        dummy_data(
            grid_mins=np.array(mins),
            grid_maxes=np.array(maxes),
            input_dim=len(mins),
            output_dim=1,
        )
        for mins, maxes in good_bounds
    ]
    for bad_h in invalid_hs:
        for good_dummy in good_dummy_inputs:
            with pytest.raises(ValueError, match="h must be an integer"):
                prob_over_grid(*good_dummy, h=bad_h)


def test_q_prior_data_prior_data_length_mismatch_raises():
    """q_prior_data and prior_data must have the same length. Higher dim tests included too."""

    def test_bad_length(input_dim: int, output_dim: int) -> None:
        q_data, prior_data, q_prior_data, bin_edges, grid_mins, grid_maxes = (
            dummy_data(
                grid_mins=np.zeros(input_dim),
                grid_maxes=np.ones(input_dim),
                input_dim=input_dim,
                output_dim=output_dim,
            )
        )
        # Truncate prior_data intentionally
        prior_data_bad = prior_data[:-1]

        with pytest.raises(ValueError, match="not same length"):
            prob_over_grid(
                q_data,
                prior_data_bad,
                q_prior_data,
                bin_edges,
                grid_mins,
                grid_maxes,
                h=3,
            )

    test_bad_length(1, 1)  # 1D
    test_bad_length(2, 1)  # 2D
    test_bad_length(2, 2)  # 2D
    test_bad_length(3, 2)  # 3D
    test_bad_length(3, 1)  # 3D


def test_min_max_match_inputs():
    """Min/Max boundaries must match the input box. In higher dimensions too."""

    xmin, xmax = -3.25, 3.75
    ymin, ymax = 5.5, 10
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

    dummy_inputs = dummy_data(
        grid_mins=np.array([xmin, ymin, zmin]),
        grid_maxes=np.array([xmax, ymax, zmax]),
        seed=1,
        input_dim=3,
        output_dim=2,
    )
    df = prob_over_grid(*dummy_inputs, h=3)
    assert (df["xmin"].min(), df["xmax"].max()) == (xmin, xmax)
    assert (df["ymin"].min(), df["ymax"].max()) == (ymin, ymax)
    assert (df["zmin"].min(), df["zmax"].max()) == (zmin, zmax)


def test_x_and_y_grids_respect_bounds_and_spacing():
    """X and Y grid edges start, end, and step according to their respective bounds."""
    xmin, xmax = 0, 1
    ymin, ymax = 10, 30
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
    # Global bounds for each axis
    xmin, xmax = 0, 1
    ymin, ymax = 10, 30
    zmin, zmax = -1, 5

    dummy_inputs = dummy_data(
        grid_mins=np.array([xmin, ymin, zmin]),
        grid_maxes=np.array([xmax, ymax, zmax]),
        seed=2,
        input_dim=3,
        output_dim=2,
    )
    h = 4

    df = prob_over_grid(*dummy_inputs, h=h)

    # vector of minimums for each axis stored in df
    xmins, ymins, zmins = [
        np.sort(df[min].unique()) for min in ["xmin", "ymin", "zmin"]
    ]

    dx, dy, dz = (xmax - xmin) / h, (ymax - ymin) / h, (zmax - zmin) / h

    assert dx != dy and dy != dz and dx != dz

    def check_xyz_grid(minimums, global_low, global_high, delta):
        # x, y, and z grid starts/ends at the left/right bound
        assert np.isclose(minimums[0], global_low)
        assert np.isclose(minimums[-1] + delta, global_high)

        # x, y, z grid edges are uniformly spaced by dx, dy, dz
        assert np.allclose(np.diff(minimums), delta)

    check_xyz_grid(xmins, xmin, xmax, dx)
    check_xyz_grid(ymins, ymin, ymax, dy)
    check_xyz_grid(zmins, zmin, zmax, dz)


def test_grid_has_expected_number_of_unique_edges():
    """Grid has h unique xmin/xmax and ymin/ymax edges. In higher dimensions too."""
    h = 7
    xmin, xmax, ymin, ymax = -7, 5, 2, 11
    dummy_inputs = dummy_data(
        grid_mins=np.array([xmin, ymin]),
        grid_maxes=np.array([xmax, ymax]),
        seed=3,
    )

    df = prob_over_grid(*dummy_inputs, h=h)
    for edge in ["xmin", "xmax", "ymin", "ymax"]:
        assert len(df[edge].unique()) == h

    # Higher dimensions too
    xmin, xmax, ymin, ymax, zmin, zmax = -7, 5, 2, 11, -1, 3
    dummy_inputs = dummy_data(
        grid_mins=np.array([xmin, ymin, zmin]),
        grid_maxes=np.array([xmax, ymax, zmax]),
        seed=3,
        input_dim=3,
        output_dim=2,
    )

    df = prob_over_grid(*dummy_inputs, h=h)

    for edge in ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]:
        assert len(df[edge].unique()) == h


def test_grid_edges_are_contiguous():
    """Adjacent grid edges touch with no gaps or overlaps."""
    h = 7
    bounds = -7, 5, 2, 11, 0, 3  # xmin, xmax, ymin, ymax, zmin, zmax

    for input_dim in [1, 2, 3]:
        # Prepare dummy data for this dimension
        dummy_inputs = dummy_data(
            grid_mins=np.array(
                bounds[::2][:input_dim]
            ),  # even indices up to input dum
            grid_maxes=np.array(
                bounds[1::2][:input_dim]
            ),  # odd indices up to input dim
            seed=3,
            input_dim=input_dim,
            output_dim=max(input_dim - 1, 1),
        )
        df = prob_over_grid(*dummy_inputs, h=h)

        # Extract relevant axes for this dimension
        axes = ("x", "y", "z")
        axes = axes[:input_dim]

        # Save a dictionary of minimum vectors for each axis:
        # min_vectors = {"x" : np.sort(df["xmin"].unique()),
        #                "y": np.sort(df["ymin"].unique(), ... }
        # and similary save a dictionary of maximum vectors
        # This avoids hardcoding "xmin", "ymin", ....
        min_vectors = {ax: np.sort(df[f"{ax}min"].unique()) for ax in axes}
        max_vectors = {ax: np.sort(df[f"{ax}max"].unique()) for ax in axes}

        # Check that adjacent edges touch
        # E.g., The second element of xmin is the first element of xmax, and so on,
        # but for each of x, y, z
        for min_vec, max_vec in zip(min_vectors.values(), max_vectors.values()):
            assert np.allclose(min_vec[1:], max_vec[:-1])


def test_grid_edges_have_uniform_spacing():
    """Grid edges are evenly spaced along each axis."""
    h = 7
    bounds = -7, 5, 2, 11, 0, 3  # xmin, xmax, ymin, ymax, zmin, zmax

    for input_dim in [1, 2, 3]:
        dummy_inputs = dummy_data(
            grid_mins=np.array(
                bounds[::2][:input_dim]
            ),  # Even indices up to dim
            grid_maxes=np.array(
                bounds[1::2][:input_dim]
            ),  # Odd indices up to dim
            seed=3,
            input_dim=input_dim,
            output_dim=max(
                input_dim - 1, 1
            ),  # One less dim unless already at 1
        )

        df = prob_over_grid(*dummy_inputs, h=h)
        axis_names = ["x", "y", "z"][:input_dim]

        # E.g., first element is xmins = np.sort(df["xmin"].unique())
        min_vectors = [np.sort(df[f"{ax}min"].unique()) for ax in axis_names]

        # E.g., first element is dx = (xmax - xmin)/h
        expected_widths = [
            (bounds[2 * i + 1] - bounds[2 * i]) / h for i in range(input_dim)
        ]

        for i in range(input_dim):
            # Check xmax - xmin = np.diff(xmins), e.g.
            assert np.allclose(np.diff(min_vectors[i]), expected_widths[i])


def test_tiles_have_constant_positive_dimensions():
    """All grid tiles have equal positive width and height.
    Tests 1D, 2D, and 3D Lambdas, and 1D, 1D, and 2D D's, respectively.
    """
    h = 4
    bounds = -3.5, 4.5, -10, -2, 0, 2  # xmin, xmax, ymin, ymax, zmin, zmax
    for input_dim in [1, 2, 3]:
        dummy_inputs = dummy_data(
            grid_mins=np.array(
                bounds[::2][:input_dim]
            ),  # Even indices up to dim
            grid_maxes=np.array(
                bounds[1::2][:input_dim]
            ),  # Odd indices up to dim
            seed=4,
            input_dim=input_dim,
            output_dim=max(
                input_dim - 1, 1
            ),  # One less dim unless already at 1
        )

        df = prob_over_grid(*dummy_inputs, h=h)

        # This stores a list [dx = (xmax - xmin)/ h, dy = (ymax - ymin)/h, ...]
        # according to the dimension
        # Since the bounds are stored by 2 per axis, we look at 2i + 1 and 2i
        # indices.
        expected_widths = [
            (bounds[2 * i + 1] - bounds[2 * i]) / h for i in range(0, input_dim)
        ]
        axis_names = ["x", "y", "z"][:input_dim]

        # Store list of form df["xmax"] - df["xmin"] for each axis, converted to
        # a numpy array
        actual_widths = [
            (df[f"{ax}max"] - df[f"{ax}min"]).to_numpy() for ax in axis_names
        ]
        for i in range(input_dim):
            # Width differences match expectations
            # Note actual_widths[i] is array, while expected_widths[i] is a float
            assert np.allclose(actual_widths[i], expected_widths[i])

            # widths for each axis are all positive
            assert np.all(actual_widths[i] > 0)


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
    dummy_inputs = dummy_data(
        grid_mins=np.array([0, -2, 1]),
        grid_maxes=np.array([3, 2, 5]),
        seed=5,
        input_dim=3,
        output_dim=2,
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
    too low, this will fail. For example, this test fails if we replace (50, 200) with (50, 10).
    Tests 1D, 2D, and 3D Lambdas, and 1D, 1D, and 2D D's, respectively.
    """
    boxes = [
        (0, 1),  # 1D
        (-0.5, 1.5),
        (0, 1, 0, 1),  # 2D
        (-0.5, 1.5, -0.5, 1.5),
        (0, 1, 0, 1, 0, 1),  # 3D
        (-0.5, 1.5, -0.5, 1.5, -1, 2),
    ]

    for seed in [0, 1, 2, 7]:
        # Make N_prior large enough
        for M_obs, N_prior in [(50, 200), (200, 500), (1000, 2000)]:
            for box in boxes:
                dummy_inputs = dummy_data(
                    grid_mins=np.array(box[::2]),  # even entries
                    grid_maxes=np.array(box[1::2]),  # odd entries
                    seed=seed,
                    M_obs=M_obs,
                    N_prior=N_prior,
                    input_dim=len(box) // 2,
                    output_dim=max(
                        len(box) // 2 - 1, 1
                    ),  # reduce by 1 or keep at 1
                )
                df = prob_over_grid(*dummy_inputs, h=1)
                assert len(df) == 1
                assert np.isclose(df["prob"].iloc[0], 1)


def test_probs_sum_to_one_many_random_cases():
    """
    Sum of all probabilities is 1 across many seeds/boxes/h, for prior having subset of
    Lambda support, and prior sample size sufficiently large to fill all bins.
    Tests 1D, 2D, and 3D Lambdas, and 1D, 1D, and 2D D's, respectively.
    """

    boxes = [
        (0, 1),  # 1D: xmin, xmax
        (0, 1, 0, 1),  # 2D: xmin, xmax, ymin, ymax
        (-0.5, 1.5, -0.5, 1.5),
        (0, 10, 0, 10),
        (0, 1, 0, 1, 0, 1),  # 3D: xmin, xmax, ymin, ymax, zmin, zmax
        (-0.5, 1.5, -0.5, 1.5, -1, 2),
        (0, 10, 0, 10, -0.1, 11),
    ]
    for seed in [0, 1, 2, 11]:
        for M_obs, N_prior in [(200, 500), (1000, 2000), (2000, 5000)]:
            for h in [1, 2, 3, 5, 6]:  # don't make h large, is expensive
                for bounds in boxes:
                    dummy_inputs = dummy_data(
                        grid_mins=np.array(bounds[::2]),  # even elements
                        grid_maxes=np.array(bounds[1::2]),  # odd elements
                        seed=seed,
                        M_obs=M_obs,
                        N_prior=N_prior,
                        input_dim=len(bounds) // 2,
                        output_dim=max(
                            len(bounds) // 2 - 1, 1
                        ),  # Reduce output dim by one or keep at 1
                    )

                    df = prob_over_grid(*dummy_inputs, h=h)
                    assert np.isclose(df["prob"].sum(), 1)


def test_prob_mass_is_assigned_to_correct_x_column():
    """Probability mass falls entirely in the left column when all x_prior < mid-point."""
    h = 2  # a 2 x 2 grid
    xmin, xmax, ymin, ymax = 0, 1, 0, 1
    N = 200

    rng = np.random.default_rng(seed=20251028)
    # All points lie strictly in the left x-tile (x < 0.5)
    x_prior = np.full(N, 0.25)
    y_prior = rng.random(N)
    prior_data = np.column_stack([x_prior, y_prior])

    q_prior_data = rng.random((N, 1))
    q_data = rng.random((10, 1))
    bin_edges = np.array([[0], [1]])

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
    assert np.allclose(col_sums, [1, 0], atol=1e-12)


def test_prob_mass_is_assigned_to_correct_bin_higher_dim():
    """Probability mass falls entirely in the correct bins when all x_prior > mid-point,
    y_prior < mid_point."""
    h = 2  # a 2 x 2 x 2 grid
    xmin, xmax, ymin, ymax, zmin, zmax = 0, 1, 0, 1, 0, 1
    N = 200
    rng = np.random.default_rng(seed=20251028)
    # All points lie strictly in the right x-tiles (x > 0.5) and y-tiles (y < 0.5)
    x_prior = np.full(N, 0.75)
    y_prior = np.full(N, 0.25)
    z_prior = rng.random(N)
    prior_data = np.column_stack([x_prior, y_prior, z_prior])

    q_prior_data = rng.random((N, 2))
    q_data = rng.random((10, 2))
    bin_edges = np.array([[0, 0], [1, 1]])

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
    xmin, xmax, ymin, ymax = 0, 1, 0, 1
    N = 200

    rng = np.random.default_rng(seed=20251028)
    # All points with y in [0, 0.5)
    y_prior = np.full(N, 0.25)
    x_prior = rng.random(N)
    prior_data = np.column_stack([x_prior, y_prior])

    # Make per-tile prob = fraction of points in tile (single q-bin)
    q_prior_data = rng.random((N, 1))
    q_data = rng.random((10, 1))
    bin_edges = np.array([[0], [1]])

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
    assert np.allclose(row_sums, [1, 0])


def test_prob_mass_is_assigned_to_correct_single_bin_higher_dim():
    """Higher dim: Probability mass falls entirely in single bin (x, y > 0.5, z < 0.5)"""
    h = 2  # a 2 x 2 x 2 grid
    xmin, xmax, ymin, ymax, zmin, zmax = 0, 1, 0, 1, 0, 1
    N = 200
    rng = np.random.default_rng(seed=20251028)
    # All points lie strictly in the left z-tiles (z > 0.5)
    z_prior = np.full(N, 0.25)
    y_prior = np.full(N, 0.75)
    x_prior = np.full(N, 0.75)
    prior_data = np.column_stack([x_prior, y_prior, z_prior])

    q_prior_data = rng.random((N, 2))
    q_data = rng.random((10, 2))
    bin_edges = np.array([[0, 0], [1, 1]])

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
    xmin, xmax, ymin, ymax = 0, 1, 0, 1
    N = 200

    # All points lie strictly outside the box (to the right and above)
    x_prior = np.full(N, 2)  # x > xmax
    y_prior = np.full(N, 2)  # y > ymax
    prior_data = np.column_stack([x_prior, y_prior])

    q_prior_data = np.zeros(N)[:, None]
    q_data = np.zeros(10)[:, None]
    bin_edges = np.array([[0], [1]])

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
    assert np.allclose(df["prob"].to_numpy(), 0)


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
    bin_edges = np.array([[0, 0], [1, 1]])

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
    assert np.allclose(df["prob"].to_numpy(), 0)
