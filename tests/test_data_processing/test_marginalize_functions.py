"""Unit tests for marginalizing functions in src.data_processing.py."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data_processing import (
    marginalize_out_lambda,
    marginalize_out_U,
    marginalize_scp_df,
)
from src.estimate_functions import synthetic_scp
from tests.test_estimate_functions.test_estimate_functions import EstimationParams
from tests.test_estimate_functions.test_estimate_functions_higher_dim import (
    gaussian_tgd_3d,
    uniform_prior_3d,
)

XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = -8, 8, -8, 8, 0, 1


FIXED_SCP_PARAMS: EstimationParams = {
    "J": 10000,
    "M": 20,
    "Q": lambda x, y, z: np.array([x**2 + 3 * y**2 - z, x + z]),
    "grid_mins": np.array([XMIN, YMIN, ZMIN]),
    "grid_maxes": np.array([XMAX, YMAX, ZMAX]),
    "lambda_grid_size": 10,
    "seed": 20251028,
}

probs_df_3d, _, _ = synthetic_scp(
    K=10000, tgd=gaussian_tgd_3d, prior=uniform_prior_3d, **FIXED_SCP_PARAMS
)

"""
Let's create a custom dataframe that has the structure of an SCP solution but
not actually from the algorithm. We grid up the cube [0, 1]^3 into rows, sorted
by x, y, then z axes. Let the probability just be the numbers 1 through 27 of
the 27 resulting rows. The subsequent tests using this are vert manual, and the
easiest way to understand the set-up is to write it up in a spreadsheet.
"""
custom_prob_df = pd.DataFrame({
    "xmin": np.array(9 * [0] + 9 * [1] + 9 * [2]),
    "xmax": np.array(9 * [1] + 9 * [2] + 9 * [3]),
    "ymin": np.array(3 * (3 * [0] + 3 * [1] + 3 * [2])),
    "ymax": np.array(3 * (3 * [1] + 3 * [2] + 3 * [3])),
    "zmin": np.array(9 * [0, 1, 2]),
    "zmax": np.array(9 * [1, 2, 3]),
    "prob": np.array(list(range(1, 28))),
})


def test_marginalize_U_or_lambda() -> None:
    """Check marginalize_out_U and marginalize_out_Lambda.

    Both functions should agree with directly applying marginalize_scp_df.
    """
    assert set(probs_df_3d.columns) == {
        "xmin",
        "xmax",
        "ymin",
        "ymax",
        "zmin",
        "zmax",
        "prob",
    }

    marg_out_z = marginalize_scp_df(probs_df_3d, ["x", "y"])
    marg_out_z_alternative = marginalize_out_U(probs_df_3d)
    assert_frame_equal(marg_out_z, marg_out_z_alternative)
    assert list(marg_out_z.columns) == ["xmin", "xmax", "ymin", "ymax", "prob"]

    marg_out_x_and_y = marginalize_scp_df(probs_df_3d, ["z"])
    marg_out_x_and_y_alternative = marginalize_out_lambda(probs_df_3d)
    assert_frame_equal(
        marg_out_x_and_y, marg_out_x_and_y_alternative, rtol=1e-5, atol=1e-8
    )
    assert list(marg_out_x_and_y.columns) == ["zmin", "zmax", "prob"]

    marg_out_y = marginalize_scp_df(probs_df_3d, ["x", "z"])
    assert list(marg_out_y.columns) == ["xmin", "xmax", "zmin", "zmax", "prob"]


def test_marginalize_scp_df_over_one_var() -> None:
    """Test marginalize_scp_df with custom prob_df, one variable removed."""
    marginalize_out_z = marginalize_scp_df(custom_prob_df, ["x", "y"])

    expected_df_no_z = pd.DataFrame({
        "xmin": np.array(3 * [0] + 3 * [1] + 3 * [2]),
        "xmax": np.array(3 * [1] + 3 * [2] + 3 * [3]),
        "ymin": np.array(3 * [0, 1, 2]),
        "ymax": np.array(3 * [1, 2, 3]),
        "prob": np.array([6, 15, 24, 33, 42, 51, 60, 69, 78]),
    })
    assert_frame_equal(marginalize_out_z, expected_df_no_z)

    marginalize_out_x = marginalize_scp_df(custom_prob_df, ["y", "z"])
    expected_df_no_x = pd.DataFrame({
        "ymin": np.array(3 * [0] + 3 * [1] + 3 * [2]),
        "ymax": np.array(3 * [1] + 3 * [2] + 3 * [3]),
        "zmin": np.array(3 * [0, 1, 2]),
        "zmax": np.array(3 * [1, 2, 3]),
        "prob": np.array([30, 33, 36, 39, 42, 45, 48, 51, 54]),
    })
    assert_frame_equal(marginalize_out_x, expected_df_no_x)


def test_marginalize_scp_df_over_two_vars() -> None:
    """Test marginalize_scp_df with custom prob_df, two variables removed."""
    marginalize_out_y_and_z = marginalize_scp_df(custom_prob_df, ["x"])
    expected_df_only_x = pd.DataFrame({
        "xmin": np.array([0, 1, 2]),
        "xmax": np.array([1, 2, 3]),
        "prob": np.array([45, 126, 207]),
    })
    assert_frame_equal(marginalize_out_y_and_z, expected_df_only_x)

    marginalize_out_x_and_y = marginalize_scp_df(custom_prob_df, ["z"])
    expected_df_only_z = pd.DataFrame({
        "zmin": np.array([0, 1, 2]),
        "zmax": np.array([1, 2, 3]),
        "prob": np.array([117, 126, 135]),
    })
    assert_frame_equal(marginalize_out_x_and_y, expected_df_only_z)


def test_marginalize_scp_df_edge_cases() -> None:
    """Test marginalize_scp_df with custom prob_df, unexpected inputs.

    Try marginalizing out all columns (does nothing), adding a duplicate column
    (ignores duplicate), marginalizing a fake column (KeyError).
    """
    marginalize_everything = marginalize_scp_df(custom_prob_df, ["x", "y", "z"])
    assert_frame_equal(marginalize_everything, custom_prob_df)

    # Adding a duplicate of variable makes no difference
    marginalize_dup_vars = marginalize_scp_df(custom_prob_df, ["y", "y", "z"])
    marginalize_out_x = marginalize_scp_df(custom_prob_df, ["y", "z"])
    assert_frame_equal(marginalize_dup_vars, marginalize_out_x)

    with pytest.raises(KeyError):
        marginalize_scp_df(custom_prob_df, ["w"])
