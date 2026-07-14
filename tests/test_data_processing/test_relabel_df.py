"""Unit tests for src.data_processing.py::relabel_2D_scp_df()."""

import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from src.data_processing import marginalize_scp_df, relabel_2D_scp_df
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


def test_relabel_2D_scp_df() -> None:
    """Test our relabel_2D_scp_df function."""
    with pytest.raises(ValueError, match="Too many columns in df"):
        # Probs_df has x, y, and z cols
        relabel_2D_scp_df(probs_df_3d, "y", "z")

    # now make function only have an x and z column
    marg_out_y = marginalize_scp_df(probs_df_3d, ["x", "z"])

    # Try using the wrong column
    with pytest.raises(ValueError, match="df missing y column"):
        relabel_2D_scp_df(marg_out_y, "y", "z")

    relabeled = relabel_2D_scp_df(marg_out_y, "x", "z")
    assert set(relabeled.columns) == {"xmin", "xmax", "ymin", "ymax", "prob"}

    # swap order leaves columns unchanged
    relabeled_v2 = relabel_2D_scp_df(marg_out_y, "z", "x")
    assert set(relabeled_v2.columns) == {"xmin", "xmax", "ymin", "ymax", "prob"}

    # But swapping order should give different result
    with pytest.raises(AssertionError):
        assert_frame_equal(marg_out_y, relabeled_v2)

    # Relabeling data already with x and y only should do nothing
    marg_out_z = marginalize_scp_df(probs_df_3d, ["x", "y"])
    relabeled_v3 = relabel_2D_scp_df(marg_out_z, "x", "y")
    assert_frame_equal(marg_out_z, relabeled_v3)

    # But swapping order should give different result even in this case
    relabeled_v4 = relabel_2D_scp_df(marg_out_z, "y", "x")
    with pytest.raises(AssertionError):
        assert_frame_equal(marg_out_z, relabeled_v4)
