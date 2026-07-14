"""
Unit tests for `src.estimate_functions.py` with higher dimensions.

Check that output dataframes have right properties, and that the synthetic
(generate Q by drawing from the TGD) and empirical (pass in Q directly)
estimations match. Doesn't directly test correctness of SCP algorithm (other
test modules do).
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.estimate_functions import empirical_scp, synthetic_scp
from src.sampler_functions import sample_multivar_gaussian, unif_dist_hyperrect
from tests.test_estimate_functions.test_estimate_functions import EstimationParams

XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = -8, 8, -8, 8, 0, 1


FIXED_SCP_PARAMS_3d: EstimationParams = {
    "J": 10000,
    "M": 20,
    "Q": lambda x, y, z: np.array([x**2 + 3 * y**2 - z, x + z]),
    "grid_mins": np.array([XMIN, YMIN, ZMIN]),
    "grid_maxes": np.array([XMAX, YMAX, ZMAX]),
    "lambda_grid_size": 10,
    "seed": 20251028,
}


def gaussian_tgd_3d(rng: np.random.Generator, J: int) -> np.ndarray:
    """Create 3D Gaussian TGD.

    Returns
    -------
    (J, 3) numpy array
    """
    mean = np.array([2, -1, 1])
    cov = np.diag([2, 1, 1.5])
    return sample_multivar_gaussian(mean, cov, rng, J)


def gaussian_tgd_huge_variance_3d(rng: np.random.Generator, J: int) -> np.ndarray:
    """Create 3D Gaussian TGD with high variance.

    Returns
    -------
    (J, 3) numpy array
    """
    mean = np.array([5, -5, 0])
    cov = np.diag([10, 10, 7])
    return sample_multivar_gaussian(mean, cov, rng, J)


def gaussian_tgd_4d(rng: np.random.Generator, J: int) -> np.ndarray:
    """Create 4D Gaussian TGD.

    Returns
    -------
    (J, 4) numpy array
    """
    mean = np.array([0, 2, -1, 1])
    cov = np.diag([1, 2, 1, 1.5])
    return sample_multivar_gaussian(mean, cov, rng, J)


def gaussian_tgd_5d(rng: np.random.Generator, J: int) -> np.ndarray:
    """Create 5D Gaussian TGD.

    Returns
    -------
    (J, 5) numpy array
    """
    mean = np.array([0, 2, -1, 1, 0])
    cov = np.diag([1, 2, 1, 1.5, 1])
    return sample_multivar_gaussian(mean, cov, rng, J)


def uniform_prior_3d(rng: np.random.Generator, J: int) -> np.ndarray:
    """Make simple uniform prior.

    Returns
    -------
    (J, 3) numpy array
    """
    mins = np.array(3 * [-5])
    maxes = np.array(3 * [5])
    return unif_dist_hyperrect(mins, maxes, rng, J)


def huge_uniform_prior_3d(rng: np.random.Generator, J: int) -> np.ndarray:
    """Make uniform prior with larger domain.

    Returns
    -------
    (J, 3) numpy array
    """
    mins = np.array(3 * [-20])
    maxes = np.array(3 * [20])
    return unif_dist_hyperrect(mins, maxes, rng, J)


# Our set of 8 experiments. For each fixed choice of distribution, we run an
# experiment using a synthetic TGD, then feed that into the empirical
# experiment.

# 1(A) Synthetic with regular TGD, regular prior
probs_df, prior_df, tgd_df = synthetic_scp(
    K=10000, tgd=gaussian_tgd_3d, prior=uniform_prior_3d, **FIXED_SCP_PARAMS_3d
)


# 1(B) Empirical: Regular TGD, regular prior
probs_df_emp, prior_df_emp = empirical_scp(
    q_data=tgd_df[["q_x", "q_y"]].to_numpy(),
    prior=uniform_prior_3d,
    **FIXED_SCP_PARAMS_3d,
)


# 2(A) Synthetic with regular TGD, huge prior
probs_df_reg_tgd_huge_pr, prior_df_reg_tgd_huge_pr, tgd_df_reg_tgd_huge_pr = (
    synthetic_scp(
        K=10000, tgd=gaussian_tgd_3d, prior=huge_uniform_prior_3d, **FIXED_SCP_PARAMS_3d
    )
)

# 2(B) Empirical with regular TGD, huge prior
probs_df_reg_tgd_huge_pr_emp, prior_df_reg_tgd_huge_pr_emp = empirical_scp(
    q_data=tgd_df_reg_tgd_huge_pr[["q_x", "q_y"]].to_numpy(),
    prior=huge_uniform_prior_3d,
    **FIXED_SCP_PARAMS_3d,
)

# 3(A) Synthetic with huge TGD, regular prior
probs_df_huge_tgd_reg_pr, prior_df_huge_tgd_reg_pr, tgd_df_huge_tgd_reg_pr = (
    synthetic_scp(
        K=10000,
        tgd=gaussian_tgd_huge_variance_3d,
        prior=uniform_prior_3d,
        **FIXED_SCP_PARAMS_3d,
    )
)

# 3(B) Empirical with huge TGD, regular prior
probs_df_huge_tgd_reg_pr_emp, prior_df_huge_tgd_reg_pr_emp = empirical_scp(
    q_data=tgd_df_huge_tgd_reg_pr[["q_x", "q_y"]].to_numpy(),
    prior=uniform_prior_3d,
    **FIXED_SCP_PARAMS_3d,
)

# 4(A) Synthetic with huge TGD, huge prior
(probs_df_huge_tgd_huge_pr, prior_df_huge_tgd_huge_pr, tgd_df_huge_tgd_huge_pr) = (
    synthetic_scp(
        K=10000,
        tgd=gaussian_tgd_huge_variance_3d,
        prior=huge_uniform_prior_3d,
        **FIXED_SCP_PARAMS_3d,
    )
)

# 4(B) Empirical with huge TGD, huge prior
probs_df_huge_tgd_huge_pr_emp, prior_df_huge_tgd_huge_pr_emp = empirical_scp(
    q_data=tgd_df_huge_tgd_huge_pr[["q_x", "q_y"]].to_numpy(),
    prior=huge_uniform_prior_3d,
    **FIXED_SCP_PARAMS_3d,
)

# A single 4D --> 1D experiment, to check columns for Lambda (4D) and q (1D)
probs_df_4D, prior_df_4D, tgd_df_4d = synthetic_scp(
    K=1000,
    J=500,
    M=20,
    Q=lambda x, y, z, w: (x + y + z + w) / 50,
    tgd=gaussian_tgd_4d,
    prior=gaussian_tgd_4d,
    grid_mins=np.array([-20, -20, -20, -20]),
    grid_maxes=np.array([20, 20, 20, 20]),
    lambda_grid_size=10,
    seed=20251028,
)

# A single 5D --> 2D experiment, to check columns for Lambda (5D) and q (2D)
probs_df_5D, prior_df_5D, tgd_df_5d = synthetic_scp(
    K=50,
    J=100,
    M=10,
    Q=lambda x, y, z, w, r: np.array([x + y + z + w + r, x]) / 50,
    tgd=gaussian_tgd_5d,
    prior=gaussian_tgd_5d,
    grid_mins=np.array(5 * [-20]),
    grid_maxes=np.array(5 * [20]),
    lambda_grid_size=5,
    seed=20251028,
)


def test_estimated_prob_df_structure() -> None:
    """Validate prob_df output."""
    assert isinstance(probs_df, pd.DataFrame)
    expected_cols_3D = {"xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "prob"}
    assert set(probs_df.columns) == expected_cols_3D

    # 4D
    assert isinstance(probs_df_4D, pd.DataFrame)
    expected_cols_4D = expected_cols_3D | {"wmin", "wmax"}
    assert set(probs_df_4D.columns) == expected_cols_4D

    # 5D version should have different, unpacked columns
    assert isinstance(probs_df_5D, pd.DataFrame)
    expected_cols_5D = {"mins", "maxes", "prob"}
    assert set(probs_df_5D.columns) == expected_cols_5D


def test_estimated_prob_df_bounds() -> None:
    """Validate box positions in prob_df."""
    assert (probs_df["xmin"] < probs_df["xmax"]).all()
    assert (probs_df["ymin"] < probs_df["ymax"]).all()
    assert (probs_df["zmin"] < probs_df["zmax"]).all()


def test_estimated_probs_df_prob_sums_to_one() -> None:
    """
    Check that probability estimates sum to 1.

    Validate probabilities from prob_df, prob_df_huge_prior,
    prob_df_huge_tgd_and_prior, and the accompanying empirical versions of
    these.
    """
    # Experiment 1(A) and 1(B)
    assert np.isclose(probs_df["prob"].sum(), 1)
    assert np.isclose(probs_df_emp["prob"].sum(), 1)

    # 2(A) and 2(B)
    assert np.isclose(probs_df_reg_tgd_huge_pr["prob"].sum(), 1)
    assert np.isclose(probs_df_reg_tgd_huge_pr_emp["prob"].sum(), 1)

    # 3(A) and 3(B)
    assert np.isclose(probs_df_huge_tgd_reg_pr["prob"].sum(), 1)
    assert np.isclose(probs_df_huge_tgd_reg_pr_emp["prob"].sum(), 1)

    # 4(A) and 4(B)
    assert np.isclose(probs_df_huge_tgd_huge_pr["prob"].sum(), 1)
    assert np.isclose(probs_df_huge_tgd_huge_pr_emp["prob"].sum(), 1)


def test_prior_df_structure() -> None:
    """Validate prior_df output."""
    assert isinstance(prior_df, pd.DataFrame)
    assert set(prior_df.columns) == {"source", "x", "y", "z", "q_x", "q_y"}
    assert (prior_df["source"] == "prior").all()

    # 4D has no q_y since it is real-valued function
    assert isinstance(prior_df_4D, pd.DataFrame)
    assert set(prior_df_4D.columns) == {"source", "x", "y", "z", "w", "q_x"}
    assert (prior_df_4D["source"] == "prior").all()

    # Check 5D experiment does NOT have the same columns
    assert isinstance(prior_df_5D, pd.DataFrame)
    assert set(prior_df_5D.columns) == {"source", "lambda", "q_x", "q_y"}
    assert (prior_df_5D["source"] == "prior").all()


def test_tgd_df_structure() -> None:
    """Validate tgd_df output."""
    assert isinstance(tgd_df, pd.DataFrame)
    assert set(tgd_df.columns) == {"source", "x", "y", "z", "q_x", "q_y"}
    assert (tgd_df["source"] == "tgd").all()

    # 4D is real-valued so no q_y
    assert isinstance(tgd_df_4d, pd.DataFrame)
    assert set(tgd_df_4d.columns) == {"source", "x", "y", "z", "w", "q_x"}
    assert (tgd_df_4d["source"] == "tgd").all()

    # Check 5D experiment does NOT have those columns
    assert isinstance(tgd_df_5d, pd.DataFrame)
    assert set(tgd_df_5d.columns) == {"source", "lambda", "q_x", "q_y"}
    assert (tgd_df_5d["source"] == "tgd").all()


def test_prior_df_q_matches_quadratic() -> None:
    """Check that prior_df correctly saved values of Q from x and y."""
    Q = [
        prior_df["x"] ** 2 + 3 * prior_df["y"] ** 2 - prior_df["z"],
        prior_df["x"] + prior_df["z"],
    ]
    q_vals = prior_df[["q_x", "q_y"]].to_numpy()
    assert np.allclose(np.array(Q).T, q_vals)


def test_tgd_df_q_matches_quadratic() -> None:
    """Check that tgd_df correctly saved values of Q from x and y."""
    Q = [
        tgd_df["x"] ** 2 + 3 * tgd_df["y"] ** 2 - tgd_df["z"],
        tgd_df["x"] + tgd_df["z"],
    ]
    q_vals = tgd_df[["q_x", "q_y"]].to_numpy()
    assert np.allclose(np.array(Q).T, q_vals)


def test_synthetic_and_empirical_match() -> None:
    """synthetic_scp() and empirical_scp() should give the same output."""
    # Compare 1(A) and 1(B)
    assert_frame_equal(probs_df, probs_df_emp, check_exact=True)
    assert_frame_equal(prior_df, prior_df_emp, check_exact=True)

    # Compare 2(A) and 2(B)
    assert_frame_equal(
        probs_df_reg_tgd_huge_pr, probs_df_reg_tgd_huge_pr_emp, check_exact=True
    )
    assert_frame_equal(
        prior_df_reg_tgd_huge_pr, prior_df_reg_tgd_huge_pr_emp, check_exact=True
    )

    # Compare 3(A) and 3(B)
    assert_frame_equal(
        probs_df_huge_tgd_reg_pr, probs_df_huge_tgd_reg_pr_emp, check_exact=True
    )
    assert_frame_equal(
        prior_df_huge_tgd_reg_pr, prior_df_huge_tgd_reg_pr_emp, check_exact=True
    )

    # Compare 4(A) and 4(B)
    assert_frame_equal(
        probs_df_huge_tgd_huge_pr, probs_df_huge_tgd_huge_pr_emp, check_exact=True
    )
    assert_frame_equal(
        prior_df_huge_tgd_huge_pr, prior_df_huge_tgd_huge_pr_emp, check_exact=True
    )


def test_SCP_bad_bounds() -> None:
    """Test bad inputs into SCP.

    Use invalid grid_mins or grid_maxes, either due to the wrong length,
    non-monotonic, or being partially empty.
    """
    bad_fixed_params = {
        **FIXED_SCP_PARAMS_3d,
        "prior": uniform_prior_3d,
        "grid_maxes": np.array([XMAX, YMAX]),  # missing zmax
    }
    q_data_fixed = tgd_df[["q_x", "q_y"]].to_numpy()
    with pytest.raises(ValueError, match="Incompatible lengths"):
        synthetic_scp(K=1000, tgd=gaussian_tgd_3d, **bad_fixed_params)
    with pytest.raises(ValueError, match="Incompatible lengths"):
        empirical_scp(q_data=q_data_fixed, **bad_fixed_params)

    bad_fixed_params = {
        **bad_fixed_params,
        "grid_maxes": np.array([XMAX, YMAX, ZMIN]),  # zmin == zmax now
    }
    with pytest.raises(ValueError, match="grid_mins >= grid_maxes"):
        synthetic_scp(K=1000, tgd=gaussian_tgd_3d, **bad_fixed_params)
    with pytest.raises(ValueError, match="grid_mins >= grid_maxes"):
        empirical_scp(q_data=q_data_fixed, **bad_fixed_params)

    bad_fixed_params = {**bad_fixed_params, "grid_maxes": [], "grid_mins": []}
    msg = r"cannot be empty"
    with pytest.raises(ValueError, match=msg):
        synthetic_scp(K=10000, tgd=gaussian_tgd_3d, **bad_fixed_params)
    with pytest.raises(ValueError, match=msg):
        empirical_scp(q_data=q_data_fixed, **bad_fixed_params)

    bad_fixed_params = {
        **bad_fixed_params,
        "grid_maxes": [],
        "grid_mins": np.array([1]),
    }
    msg = r"cannot be empty"
    with pytest.raises(ValueError, match=msg):
        synthetic_scp(K=10000, tgd=gaussian_tgd_3d, **bad_fixed_params)
    with pytest.raises(ValueError, match=msg):
        empirical_scp(q_data=q_data_fixed, **bad_fixed_params)
