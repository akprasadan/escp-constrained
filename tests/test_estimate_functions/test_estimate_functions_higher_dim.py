"""
Unit tests for `src.estimate_functions` with higher dimensions.

Check that output dataframes have right properties, and that the synthetic
(generate Q by drawing from the TGD) and empirical (pass in Q directly) estimations match.
Doesn't directly test correctness of eSCP algorithm (other test modules do).
"""

from collections.abc import Callable
from typing import TypedDict

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from src.estimate_functions import synthetic_scp
from src.estimate_functions import empirical_scp

XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = -8, 8, -8, 8, 0, 1


# Typed dictionary class to store parameters we hold fixed in each SCP run
# https://typing.python.org/en/latest/spec/callables.html#unpack-kwargs
class EstimationParams(TypedDict):
    J: int
    M: int
    Q: Callable[..., np.ndarray]
    grid_mins: np.ndarray
    grid_maxes: np.ndarray
    lambda_grid_size: int
    seed: int | None


FIXED_SCP_PARAMS: EstimationParams = {
    "J": 10000,
    "M": 20,
    "Q": lambda x, y, z: np.array([x**2 + 3 * y**2 - z, x + z]),
    "grid_mins": np.array([XMIN, YMIN, ZMIN]),
    "grid_maxes": np.array([XMAX, YMAX, ZMAX]),
    "lambda_grid_size": 10,
    "seed": 20251028,
}


def gaussian_tgd(rng: np.random.Generator, J: int):
    x = rng.normal(2, 2, size=J)
    y = rng.normal(-1, 1, size=J)
    z = rng.normal(1, 1.5, size=J)
    return np.column_stack((x, y, z))


def gaussian_tgd_huge_variance(rng: np.random.Generator, J: int):
    x = rng.normal(5, 10, size=J)
    y = rng.normal(-5, 10, size=J)
    z = rng.normal(0, 7, size=J)
    return np.column_stack((x, y, z))


def gaussian_tgd_4d(rng: np.random.Generator, J: int):
    w = rng.normal(0, 1, size=J)
    x = rng.normal(2, 2, size=J)
    y = rng.normal(-1, 1, size=J)
    z = rng.normal(1, 1.5, size=J)
    return np.column_stack((w, x, y, z))


def uniform_prior(rng: np.random.Generator, J: int):
    x = rng.uniform(-5, 5, size=J)
    y = rng.uniform(-5, 5, size=J)
    z = rng.uniform(-5, 5, size=J)
    return np.column_stack((x, y, z))


def huge_uniform_prior(rng: np.random.Generator, J: int):
    x = rng.uniform(-20, 20, size=J)
    y = rng.uniform(-20, 20, size=J)
    z = rng.uniform(-20, 20, size=J)
    return np.column_stack((x, y, z))


# Our set of 8 experiments. For each fixed choice of distribution,
# we run an experiment using a synthetic TGD, then feed that into the empirical experiment.

# 1(A) Synthetic with regular TGD, regular prior
probs_df, prior_df, tgd_df = synthetic_scp(
    K=10000, tgd=gaussian_tgd, prior=uniform_prior, **FIXED_SCP_PARAMS
)


# 1(B) Empirical: Regular TGD, regular prior
probs_df_emp, prior_df_emp = empirical_scp(
    q_data=tgd_df[["q_x", "q_y"]].to_numpy(),
    prior=uniform_prior,
    **FIXED_SCP_PARAMS
)


# 2(A) Synthetic with regular TGD, huge prior
probs_df_reg_tgd_huge_pr, prior_df_reg_tgd_huge_pr, tgd_df_reg_tgd_huge_pr = (
    synthetic_scp(
        K=10000, tgd=gaussian_tgd, prior=huge_uniform_prior, **FIXED_SCP_PARAMS
    )
)

# 2(B) Empirical with regular TGD, huge prior
probs_df_reg_tgd_huge_pr_emp, prior_df_reg_tgd_huge_pr_emp = empirical_scp(
    q_data=tgd_df_reg_tgd_huge_pr[["q_x", "q_y"]].to_numpy(),
    prior=huge_uniform_prior,
    **FIXED_SCP_PARAMS
)

# 3(A) Synthetic with huge TGD, regular prior
probs_df_huge_tgd_reg_pr, prior_df_huge_tgd_reg_pr, tgd_df_huge_tgd_reg_pr = (
    synthetic_scp(
        K=10000,
        tgd=gaussian_tgd_huge_variance,
        prior=uniform_prior,
        **FIXED_SCP_PARAMS
    )
)

# 3(B) Empirical with huge TGD, regular prior
probs_df_huge_tgd_reg_pr_emp, prior_df_huge_tgd_reg_pr_emp = empirical_scp(
    q_data=tgd_df_huge_tgd_reg_pr[["q_x", "q_y"]].to_numpy(),
    prior=uniform_prior,
    **FIXED_SCP_PARAMS
)

# 4(A) Synthetic with huge TGD, huge prior
(
    probs_df_huge_tgd_huge_pr,
    prior_df_huge_tgd_huge_pr,
    tgd_df_huge_tgd_huge_pr,
) = synthetic_scp(
    K=10000,
    tgd=gaussian_tgd_huge_variance,
    prior=huge_uniform_prior,
    **FIXED_SCP_PARAMS
)

# 4(B) Empirical with huge TGD, huge prior
probs_df_huge_tgd_huge_pr_emp, prior_df_huge_tgd_huge_pr_emp = empirical_scp(
    q_data=tgd_df_huge_tgd_huge_pr[["q_x", "q_y"]].to_numpy(),
    prior=huge_uniform_prior,
    **FIXED_SCP_PARAMS
)

# A single 4D --> 1D experiment, to check columns for Lambda (4D) and q (1D)
probs_df_4D, prior_df_4D, tgd_df_4d = synthetic_scp(
    K=1000,
    J=500,
    M=20,
    Q=lambda w, x, y, z: (w + x + y + z) / 50,
    tgd=gaussian_tgd_4d,
    prior=gaussian_tgd_4d,
    grid_mins=np.array([-20, -20, -20, -20]),
    grid_maxes=np.array([20, 20, 20, 20]),
    lambda_grid_size=10,
    seed=20251028,
)


def test_estimated_prob_df_structure():
    "Validate prob_df output."
    assert isinstance(probs_df, pd.DataFrame)
    expected_cols = set(
        ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax", "prob"]
    )
    assert set(probs_df.columns) == expected_cols

    # 4D version should have different columns
    assert isinstance(probs_df_4D, pd.DataFrame)
    expected_cols = set(["mins", "maxes", "prob"])
    assert set(probs_df_4D.columns) == expected_cols


def test_estimated_prob_df_bounds():
    "Validate box positions in prob_df."
    assert (probs_df["xmin"] < probs_df["xmax"]).all()
    assert (probs_df["ymin"] < probs_df["ymax"]).all()
    assert (probs_df["zmin"] < probs_df["zmax"]).all()


def test_estimated_probs_df_prob_sums_to_one():
    """Validate probabilities from prob_df, prob_df_huge_prior, prob_df_huge_tgd_and_prior,
    and the accompanying empirical versions of these."""

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


def test_prior_df_structure():
    "Validate prior_df output."
    assert isinstance(prior_df, pd.DataFrame)
    assert set(prior_df.columns) == {"source", "x", "y", "z", "q_x", "q_y"}
    assert (prior_df["source"] == "prior").all()

    # Check 4D experiment does NOT have those columns
    assert isinstance(prior_df_4D, pd.DataFrame)
    assert set(prior_df_4D.columns) == {"source", "lambda", "q_x"}
    assert (prior_df["source"] == "prior").all()


def test_tgd_df_structure():
    "Validate tgd_df output."
    assert isinstance(tgd_df, pd.DataFrame)
    assert set(tgd_df.columns) == {"source", "x", "y", "z", "q_x", "q_y"}
    assert (tgd_df["source"] == "tgd").all()

    # Check 4D experiment does NOT have those columns
    assert isinstance(tgd_df_4d, pd.DataFrame)
    assert set(tgd_df_4d.columns) == {"source", "lambda", "q_x"}
    assert (tgd_df_4d["source"] == "tgd").all()


def test_prior_df_q_matches_quadratic():
    "Check that prior_df correctly saved values of Q from x and y"
    Q = [
        prior_df["x"] ** 2 + 3 * prior_df["y"] ** 2 - prior_df["z"],
        prior_df["x"] + prior_df["z"],
    ]
    q_vals = prior_df[["q_x", "q_y"]].to_numpy()
    assert np.allclose(np.array(Q).T, q_vals)


def test_tgd_df_q_matches_quadratic():
    "Check that tgd_df correctly saved values of Q from x and y"
    Q = [
        tgd_df["x"] ** 2 + 3 * tgd_df["y"] ** 2 - tgd_df["z"],
        tgd_df["x"] + tgd_df["z"],
    ]
    q_vals = tgd_df[["q_x", "q_y"]].to_numpy()
    assert np.allclose(np.array(Q).T, q_vals)


def test_synthetic_and_empirical_match():
    """synthetic_scp() and empirical_scp() should give the same output"""
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
        probs_df_huge_tgd_huge_pr,
        probs_df_huge_tgd_huge_pr_emp,
        check_exact=True,
    )
    assert_frame_equal(
        prior_df_huge_tgd_huge_pr,
        prior_df_huge_tgd_huge_pr_emp,
        check_exact=True,
    )


def test_SCP_bad_bounds():
    """Use invalid grid_mins or grid_maxes, either due to the wrong length,
    non-monotonic, or being partially empty."""
    bad_fixed_params = {
        **FIXED_SCP_PARAMS,
        "prior": uniform_prior,
        "grid_maxes": np.array([XMAX, YMAX]),  # missing zmax
    }
    q_data_fixed = tgd_df[["q_x", "q_y"]].to_numpy()
    with pytest.raises(ValueError, match="Incompatible lengths"):
        synthetic_scp(K=1000, tgd=gaussian_tgd, **bad_fixed_params)
    with pytest.raises(ValueError, match="Incompatible lengths"):
        empirical_scp(q_data=q_data_fixed, **bad_fixed_params)

    bad_fixed_params = {
        **bad_fixed_params,
        "grid_maxes": np.array([XMAX, YMAX, ZMIN]),  # zmin == zmax now
    }
    with pytest.raises(ValueError, match="grid_mins >= grid_maxes"):
        synthetic_scp(K=1000, tgd=gaussian_tgd, **bad_fixed_params)
    with pytest.raises(ValueError, match="grid_mins >= grid_maxes"):
        empirical_scp(q_data=q_data_fixed, **bad_fixed_params)

    bad_fixed_params = {**bad_fixed_params, "grid_maxes": [], "grid_mins": []}
    msg = r"cannot be empty"
    with pytest.raises(ValueError, match=msg):
        synthetic_scp(K=10000, tgd=gaussian_tgd, **bad_fixed_params)
    with pytest.raises(ValueError, match=msg):
        empirical_scp(q_data=q_data_fixed, **bad_fixed_params)

    bad_fixed_params = {
        **bad_fixed_params,
        "grid_maxes": [],
        "grid_mins": np.array([1]),
    }
    msg = r"cannot be empty"
    with pytest.raises(ValueError, match=msg):
        synthetic_scp(K=10000, tgd=gaussian_tgd, **bad_fixed_params)
    with pytest.raises(ValueError, match=msg):
        empirical_scp(q_data=q_data_fixed, **bad_fixed_params)
