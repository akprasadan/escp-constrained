"""
Unit tests for `src.estimate_functions.py` (2D).

Check that output dataframes have right properties, and that the synthetic
(generate Q by drawing from the TGD) and empirical (pass in Q directly)
estimations match. Doesn't directly test correctness of SCP algorithm (other
test modules do).
"""

from collections.abc import Callable
from typing import TypedDict

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.estimate_functions import empirical_scp, synthetic_scp
from src.sampler_functions import sample_multivar_gaussian, unif_dist_hyperrect

XMIN, XMAX, YMIN, YMAX = -8, 8, -8, 8


class EstimationParams(TypedDict):
    """
    Typed dictionary class to store parameters we hold fixed in each SCP run.

    Notes
    -----
    https://typing.python.org/en/latest/spec/callables.html#unpack-kwargs.
    """

    J: int
    M: int
    Q: Callable[..., np.ndarray]
    grid_mins: np.ndarray
    grid_maxes: np.ndarray
    lambda_grid_size: int
    seed: int


FIXED_SCP_PARAMS_2d: EstimationParams = {
    "J": 10000,
    "M": 20,
    "Q": lambda x, y: x**2 + 3 * y**2,
    "grid_mins": np.array([XMIN, YMIN]),
    "grid_maxes": np.array([XMAX, YMAX]),
    "lambda_grid_size": 100,
    "seed": 20251028,
}


def gaussian_tgd_2d(rng: np.random.Generator, J: int) -> np.ndarray:
    """Make simple 2D Gaussian TGD.

    Returns
    -------
    (J, 2) numpy array
    """
    mean = np.array([2, -1])
    cov = np.diag([2, 1])
    return sample_multivar_gaussian(mean, cov, rng, J)


def gaussian_tgd_huge_variance_2d(rng: np.random.Generator, J: int) -> np.ndarray:
    """Make 2D Gaussian TGD with high variance.

    Returns
    -------
    (J, 2) numpy array
    """
    mean = np.array([5, -5])
    cov = 10 * np.identity(2)
    return sample_multivar_gaussian(mean, cov, rng, J)


def uniform_prior_2d(rng: np.random.Generator, J: int) -> np.ndarray:
    """Make simple 2D Uniform prior on [-5, 5]^2.

    Returns
    -------
    (J, 2) numpy array
    """
    mins = np.array([-5, -5])
    maxes = np.array([5, 5])
    return unif_dist_hyperrect(mins, maxes, rng, J)


def huge_uniform_prior_2d(rng: np.random.Generator, J: int) -> np.ndarray:
    """Make simple 2D Uniform prior on [-20, 20]^2.

    Returns
    -------
    (J, 2) numpy array
    """
    mins = np.array([-20, -20])
    maxes = np.array([20, 20])
    return unif_dist_hyperrect(mins, maxes, rng, J)


# Our set of 8 experiments. For each fixed choice of distribution, we run an
# experiment using a synthetic TGD, then feed that into the empirical
# experiment.

# 1(A) Synthetic with regular TGD, regular prior
probs_df, prior_df, tgd_df = synthetic_scp(
    K=10000, tgd=gaussian_tgd_2d, prior=uniform_prior_2d, **FIXED_SCP_PARAMS_2d
)


# 1(B) Empirical: Regular TGD, regular prior
probs_df_emp, prior_df_emp = empirical_scp(
    q_data=tgd_df["q_x"].to_numpy(), prior=uniform_prior_2d, **FIXED_SCP_PARAMS_2d
)


# 2(A) Synthetic with regular TGD, huge prior
probs_df_reg_tgd_huge_pr, prior_df_reg_tgd_huge_pr, tgd_df_reg_tgd_huge_pr = (
    synthetic_scp(
        K=10000, tgd=gaussian_tgd_2d, prior=huge_uniform_prior_2d, **FIXED_SCP_PARAMS_2d
    )
)

# 2(B) Empirical with regular TGD, huge prior
probs_df_reg_tgd_huge_pr_emp, prior_df_reg_tgd_huge_pr_emp = empirical_scp(
    q_data=tgd_df_reg_tgd_huge_pr["q_x"].to_numpy(),
    prior=huge_uniform_prior_2d,
    **FIXED_SCP_PARAMS_2d,
)

# 3(A) Synthetic with huge TGD, regular prior
probs_df_huge_tgd_reg_pr, prior_df_huge_tgd_reg_pr, tgd_df_huge_tgd_reg_pr = (
    synthetic_scp(
        K=10000,
        tgd=gaussian_tgd_huge_variance_2d,
        prior=uniform_prior_2d,
        **FIXED_SCP_PARAMS_2d,
    )
)

# 3(B) Empirical with huge TGD, regular prior
probs_df_huge_tgd_reg_pr_emp, prior_df_huge_tgd_reg_pr_emp = empirical_scp(
    q_data=tgd_df_huge_tgd_reg_pr["q_x"].to_numpy(),
    prior=uniform_prior_2d,
    **FIXED_SCP_PARAMS_2d,
)

# 4(A) Synthetic with huge TGD, huge prior
(probs_df_huge_tgd_huge_pr, prior_df_huge_tgd_huge_pr, tgd_df_huge_tgd_huge_pr) = (
    synthetic_scp(
        K=10000,
        tgd=gaussian_tgd_huge_variance_2d,
        prior=huge_uniform_prior_2d,
        **FIXED_SCP_PARAMS_2d,
    )
)

# 4(B) Empirical with huge TGD, huge prior
probs_df_huge_tgd_huge_pr_emp, prior_df_huge_tgd_huge_pr_emp = empirical_scp(
    q_data=tgd_df_huge_tgd_huge_pr["q_x"].to_numpy(),
    prior=huge_uniform_prior_2d,
    **FIXED_SCP_PARAMS_2d,
)


def test_estimated_prob_df_structure() -> None:
    """Validate prob_df output."""
    assert isinstance(probs_df, pd.DataFrame)
    assert set(probs_df.columns) == {"xmin", "xmax", "ymin", "ymax", "prob"}


def test_estimated_prob_df_bounds() -> None:
    """Validate box positions in prob_df."""
    assert (probs_df["xmin"] < probs_df["xmax"]).all()
    assert (probs_df["ymin"] < probs_df["ymax"]).all()


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
    assert set(prior_df.columns) == {"source", "x", "y", "q_x"}
    assert (prior_df["source"] == "prior").all()


def test_tgd_df_structure() -> None:
    """Validate tgd_df output."""
    assert isinstance(tgd_df, pd.DataFrame)
    assert set(tgd_df.columns) == {"source", "x", "y", "q_x"}
    assert (tgd_df["source"] == "tgd").all()


def test_prior_df_q_matches_quadratic() -> None:
    """Check that prior_df correctly saved values of Q from x and y."""
    expected_q_vals = prior_df["x"] ** 2 + 3 * prior_df["y"] ** 2
    q_vals = prior_df["q_x"]
    assert np.allclose(expected_q_vals.to_numpy(), q_vals)


def test_tgd_df_q_matches_quadratic() -> None:
    """Check that tgd_df correctly saved values of Q from x and y."""
    Q = tgd_df["x"] ** 2 + 3 * tgd_df["y"] ** 2
    q_vals = tgd_df["q_x"]
    assert np.allclose(Q.to_numpy(), q_vals)


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
        **FIXED_SCP_PARAMS_2d,
        "prior": uniform_prior_2d,
        "grid_maxes": np.array([XMAX]),  # grid mins is [xmin, ymin]
    }
    q_data_fixed = tgd_df["q_x"].to_numpy()
    with pytest.raises(ValueError, match="Incompatible lengths"):
        synthetic_scp(K=1000, tgd=gaussian_tgd_2d, **bad_fixed_params)
    with pytest.raises(ValueError, match="Incompatible lengths"):
        empirical_scp(q_data=q_data_fixed, **bad_fixed_params)

    bad_fixed_params = {
        **bad_fixed_params,
        "grid_maxes": np.array([XMAX, YMIN]),  # grid mins is [xmin, ymin]
    }
    with pytest.raises(ValueError, match="grid_mins >= grid_maxes"):
        synthetic_scp(K=1000, tgd=gaussian_tgd_2d, **bad_fixed_params)
    with pytest.raises(ValueError, match="grid_mins >= grid_maxes"):
        empirical_scp(q_data=q_data_fixed, **bad_fixed_params)

    bad_fixed_params = {**bad_fixed_params, "grid_maxes": [], "grid_mins": []}
    msg = r"cannot be empty"
    with pytest.raises(ValueError, match=msg):
        synthetic_scp(K=10000, tgd=gaussian_tgd_2d, **bad_fixed_params)
    with pytest.raises(ValueError, match=msg):
        empirical_scp(q_data=q_data_fixed, **bad_fixed_params)

    bad_fixed_params = {
        **bad_fixed_params,
        "grid_maxes": [],
        "grid_mins": np.array([1]),
    }
    msg = r"cannot be empty"
    with pytest.raises(ValueError, match=msg):
        synthetic_scp(K=10000, tgd=gaussian_tgd_2d, **bad_fixed_params)
    with pytest.raises(ValueError, match=msg):
        empirical_scp(q_data=q_data_fixed, **bad_fixed_params)
