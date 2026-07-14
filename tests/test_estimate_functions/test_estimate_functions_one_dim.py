"""
Unit tests for `src.estimate_functions.py` (1D).

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
from src.sampler_functions import sample_univar_gaussian, unif_dist_hyperrect
from tests.test_estimate_functions.test_estimate_functions import EstimationParams

XMIN, XMAX = -8, 8


FIXED_SCP_PARAMS_1D: EstimationParams = {
    "J": 10000,
    "M": 20,
    "Q": lambda x: x**2 + 3,
    "grid_mins": np.array([XMIN]),
    "grid_maxes": np.array([XMAX]),
    "lambda_grid_size": 100,
    "seed": 20251028,
}


def gaussian_tgd_1d(rng: np.random.Generator, J: int) -> np.ndarray:
    """Make 1D Gaussian TGD.

    Returns
    -------
    (J, 1) numpy array
    """
    return sample_univar_gaussian(2, 2, rng, J)[:, None]


def gaussian_tgd_huge_variance_1d(rng: np.random.Generator, J: int) -> np.ndarray:
    """Make 1D Gaussian TGD with high variance.

    Returns
    -------
    (J, 1) numpy array
    """
    return sample_univar_gaussian(5, 10, rng, J)[:, None]


def uniform_prior_1d(rng: np.random.Generator, J: int) -> np.ndarray:
    """Make simple uniform prior on [-5, 5].

    Returns
    -------
    (J, 1) numpy array
    """
    return unif_dist_hyperrect(np.array([-5]), np.array([5]), rng, J)


def huge_uniform_prior_1d(rng: np.random.Generator, J: int) -> np.ndarray:
    """Make simple uniform prior on [-20, 20].

    Returns
    -------
    (J, 1) numpy array
    """
    return unif_dist_hyperrect(np.array([-20]), np.array([20]), rng, J)


# Our set of 8 experiments. For each fixed choice of distribution, we run an
# experiment using a synthetic TGD, then feed that into the empirical
# experiment.

# 1(A) Synthetic with regular TGD, regular prior
probs_df, prior_df, tgd_df = synthetic_scp(
    K=10000, tgd=gaussian_tgd_1d, prior=uniform_prior_1d, **FIXED_SCP_PARAMS_1D
)


# 1(B) Empirical: Regular TGD, regular prior
probs_df_emp, prior_df_emp = empirical_scp(
    q_data=tgd_df["q_x"].to_numpy(), prior=uniform_prior_1d, **FIXED_SCP_PARAMS_1D
)


# 2(A) Synthetic with regular TGD, huge prior
probs_df_reg_tgd_huge_pr, prior_df_reg_tgd_huge_pr, tgd_df_reg_tgd_huge_pr = (
    synthetic_scp(
        K=10000, tgd=gaussian_tgd_1d, prior=huge_uniform_prior_1d, **FIXED_SCP_PARAMS_1D
    )
)

# 2(B) Empirical with regular TGD, huge prior
probs_df_reg_tgd_huge_pr_emp, prior_df_reg_tgd_huge_pr_emp = empirical_scp(
    q_data=tgd_df_reg_tgd_huge_pr["q_x"].to_numpy(),
    prior=huge_uniform_prior_1d,
    **FIXED_SCP_PARAMS_1D,
)

# 3(A) Synthetic with huge TGD, regular prior
probs_df_huge_tgd_reg_pr, prior_df_huge_tgd_reg_pr, tgd_df_huge_tgd_reg_pr = (
    synthetic_scp(
        K=10000,
        tgd=gaussian_tgd_huge_variance_1d,
        prior=uniform_prior_1d,
        **FIXED_SCP_PARAMS_1D,
    )
)

# 3(B) Empirical with huge TGD, regular prior
probs_df_huge_tgd_reg_pr_emp, prior_df_huge_tgd_reg_pr_emp = empirical_scp(
    q_data=tgd_df_huge_tgd_reg_pr["q_x"].to_numpy(),
    prior=uniform_prior_1d,
    **FIXED_SCP_PARAMS_1D,
)

# 4(A) Synthetic with huge TGD, huge prior
(probs_df_huge_tgd_huge_pr, prior_df_huge_tgd_huge_pr, tgd_df_huge_tgd_huge_pr) = (
    synthetic_scp(
        K=10000,
        tgd=gaussian_tgd_huge_variance_1d,
        prior=huge_uniform_prior_1d,
        **FIXED_SCP_PARAMS_1D,
    )
)

# 4(B) Empirical with huge TGD, huge prior
probs_df_huge_tgd_huge_pr_emp, prior_df_huge_tgd_huge_pr_emp = empirical_scp(
    q_data=tgd_df_huge_tgd_huge_pr["q_x"].to_numpy(),
    prior=huge_uniform_prior_1d,
    **FIXED_SCP_PARAMS_1D,
)


def test_estimated_prob_df_structure() -> None:
    """Validate prob_df output."""
    assert isinstance(probs_df, pd.DataFrame)
    assert set(probs_df.columns) == {"xmin", "xmax", "prob"}


def test_estimated_prob_df_bounds() -> None:
    """Validate box positions in prob_df."""
    assert (probs_df["xmin"] < probs_df["xmax"]).all()


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
    assert set(prior_df.columns) == {"source", "x", "q_x"}
    assert (prior_df["source"] == "prior").all()


def test_tgd_df_structure() -> None:
    """Validate tgd_df output."""
    assert isinstance(tgd_df, pd.DataFrame)
    assert set(tgd_df.columns) == {"source", "x", "q_x"}
    assert (tgd_df["source"] == "tgd").all()


def test_prior_df_q_matches_quadratic() -> None:
    """Check that prior_df correctly saved values of Q from x and y."""
    expected_q_vals = prior_df["x"] ** 2 + 3
    q_vals = prior_df["q_x"]
    assert np.allclose(expected_q_vals.to_numpy(), q_vals)


def test_tgd_df_q_matches_quadratic() -> None:
    """Check that tgd_df correctly saved values of Q from x and y."""
    Q = tgd_df["x"] ** 2 + 3
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
    q_data_fixed = tgd_df["q_x"].to_numpy()

    bad_fixed_params = {
        **FIXED_SCP_PARAMS_1D,
        "prior": uniform_prior_1d,
        "grid_maxes": np.array([XMIN]),  # grid mins is [xmin, ymin]
    }
    with pytest.raises(ValueError, match="grid_mins >= grid_maxes"):
        synthetic_scp(K=1000, tgd=gaussian_tgd_1d, **bad_fixed_params)
    with pytest.raises(ValueError, match="grid_mins >= grid_maxes"):
        empirical_scp(q_data=q_data_fixed, **bad_fixed_params)

    bad_fixed_params = {**bad_fixed_params, "grid_maxes": [], "grid_mins": []}
    msg = r"cannot be empty"
    with pytest.raises(ValueError, match=msg):
        synthetic_scp(K=10000, tgd=gaussian_tgd_1d, **bad_fixed_params)
    with pytest.raises(ValueError, match=msg):
        empirical_scp(q_data=q_data_fixed, **bad_fixed_params)

    bad_fixed_params = {
        **bad_fixed_params,
        "grid_maxes": [],
        "grid_mins": np.array([1]),
    }
    msg = r"cannot be empty"
    with pytest.raises(ValueError, match=msg):
        synthetic_scp(K=10000, tgd=gaussian_tgd_1d, **bad_fixed_params)
    with pytest.raises(ValueError, match=msg):
        empirical_scp(q_data=q_data_fixed, **bad_fixed_params)
