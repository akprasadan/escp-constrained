"""
Unit tests for `src.estimate_functions`.

Check that output dataframes have right properties, and that the synthetic
(generate Q by drawing from the TGD) and empirical (pass in Q directly) estimations match.
Doesn't directly test correctness of eSCP algorithm (other test modules do).
"""

from collections.abc import Callable
from typing import TypedDict

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.estimate_functions import synthetic_scp as synthetic_scp
from src.estimate_functions import empirical_scp as empirical_scp

XMIN, XMAX, YMIN, YMAX = -8, 8, -8, 8


# Typed dictionary class to store parameters we hold fixed in each SCP run
# https://typing.python.org/en/latest/spec/callables.html#unpack-kwargs
class EstimationParams(TypedDict):
    J: int
    M: int
    Q: Callable[[np.ndarray, np.ndarray], np.ndarray]
    lambda_bounds: tuple[float, float, float, float]
    lambda_grid_size: int
    seed: int | None


FIXED_SCP_PARAMS: EstimationParams = {
    "J": 10000,
    "M": 20,
    "Q": lambda x, y: x**2 + 3 * y**2,
    "lambda_bounds": (XMIN, XMAX, YMIN, YMAX),
    "lambda_grid_size": 100,
    "seed": 20251028,
}


def gaussian_tgd(rng, J):
    x = rng.normal(2.0, 2, size=J)
    y = rng.normal(-1.0, 1, size=J)
    return np.column_stack((x, y))


def gaussian_tgd_huge_variance(rng, J):
    x = rng.normal(5.0, 10, size=J)
    y = rng.normal(-5.0, 10, size=J)
    return np.column_stack((x, y))


def uniform_prior(rng, J):
    x = rng.uniform(-5.0, 5.0, size=J)
    y = rng.uniform(-5.0, 5.0, size=J)
    return np.column_stack((x, y))


def huge_uniform_prior(rng, J):
    x = rng.uniform(-20.0, 20.0, size=J)
    y = rng.uniform(-20.0, 20.0, size=J)
    return np.column_stack((x, y))


# Our set of 8 experiments. For each fixed choice of distribution,
# we run an experiment using a synthetic TGD, then feed that into the empirical experiment.

# 1(A) Synthetic with regular TGD, regular prior
probs_df, prior_df, tgd_df = synthetic_scp(
    K=10000, tgd=gaussian_tgd, prior=uniform_prior, **FIXED_SCP_PARAMS
)

# 1(B) Empirical: Regular TGD, regular prior
probs_df_emp, prior_df_emp = empirical_scp(
    q_data=tgd_df["q"].to_numpy(), prior=uniform_prior, **FIXED_SCP_PARAMS
)

# 2(A) Synthetic with regular TGD, huge prior
probs_df_reg_tgd_huge_pr, prior_df_reg_tgd_huge_pr, tgd_df_reg_tgd_huge_pr = synthetic_scp(
    K=10000, tgd=gaussian_tgd, prior=huge_uniform_prior, **FIXED_SCP_PARAMS
)

# 2(B) Empirical with regular TGD, huge prior
probs_df_reg_tgd_huge_pr_emp, prior_df_reg_tgd_huge_pr_emp = empirical_scp(
    q_data=tgd_df_reg_tgd_huge_pr["q"].to_numpy(), prior=huge_uniform_prior, **FIXED_SCP_PARAMS
)

# 3(A) Synthetic with huge TGD, regular prior
probs_df_huge_tgd_reg_pr, prior_df_huge_tgd_reg_pr, tgd_df_huge_tgd_reg_pr = synthetic_scp(
    K=10000, tgd=gaussian_tgd_huge_variance, prior=uniform_prior, **FIXED_SCP_PARAMS
)

# 3(B) Empirical with huge TGD, regular prior
probs_df_huge_tgd_reg_pr_emp, prior_df_huge_tgd_reg_pr_emp = empirical_scp(
    q_data=tgd_df_huge_tgd_reg_pr["q"].to_numpy(), prior=uniform_prior, **FIXED_SCP_PARAMS
)

# 4(A) Synthetic with huge TGD, huge prior
probs_df_huge_tgd_huge_pr, prior_df_huge_tgd_huge_pr, tgd_df_huge_tgd_huge_pr = synthetic_scp(
    K=10000, tgd=gaussian_tgd_huge_variance, prior=huge_uniform_prior, **FIXED_SCP_PARAMS
)

# 4(B) Empirical with huge TGD, huge prior
probs_df_huge_tgd_huge_pr_emp, prior_df_huge_tgd_huge_pr_emp = empirical_scp(
    q_data=tgd_df_huge_tgd_huge_pr["q"].to_numpy(), prior=huge_uniform_prior, **FIXED_SCP_PARAMS
)


def test_estimated_prob_df_structure():
    "Validate prob_df output."
    assert isinstance(probs_df, pd.DataFrame)
    assert set(probs_df.columns) == {"xmin", "xmax", "ymin", "ymax", "prob"}


def test_estimated_prob_df_bounds():
    "Validate box positions in prob_df."
    assert (probs_df["xmin"] < probs_df["xmax"]).all()
    assert (probs_df["ymin"] < probs_df["ymax"]).all()


def test_estimated_probs_df_prob_sums_to_one():
    """Validate probabilities from prob_df, prob_df_huge_prior, prob_df_huge_tgd_and_prior,
    and the accompanying empirical versions of these."""

    # Experiment 1(A) and 1(B)
    assert np.isclose(probs_df["prob"].sum(), 1.0)
    assert np.isclose(probs_df_emp["prob"].sum(), 1.0)

    # 2(A) and 2(B)
    assert np.isclose(probs_df_reg_tgd_huge_pr["prob"].sum(), 1.0)
    assert np.isclose(probs_df_reg_tgd_huge_pr_emp["prob"].sum(), 1.0)

    # 3(A) and 3(B)
    assert np.isclose(probs_df_huge_tgd_reg_pr["prob"].sum(), 1.0)
    assert np.isclose(probs_df_huge_tgd_reg_pr_emp["prob"].sum(), 1.0)

    # 4(A) and 4(B)
    assert np.isclose(probs_df_huge_tgd_huge_pr["prob"].sum(), 1.0)
    assert np.isclose(probs_df_huge_tgd_huge_pr_emp["prob"].sum(), 1.0)


def test_prior_df_structure():
    "Validate prior_df output."
    assert isinstance(prior_df, pd.DataFrame)
    assert set(prior_df.columns) == {"source", "x", "y", "q"}
    assert (prior_df["source"] == "prior").all()


def test_tgd_df_structure():
    "Validate tgd_df output."
    assert isinstance(tgd_df, pd.DataFrame)
    assert set(tgd_df.columns) == {"source", "x", "y", "q"}
    assert (tgd_df["source"] == "tgd").all()


def test_prior_df_q_matches_quadratic():
    "Check that prior_df correctly saved values of Q from x and y"
    Q = prior_df["x"] ** 2 + 3 * prior_df["y"] ** 2
    assert np.allclose(Q.to_numpy(), prior_df["q"].to_numpy())


def test_tgd_df_q_matches_quadratic():
    "Check that tgd_df correctly saved values of Q from x and y"
    Q = tgd_df["x"] ** 2 + 3 * tgd_df["y"] ** 2
    assert np.allclose(Q.to_numpy(), tgd_df["q"].to_numpy())


def test_synthetic_and_empirical_match():
    """synthetic_scp() and empirical_scp() should give the same output"""
    # Compare 1(A) and 1(B)
    assert_frame_equal(probs_df, probs_df_emp, check_exact=True)
    assert_frame_equal(prior_df, prior_df_emp, check_exact=True)

    # Compare 2(A) and 2(B)
    assert_frame_equal(probs_df_reg_tgd_huge_pr, probs_df_reg_tgd_huge_pr_emp, check_exact=True)
    assert_frame_equal(prior_df_reg_tgd_huge_pr, prior_df_reg_tgd_huge_pr_emp, check_exact=True)

    # Compare 3(A) and 3(B)
    assert_frame_equal(probs_df_huge_tgd_reg_pr, probs_df_huge_tgd_reg_pr_emp, check_exact=True)
    assert_frame_equal(prior_df_huge_tgd_reg_pr, prior_df_huge_tgd_reg_pr_emp, check_exact=True)

    # Compare 4(A) and 4(B)
    assert_frame_equal(probs_df_huge_tgd_huge_pr, probs_df_huge_tgd_huge_pr_emp, check_exact=True)
    assert_frame_equal(prior_df_huge_tgd_huge_pr, prior_df_huge_tgd_huge_pr_emp, check_exact=True)
