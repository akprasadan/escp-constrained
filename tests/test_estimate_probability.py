"""
Unit tests for `probability_functions.prob_A()`.

Tests:
- Correct computation for mixed bins
- Skipping bins with zero prior counts
- Edge cases: all in A, none in A
- Input validation for shape, ndim, and value constraints
- Randomized check: result always in [0, 1]
"""

import numpy as np
import pytest

from src.probability_functions import prob_A as prob_A


def test_prob_A_basic_mix():
    """Mixed bins with different conditionals and weights; expected approx 0.4333."""
    prior_counts = np.array([10, 10, 10])
    prior_A_counts = np.array([5, 3, 10])
    obs_counts = np.array([20, 10, 0])
    K = 30
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K)
    assert np.isclose(est, 13 / 30)


def test_prob_A_skips_zero_nc():
    """Bins with prior_counts = 0 are skipped; expected = 0.12."""
    prior_counts = np.array([0, 5])
    prior_A_counts = np.array([0, 2])
    obs_counts = np.array([7, 3])
    K = 10
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K)
    assert np.isclose(est, 0.12)


def test_prob_A_all_in_A_equals_sum_weights():
    """If prior_A_counts == prior_counts for all bins, estimate equals sum(obs_counts)/K = 1.0."""
    prior_counts = np.array([3, 2])
    prior_A_counts = np.array([3, 2])
    obs_counts = np.array([4, 6])
    K = 10
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K)
    assert np.isclose(est, 1.0)


def test_prob_A_none_in_A_zero():
    """If prior_A_counts == 0 everywhere, estimate should be 0 regardless of weights."""
    prior_counts = np.array([5, 7, 9])
    prior_A_counts = np.array([0, 0, 0])
    obs_counts = np.array([2, 3, 5])
    K = 10
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K)
    assert est == 0.0


def test_prob_A_length_mismatch_raises():
    """Arrays with mismatched lengths should raise ValueError."""
    prior_counts = np.array([5, 5])
    prior_A_counts = np.array([2, 3, 1])  # Wrong length
    obs_counts = np.array([4, 6])
    with pytest.raises(ValueError, match="Incompatible array lengths"):
        prob_A(prior_counts, prior_A_counts, obs_counts, 10)


def test_prob_A_ndim_checks():
    """Non-1D arrays should raise ValueError. Make all of prior_counts, prior_A_counts, and obs_counts a 2D array
    so that the incompatible array lengths error is not called.
    """
    prior_counts = np.array([[5, 5]])  # 2D
    prior_A_counts = np.array([[2, 3]])
    obs_counts = np.array([[4, 1]])
    with pytest.raises(ValueError, match="is not a 1D array"):
        prob_A(prior_counts, prior_A_counts, obs_counts, 5)


def test_prob_A_prior_A_counts_gt_prior_counts_raises():
    """prior_A_counts cannot exceed prior_counts for any bin."""
    prior_counts = np.array([3, 2])
    prior_A_counts = np.array([4, 1])  # prior_A_counts[0] exceeds prior_counts[0]
    obs_counts = np.array([5, 5])
    with pytest.raises(ValueError, match="prior_A_counts > prior_counts"):
        prob_A(prior_counts, prior_A_counts, obs_counts, 10)


def test_prob_A_invalid_K_raises():
    """K must be a positive integer."""
    prior_counts = np.array([1, 1])
    prior_A_counts = np.array([0, 1])
    obs_counts = np.array([1, 1])
    for bad_K in [0, -1, 3.14]:
        with pytest.raises(ValueError, match="K must be a positive integer"):
            prob_A(prior_counts, prior_A_counts, obs_counts, bad_K)


def test_prob_A_in_unit_interval_randomized():
    """Randomized check that result is in [0, 1]."""
    rng = np.random.default_rng(0)
    C = 50
    prior_counts = rng.integers(0, 100, size=C)
    prior_A_counts = np.where(prior_counts > 0, rng.integers(0, 100, size=C), 0)
    prior_A_counts = np.minimum(prior_A_counts, prior_counts)
    obs_counts = rng.integers(0, 100, size=C)
    K = int(obs_counts.sum())

    p = prob_A(prior_counts, prior_A_counts, obs_counts, K)
    assert 0.0 <= p <= 1.0
