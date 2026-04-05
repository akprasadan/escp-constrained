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
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K=30)
    # Should get (5/10)*(20/30) + (3/10)*(10/30) + (10/10)*0/10
    # = 1/3 + 1/10 = 10/30 + 3/30 = 13/30
    assert np.isclose(est, 13 / 30)


def test_prob_A_basic_mix_higher_dim():
    """Higher dim: 3x3 binned rectangle with various weights."""
    prior_counts = np.array([[10, 6, 0], [2, 4, 0], [0, 0, 2]])
    prior_A_counts = np.array([[5, 4, 0], [1, 4, 0], [0, 0, 1]])
    obs_counts = np.array([[20, 10, 0], [0, 5, 0], [0, 2, 1]])
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K=38)

    # Should get (5/10)*(20/38) + (4/6)*(10/38) + (4/4)*(5/38) + (1/2)*(1/38)
    # = 7/12
    assert np.isclose(est, 7 / 12)

    # Change a few entries one by one
    prior_A_counts = np.array([[5, 2, 0], [1, 4, 0], [0, 0, 1]])
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K=38)
    # Should get (5/10)*(20/38) + (2/6)*(10/38) + (4/4)*(5/38) + (1/2)*(1/38)
    # = 113/228
    assert np.isclose(est, 113 / 228)

    # Spread out obs_counts (without changing K)
    obs_counts = np.array([[20, 5, 5], [1, 4, 0], [1, 1, 1]])
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K=38)
    # Should get (5/10)*(20/38) + (2/6)*(5/38) + (1/2)*(1/38) + (4/4)*(4/38) + (1/2)*(1/38)
    # = 25/57
    assert np.isclose(est, 25 / 57)


def test_prob_A_skips_zero_nc():
    """Bins with prior_counts = 0 are skipped; expected = 0.12."""
    prior_counts = np.array([0, 5])
    prior_A_counts = np.array([0, 2])
    obs_counts = np.array([7, 3])
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K=10)
    assert np.isclose(est, 0.12)


def test_prob_A_skips_zero_nc_higher_dim():
    """Higher dim: Bins with prior_counts = 0 are skipped."""
    prior_counts = np.array([[0, 0, 0], [0, 0, 10], [0, 0, 0]])
    prior_A_counts = np.array([[0, 0, 0], [0, 0, 5], [0, 0, 0]])
    obs_counts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K=45)
    # should get (5/10)*(6/45) = 1/15
    assert np.isclose(est, 1 / 15)

    # Now do same thing but swap which entry of prior_counts is non-zero
    prior_counts = np.array([[0, 0, 10], [0, 0, 0], [0, 0, 0]])
    prior_A_counts = np.array([[0, 0, 5], [0, 0, 0], [0, 0, 0]])
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K=45)
    # should get (5/10)*(3/45) = 1/30
    assert np.isclose(est, 1 / 30)

    # Again swap which entry of prior_counts is non-zero, and use diff. numbers
    prior_counts = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 37]])
    prior_A_counts = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 20]])
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K=45)
    # should get (20/37)*(9/45) = 4/37
    assert np.isclose(est, 4 / 37)


def test_prob_A_all_in_A_equals_sum_weights():
    """If prior_A_counts == prior_counts for all bins, estimate equals sum(obs_counts)/K = 1.0.
    Assumes every prior_count is non-zero."""
    prior_counts = np.array([3, 2])
    prior_A_counts = np.array([3, 2])
    obs_counts = np.array([4, 6])
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K=10)
    assert np.isclose(est, 1)


def test_prob_A_all_in_A_equals_sum_weights_some_zeros_in_A():
    """If prior_A_counts == prior_counts for all bins, estimate is 1.
    But now allow some prior_A counts to be 0. Check this removes the corresponding
    proportion of data.
    """
    prior_counts = np.array([3, 0, 5, 8])
    prior_A_counts = np.array([3, 0, 5, 8])
    obs_counts = np.array([4, 6, 2, 2])
    # Instead of getting sum (obs_counts / K) = 1, we are missing bin 2
    # So probability should be 1 - (6/14) = 4/7
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K=14)
    assert np.isclose(est, 4 / 7)


def test_prob_A_all_in_A_equals_sum_weights_higher_dim():
    """Higher_dim: If prior_A_counts == prior_counts for all bins, estimate is 1.
    But every prior_A counts must be 0 for this to occur. Otherwise we miss the proportion
    of obs_counts corresponding to this bin.
    """
    prior_counts = np.array([[1, 5, 9], [3, 1, 2], [2, 11, 9]])
    prior_A_counts = np.array([[1, 5, 9], [3, 1, 2], [2, 11, 9]])
    obs_counts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K=45)
    assert np.isclose(est, 1)


def test_prob_A_all_in_A_equals_sum_weights_higher_dim_some_zeros_in_A():
    """Higher_dim: If prior_A_counts == prior_counts for all bins, estimate is 1.
    But now allow some prior_A counts to be 0. Check this removes the corresponding
    proportion of data.
    """
    prior_counts = np.array([[1, 5, 9], [3, 1, 0], [2, 11, 9]])
    prior_A_counts = np.array([[1, 5, 9], [3, 1, 0], [2, 11, 9]])
    obs_counts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K=45)

    # Instead of getting sum (obs_counts / K), we are missing the row 2, column 3 bin.
    # So probability should be 1 - (6/45)
    assert np.isclose(est, 1 - 6 / 45)


def test_prob_A_none_in_A_zero():
    """If prior_A_counts == 0 everywhere, estimate should be 0 regardless of weights."""
    prior_counts = np.array([5, 7, 9])
    prior_A_counts = np.array([0, 0, 0])
    obs_counts = np.array([2, 3, 5])
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K=10)
    assert est == 0.0


def test_prob_A_none_in_A_zero_higher_dim():
    """Higher dim: If prior_A_counts == 0 everywhere, estimate should be 0 regardless of weights."""
    prior_counts = np.array([[1, 5, 9], [3, 1, 2], [2, 11, 9]])
    prior_A_counts = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    obs_counts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K=45)
    assert np.isclose(est, 0)

    # Same thing but randomized:
    rng = np.random.default_rng(seed=20251028)
    prior_counts = rng.integers(low=0, high=10, size=(3, 3))
    prior_A_counts = np.zeros((3, 3), dtype=np.int64)
    obs_counts = rng.integers(low=0, high=10, size=(3, 3))
    K = int(np.sum(obs_counts))
    est = prob_A(prior_counts, prior_A_counts, obs_counts, K=K)

    assert np.isclose(est, 0)


def test_prob_A_length_mismatch_raises():
    """Arrays with mismatched lengths should raise ValueError."""
    prior_counts = np.array([5, 5])
    prior_A_counts = np.array([2, 3, 1])  # Wrong length
    obs_counts = np.array([4, 6])
    with pytest.raises(ValueError, match="Incompatible array shapes"):
        prob_A(prior_counts, prior_A_counts, obs_counts, K=10)


def test_prob_A_length_mismatch_raises_higher_dim():
    """Higher dim: Arrays with mismatched lengths should raise ValueError."""
    prior_counts = np.array([[10, 6, 0], [2, 4, 0], [0, 0, 2]])
    prior_A_counts = np.array([[5, 4, 0], [1, 4, 0], [0, 0, 1], [0, 1, 2]])
    obs_counts = np.array([[20, 10, 0], [0, 5, 0], [0, 2, 1]])
    with pytest.raises(ValueError, match="Incompatible array shapes"):
        prob_A(prior_counts, prior_A_counts, obs_counts, K=38)

    # Reset prior_A counts
    prior_A_counts = np.array([[5, 4, 0], [1, 4, 0], [0, 0, 1]])
    # Mess up obs_counts length, while maintaining K = 38 for obs_counts
    # to ensure right error is called
    obs_counts = np.array([[20, 10, 0], [0, 8, 0]])
    with pytest.raises(ValueError, match="Incompatible array shapes"):
        prob_A(prior_counts, prior_A_counts, obs_counts, K=38)


def test_prob_A_prior_A_counts_gt_prior_counts_raises():
    """prior_A_counts cannot exceed prior_counts for any bin."""
    prior_counts = np.array([3, 2])
    prior_A_counts = np.array([4, 1])  # prior_A_counts[0] exceeds prior_counts[0]
    obs_counts = np.array([5, 5])
    with pytest.raises(ValueError, match="prior_A_counts > prior_counts"):
        prob_A(prior_counts, prior_A_counts, obs_counts, 10)


def test_prob_A_prior_A_counts_gt_prior_counts_raises_higher_dim():
    """Higher dim: prior_A_counts cannot exceed prior_counts for any bin."""
    prior_counts = np.array([[10, 6, 0], [2, 4, 0], [0, 0, 2]])
    # Put 1 > 0 in row 1, col 3
    prior_A_counts = np.array([[5, 4, 1], [1, 4, 0], [0, 0, 1]])
    obs_counts = np.array([[20, 10, 0], [0, 5, 0], [0, 2, 1]])
    with pytest.raises(ValueError, match="prior_A_counts > prior_counts"):
        prob_A(prior_counts, prior_A_counts, obs_counts, K=38)

    # Put 3 > 2 in row 3, col 3 (and revert 1 > 0 issue)
    prior_A_counts = np.array([[5, 4, 0], [1, 4, 0], [0, 0, 3]])
    with pytest.raises(ValueError, match="prior_A_counts > prior_counts"):
        prob_A(prior_counts, prior_A_counts, obs_counts, K=38)


def test_prob_A_invalid_K_raises():
    """K must be a positive integer."""
    prior_counts = np.array([1, 1])
    prior_A_counts = np.array([0, 1])
    obs_counts = np.array([1, 1])
    for bad_K in [0, -1, 3.14]:
        with pytest.raises(ValueError, match="K must be a positive integer"):
            prob_A(prior_counts, prior_A_counts, obs_counts, bad_K)


def test_prob_A_invalid_K_raises_higher_dim():
    """Higher dim: K must be a positive integer."""
    prior_counts = np.array([[10, 6, 0], [2, 4, 0], [0, 0, 2]])
    prior_A_counts = np.array([[5, 4, 0], [1, 4, 0], [0, 0, 1]])
    obs_counts = np.array([[20, 10, 0], [0, 5, 0], [0, 2, 1]])
    for bad_K in [0, -1, 3.14]:
        with pytest.raises(ValueError, match="K must be a positive integer"):
            prob_A(prior_counts, prior_A_counts, obs_counts, bad_K)


def test_prob_A_wrong_K_sum_raises():
    """K must be a be the sum of values in obs_counts."""
    prior_counts = np.array([1, 1, 5])
    prior_A_counts = np.array([0, 1, 4])
    obs_counts = np.array([1, 1, 0])
    for bad_K in [1, 4, 3]:
        with pytest.raises(ValueError, match=r"sum\(obs_counts\) != K"):
            prob_A(prior_counts, prior_A_counts, obs_counts, bad_K)


def test_prob_A_wrong_K_sum_raises_higher_dim():
    """Higher dim: K must be a be the sum of values in obs_counts."""
    prior_counts = np.array([[10, 6, 0], [2, 4, 0], [0, 0, 2]])
    prior_A_counts = np.array([[5, 4, 0], [1, 4, 0], [0, 0, 1]])
    obs_counts = np.array([[20, 10, 0], [0, 5, 0], [0, 2, 1]])
    for bad_K in [36, 37, 39]:
        with pytest.raises(ValueError, match=r"sum\(obs_counts\) != K"):
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


def test_prob_A_in_unit_interval_randomized_higher_dim():
    """Higher_dim: Randomized check that result is in [0, 1]."""
    rng = np.random.default_rng(0)
    prior_counts = rng.integers(0, 100, size=(3, 3))
    prior_A_counts = np.where(prior_counts > 0, rng.integers(0, 100, size=(3, 3)), 0)
    prior_A_counts = np.minimum(prior_A_counts, prior_counts)
    obs_counts = rng.integers(0, 100, size=(3, 3))
    K = int(obs_counts.sum())

    p = prob_A(prior_counts, prior_A_counts, obs_counts, K)
    assert 0.0 <= p <= 1.0
