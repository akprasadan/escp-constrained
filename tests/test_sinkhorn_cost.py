"""
Tests sinkhorn_functions.py

"""

import numpy as np
from src.sinkhorn_functions import compute_cost
from src.sampler_functions import sample_distr


def uniform_prior(rng, J):
    x = rng.uniform(-5, 5, size=J)
    y = rng.uniform(-5, 5, size=J)
    return np.column_stack((x, y))


# We'll use the same prior data everywhere to save on computation
PRIOR_DATA = sample_distr(n=10_000_000, seed=20251028, distr=uniform_prior)


def test_sinkhorn_cost_is_float():
    fixed_params = {
        "q": 1.5,
        "u": 1,
        "prior": None,
        "prior_data": PRIOR_DATA,
        "lambda_mins": np.array([0, 4]),
        "lambda_maxes": np.array([1, 5]),
        "MC_sample_size": None,
        "seed": 1,
    }
    cost = compute_cost(Q=lambda x, y, z: x + y - z, **fixed_params)
    assert isinstance(cost, float)


def test_sinkhorn_cost_at_0_0_and_uniform_prior():
    """
    We perfom several sanity checks by setting u = 0, q = 0, using
    prior = Uniform[0, 1]^2, and picking simple Q : R^3 --> R.
    Then the cost function computes:

        c(0, 0) = int_{[0,1]^2} |Q(lambda, 0)|^2 dlambda.

    Example 1: Take Q(lambda, u) = lambda_1 + 2u.
    Integral is int_0^1 int_0^1 x^2 dx dy = 1/3.

    Example 2: Q(lambda, u) = lambda_1 + lambda_2 - 3u.
    Integral is int_0^1 int_0^1 (x+y)^2 dx dy = 7/6.

    Example 3: Q(lambda, u) = u^2. Integral is 0.

    Example 4: Q(lambda, u) = u*lambda_1 + lambda_2^2.
    Integral is int_0^1 int_0^1 y^4 dx dy = 1/5.
    """
    fixed_params = {
        "q": 0,
        "u": 0,
        "prior": None,
        "prior_data": PRIOR_DATA,
        "lambda_mins": np.array([0, 0]),
        "lambda_maxes": np.array([1, 1]),
        "MC_sample_size": None,
        "seed": 0,
    }
    cost = compute_cost(Q=lambda x, y, z: x + 0 * y + 2 * z, **fixed_params)
    assert np.allclose(cost, 1 / 3, rtol=0.01)

    cost = compute_cost(Q=lambda x, y, z: x + y - 3 * z, **fixed_params)
    assert np.allclose(cost, 7 / 6, rtol=0.01)

    cost = compute_cost(Q=lambda x, y, z: z**2, **fixed_params)
    assert np.allclose(cost, 0)  # Without any numerical uncertainty

    cost = compute_cost(Q=lambda x, y, z: z * x + y**2, **fixed_params)
    assert np.allclose(cost, 1 / 5, rtol=0.01)


def test_sinkhorn_cost_not_at_origin_and_uniform_prior():
    """
    Repeat previous test but use u, q not at 0.
    Then the cost function computes:

        c(q, u) = int_{[0,1]^2} |q - Q(lambda, u)|^2 dlambda.

    Example 1: Take q = 1, u = 2. Set Q(lambda, u) = lambda_1 + 2*lambda_2 - 3u.
    Integral is int_0^1 int_0^1 (1 - (x + 2y - 6))^2 dx dy = 92/3.

    Example 2: Take q = 5, u = 1. Set Q(lambda, u) = lambda_1(1- 3u) + 2*lambda_2.
    Integral is int_0^1 int_0^1 (5 - (-2x + 2y))^2 dx dy = 77/3.
    """
    fixed_params = {
        "prior": None,
        "prior_data": PRIOR_DATA,
        "lambda_mins": np.array([0, 0]),
        "lambda_maxes": np.array([1, 1]),
        "MC_sample_size": None,
        "seed": 0,
    }
    cost = compute_cost(
        q=1, u=2, Q=lambda x, y, z: x + 2 * y - 3 * z, **fixed_params
    )
    assert np.allclose(cost, 92 / 3, rtol=0.005)

    cost = compute_cost(
        q=5, u=1, Q=lambda x, y, z: x * (1 - 3 * z) + 2 * y, **fixed_params
    )
    assert np.allclose(cost, 77 / 3, rtol=0.005)
