"""
Unit tests for src.sampling_functions.py.

Tests:

- Output shape/dtype for different sizes
- Value ranges
- Reproducibility (if same seed) and diversity (if different seeds)
- Seed handling (int vs. np.random.Generator)
- Input validation
"""

from itertools import product

import numpy as np
import pytest

from src.sampler_functions import (
    sample_distr,
    sample_multivar_gaussian,
    sample_univar_gaussian,
    unif_dist_hyperrect,
)


def custom_uniform_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    """Make simple uniform sampler."""
    mins = np.array([0, 0])
    maxes = np.array([1, 1])
    return unif_dist_hyperrect(mins, maxes, rng, n)


def custom_gaussian_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    """Non-centered, elliptical Gaussian distribution."""
    mean = np.array([2, -1])
    cov = np.diag([0.5, 2])
    return sample_multivar_gaussian(mean, cov, rng, n)


def custom_multivar_gaussian_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    """Multivariate Gaussian for 3D data."""
    mean = np.array([1, 2, 1])
    cov = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])  # eigenvalues all positive

    return sample_multivar_gaussian(mean, cov, rng, n)


def constant_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Make fully deterministic 2D distribution.

    The ith entry is [i, -i], starting at i = 1.
    """
    i = np.arange(n) + 1  # Add 1 so [0, 0] is not included.
    return np.column_stack([i, -i])


n_list = [1, 5, 20, 50, 200]
distr_list = [
    custom_uniform_sampler,
    custom_gaussian_sampler,
    constant_sampler,
    custom_multivar_gaussian_sampler,
]


def test_our_sampler_matches_native_sampler_gaussian() -> None:
    """Check our implementation of random samples matches numpy's."""
    rng = np.random.default_rng(seed=25)
    mean = 2
    sd = 5
    our_sample = sample_univar_gaussian(mean, sd, rng, 50)

    rng_copy = np.random.default_rng(seed=25)
    numpy_sample = rng_copy.normal(mean, sd, 50)
    assert np.allclose(our_sample, numpy_sample)


def test_our_sampler_matches_native_sampler_bivar_gaussian() -> None:
    """Check our implementation of random samples matches numpy's."""
    rng = np.random.default_rng(seed=25)
    mean = np.array([0.5, -2])
    cov = np.diag([1, 4])
    our_sample = sample_multivar_gaussian(mean, cov, rng, 50)

    rng_copy = np.random.default_rng(seed=25)
    numpy_sample = rng_copy.multivariate_normal(mean, cov, 50)
    assert np.allclose(our_sample, numpy_sample)


def test_our_sampler_matches_native_sampler_unif() -> None:
    """Check our implementation of random samples matches numpy's."""
    rng = np.random.default_rng(seed=25)
    mins = np.array([0, 0])
    maxes = np.array([1, 1])
    our_sample = unif_dist_hyperrect(mins, maxes, rng, 50)

    rng_copy = np.random.default_rng(seed=25)
    numpy_sample = rng_copy.uniform(mins, maxes, (50, 2))
    assert np.allclose(our_sample, numpy_sample)


def test_sampler_shapes_and_type() -> None:
    """Each sampler/seed/size combo returns (n, d) array, float dtype."""
    for n, distr, seed in product(n_list, distr_list, [0, 1, 2, 3]):
        pts = sample_distr(n=n, seed=seed, distr=distr)

        assert isinstance(pts, np.ndarray)
        assert np.issubdtype(pts.dtype, np.floating)

        assert (
            pts.shape == (n, 3)
            if distr == custom_multivar_gaussian_sampler
            else pts.shape == (n, 2)
        )


def test_uniform_sampler_range() -> None:
    """Uniform sampler should have points in the unit square."""
    for n in n_list:
        pts = sample_distr(n=n, seed=0, distr=custom_uniform_sampler)
        assert np.all(pts >= 0) and np.all(pts <= 1)


def test_gaussian_and_deterministic_samplers_not_confined_to_01() -> None:
    """Gaussian and constant sampler produces noisy arrays.

    At least some coordinates should exit the unit interval.
    """
    spread_out_samplers = [
        custom_gaussian_sampler,
        constant_sampler,
        custom_multivar_gaussian_sampler,
    ]
    for n, distr in product(n_list, spread_out_samplers):
        pts = sample_distr(n=n, seed=2, distr=distr)
        assert np.any((pts < 0) | (pts > 1))


def test_non_constant_reproducibility_same_seed_or_different() -> None:
    """Test behavior of seeds.

    For uniform/Gaussian distribution, same seed and same size give identical
    arrays. Different seeds give different ones.
    """
    for n, distr in product(n_list, [custom_uniform_sampler, custom_gaussian_sampler]):
        a = sample_distr(n=n, seed=123, distr=distr)
        b = sample_distr(n=n, seed=123, distr=distr)
        c = sample_distr(n=n, seed=124, distr=distr)

        assert np.array_equal(a, b)
        assert not np.array_equal(b, c)


def test_constant_distr_reproducibility_same_seed_or_different() -> None:
    """Test seed behavior.

    For constant distribution, same seed and same size give identical arrays.
    Different seeds give also give same arrays.
    """
    for n in n_list:
        a = sample_distr(n=n, seed=123, distr=constant_sampler)
        b = sample_distr(n=n, seed=123, distr=constant_sampler)
        c = sample_distr(n=n, seed=124, distr=constant_sampler)

        assert np.array_equal(a, b)
        assert np.array_equal(b, c)


def test_invalid_sizes_raise_error() -> None:
    """Test invalid integer input.

    Invalid n (<=0 or non-integer) should raise ValueError for each
    distribution.
    """
    bad_n_values = [0, -1, -10, 1.5, "3"]
    for bad_n, distr in product(bad_n_values, distr_list):
        with pytest.raises(ValueError, match="n must be an integer >= 1"):
            sample_distr(n=bad_n, seed=0, distr=distr)
