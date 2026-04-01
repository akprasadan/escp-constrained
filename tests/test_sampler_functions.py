"""
Unit tests for sampling_functions.py

Tests:
- Output shape/dtype for different sizes
- Value ranges
- Reproducibility (if same seed) and diversity (if different seeds).
- Seed handling (int vs. np.random.Generator).
- Input validation
"""

from itertools import product

import numpy as np
import pytest

from src.sampler_functions import sample_distr as sample_distr


def uniform_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(0, 1, size=(n, 2))


def gaussian_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    """Non-centered, elliptical Gaussian distribution."""
    x = rng.normal(loc=2.0, scale=0.5, size=n)
    y = rng.normal(loc=-1.0, scale=2.0, size=n)
    return np.column_stack([x, y])


def constant_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Fully deterministic distribution of form.
    The ith entry is [i, -i], starting at i = 1."""
    i = np.arange(n) + 1  # Add 1 so [0, 0] is not included.
    return np.column_stack([i, -i])


n_list = [1, 5, 20, 50, 200]
distr_list = [uniform_sampler, gaussian_sampler, constant_sampler]


def test_sampler_shapes_and_type():
    """Each sampler/seed/sample size combo returns (n, 2) array, float dtype,."""
    for n, distr, seed in product(n_list, distr_list, [0, 1, 2]):
        pts = sample_distr(n=n, seed=seed, distr=distr)
        assert isinstance(pts, np.ndarray)
        assert pts.shape == (n, 2)
        assert np.issubdtype(pts.dtype, np.floating)


def test_uniform_sampler_range():
    """Uniform sampler should have points in the unit square"""
    for n in n_list:
        pts = sample_distr(n=n, seed=0, distr=uniform_sampler)
        assert np.all(pts >= 0.0) and np.all(pts <= 1.0)


def test_gaussian_and_deterministic_samplers_not_confined_to_01():
    """Gaussian and constant sampler produces array with values can outside [0,1]."""
    for n, distr in product(n_list, [gaussian_sampler, constant_sampler]):
        pts = sample_distr(n=n, seed=2, distr=distr)
        assert np.any((pts < 0.0) | (pts > 1.0))


def test_non_constant_reproducibility_same_seed_or_different():
    """
    For uniform/Gaussian distribution, same seed and same size give identical arrays.
    Different seeds give different ones.
    """
    for n, distr in product(n_list, [uniform_sampler, gaussian_sampler]):
        a = sample_distr(n=n, seed=123, distr=distr)
        b = sample_distr(n=n, seed=123, distr=distr)
        c = sample_distr(n=n, seed=124, distr=distr)

        assert np.array_equal(a, b)
        assert not np.array_equal(b, c)


def test_constant_distr_reproducibility_same_seed_or_different():
    """
    For constant distribution, same seed and same size give identical arrays.
    Different seeds give also give same arrays.
    """
    for n in n_list:
        a = sample_distr(n=n, seed=123, distr=constant_sampler)
        b = sample_distr(n=n, seed=123, distr=constant_sampler)
        c = sample_distr(n=n, seed=124, distr=constant_sampler)

        assert np.array_equal(a, b)
        assert np.array_equal(b, c)


def test_seed_int_vs_generator_equivalence():
    """
    Using an int seed or a Generator seeded with the same int should yield the same samples.
    Run for all sample size and distribution combinations.
    """
    for n, distr in product(n_list, distr_list):
        pts_int = sample_distr(n=n, seed=2026, distr=distr)
        gen = np.random.default_rng(2026)
        pts_gen = sample_distr(n=n, seed=gen, distr=distr)
        assert np.array_equal(pts_int, pts_gen)


def test_invalid_sizes_raise_error():
    """Invalid n (<=0 or non-integer) should raise ValueError for each distribution."""
    bad_n_values = [0, -1, -10, 1.5, "3"]
    for bad_n, distr in product(bad_n_values, distr_list):
        with pytest.raises(ValueError, match="n must be an integer >= 1"):
            sample_distr(n=bad_n, seed=0, distr=distr)
