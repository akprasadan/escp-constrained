"""
sampler_functions.py

Helper function to draw data from a trial-generating-distribution (TGD) or prior. The helper function
take in a seed, desired sample size, and a sampler functions of the form:

>>> def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
>>>     ...

These sampler functions are used as inputs into eSCP estimation functions in src/estimate_functions.py,
and those estimation functions call the helper function in this module. The sampler functions themselves
do not handle the seed logic, while these helpers do.
"""

from collections.abc import Callable

import numpy as np


def sample_distr(
    n: int,
    seed: int | np.random.Generator | None,
    distr: Callable[[np.random.Generator, int], np.ndarray],
) -> np.ndarray:
    """
    Draw n points from the trial-generating distribution (TGD) or prior.
    Any custom distribution is permitted so long as it is of the form:

        distr(rng: numpy.random.Generator, n: int) -> (n, 2) numpy array

    Parameters
    ----------
    n : int
        Number of samples to draw.
    seed : int or None
        Seed used to initialize an RNG for reproducibility.
    distr : Callable[[np.random.Generator, int], np.ndarray]
        Should take arguments (rng, n) and return (n, 2) numpy array.

    Examples
    --------
    **Gaussian sampler**

    >>> def gaussian_distr(rng, n):
    >>>    return rng.normal(0, 1, size=(n, 2))
    >>> sample_distr(10000, seed=5, distr=gaussian_distr)

    Returns
    -------
    (n, 2) numpy array containing the sampled points.
    """
    if not (isinstance(n, int) and n >= 1):
        raise ValueError("n must be an integer >= 1.")

    rng = np.random.default_rng(seed)
    return np.asarray(distr(rng, n), dtype=float)
