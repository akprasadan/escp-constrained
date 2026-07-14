"""
sampler_functions.py.

Helper function to draw data from a trial-generating-distribution (TGD) or
prior, for any dimension. The helper function take in a seed, desired sample
size, and a sampler functions of the form:

>>> def sampler(rng: np.random.Generator, n: int) -> np.ndarray:
>>>     ...

These sampler functions are used as inputs into SCP estimation functions in
src/estimate_functions.py, and those estimation functions call the helper
function in this module. The sampler functions themselves do not handle the seed
logic, while these helpers do.

Also included are some custom distributions to sample from.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd


def sample_distr(
    n: int,
    seed: int,
    distr: Callable[[np.random.Generator, int], np.ndarray],
) -> np.ndarray:
    """
    Draw n points from the trial-generating distribution (TGD) or prior.

    Any custom distribution is permitted so long as it is of the form:

        distr(rng: numpy.random.Generator, n: int) -> (n, d) numpy array

    Parameters
    ----------
    n : int
        Number of samples to draw.
    seed : int
        Seed used to initialize an RNG for reproducibility.
    distr : Callable[[np.random.Generator, int], np.ndarray]
        Should take arguments (rng, n) and return (n, d) numpy array.

    Returns
    -------
    (n, d) numpy array containing the sampled points.

    Examples
    --------
    **Gaussian sampler**

    >>> def gaussian_distr(rng, n) -> np.ndarray:
    ...     return rng.normal(0, 1, size=(n, 2))
    >>> sample_distr(10000, seed=5, distr=gaussian_distr)


    Warning
    --------
    If you need to generate several independent batches of data, do not
    repeatedly call `sample_distr()` with the same seed. You must reset the
    seed. For example, do this:

    >>> master_rng = np.random.default_rng(seed)
    >>> seed_A = int(master_rng.integers(0, 2**63 - 1))
    >>> seed_B = int(master_rng.integers(0, 2**63 - 1))

    >>> dataset_A = sample_distr(50, seed=seed_A, distr=distr)
    >>> dataset_B = sample_distr(50, seed=seed_B, distr=distr)

    Do not do:

    >>> dataset_A = sample_distr(50, seed=123, distr=distr)
    >>> dataset_B = sample_distr(50, seed=123, distr=distr)

    In the latter call, dataset_A and dataset_B will be identical.
    """
    if not (isinstance(n, int) and n >= 1):
        raise ValueError("n must be an integer >= 1.")

    rng = np.random.default_rng(seed)
    return np.array(distr(rng, n), dtype=float)


def sample_from_SCP(df: pd.DataFrame, n: int, seed: int) -> tuple[np.ndarray, ...]:
    """
    Given a 2D-4D SCP solution, sample from the estimated heatmap.

    In 2D, for example, it expects dataframe output from empirical_scp(), with
    columns xmin, xmax, ymin, ymax, and prob. The 'prob' column is assured to
    sum to 1 already. We will treat the grid as a single long vector X = (X_1,
    ...., X_m) to sample from. Using the prob vector, compute the CDF F(X_i)
    over each X_i using np.cumsum() over `probs'. Then generate U_1, ..., U_n ~
    Unif(0,1) IID, and find the "X_i" in the grid such that F(X_i) is
    approximately U_i. Then X_i will be distributed as F. E.g., if the cumsum
    vector is [0, 0.2, 0.5, 0.7, 0.9, 1], corresponding to six = 2 x 3 grid
    points, and we draw 0.3, this corresponds to picking the third (xmin, xmax,
    ymin, ymax) quadruplet.

    Parameters
    ----------
    df : pd.DataFrame
        First output dataframe, probs_df, from empirical_scp().
    n : int
        Number of input samples to draw.
    seed : int
        Seed used to initialize an RNG for reproducibility.

    Returns
    -------
    A tuple of 2, 3 or 4 numpy arrays, depending on the dimension
    """
    rng = np.random.default_rng(seed)

    # Compute CDF over the flattened grid
    cdf = np.cumsum(df["prob"])
    u = rng.random(n)
    # Sample from indices using the CDF
    idx = np.searchsorted(cdf, u)

    mins = ["xmin", "ymin", "zmin", "wmin"]
    maxes = ["xmax", "ymax", "zmax", "wmax"]

    # For the "x"s, get vector of sampled df["xmin"] and df["xmax"] rows, then
    # draw from Uniform(xmin, xmax) for each row. Then do it for the y's, and
    # z's and w's, if it exists
    samples = [
        rng.uniform(df[min_col][idx], df[max_col][idx])
        for (min_col, max_col) in zip(mins, maxes, strict=True)
        if min_col in df  # If dimension exists
    ]

    return tuple(samples)


def sample_univar_gaussian(
    mean: float, sd: float, rng: np.random.Generator, J: int
) -> np.ndarray:
    """Sample from a univariate Gaussian.

    Returns
    -------
    (J, ) numpy array
    """
    return rng.normal(loc=mean, scale=sd, size=J)


def sample_multivar_gaussian(
    mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator, J: int
) -> np.ndarray:
    """Sample from a multivariate Gaussian.

    Returns
    -------
    numpy array
    """
    # Make sure 1D parameters are also accepted
    mean = np.array([mean]) if np.isscalar(mean) else mean
    cov = np.array([[cov]]) if np.isscalar(cov) else cov
    return rng.multivariate_normal(mean, cov, size=J)


def sample_gaussian_mixture(
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    var_a: np.ndarray,
    var_b: np.ndarray,
    weight: float,
    rng: np.random.Generator,
    J: int,
) -> np.ndarray:
    """
    Sample from a multivariate Gaussian mixture with two components.

    Parameters
    ----------
    mean_a, mean_b: (D, ) numpy array
        Mean vectors of each component of Gaussian mixture
    var_a, var_b : (D, D) numpy array
        Covariance matrix of each component of the Gaussian mixture
    weight : float
        The weight of the first component of the Gaussian mixture
    rng : np.random.Generator
        For reproducibility.
    J : int
        Number of samples

    Returns
    -------
    x, y : (n, ) numpy array
        First and second coordinate arrays, each of length n.
    """
    # Generate n Unif[0,1] RVs, and let those < 0.5 be True, rest False
    comp = rng.random(J) < weight

    # All the 'true' components are assigned one Gaussian distribution Rest are
    # assigned the other Gaussian distribution
    dim = np.size(mean_a)
    samples = np.empty((J, dim))
    samples[comp] = sample_multivar_gaussian(
        mean=mean_a, cov=var_a, rng=rng, J=comp.sum()
    )
    samples[~comp] = sample_multivar_gaussian(
        mean=mean_b, cov=var_b, rng=rng, J=(~comp).sum()
    )
    return samples


def unif_dist_hyperrect(
    min_vec: np.ndarray, max_vec: np.ndarray, rng: np.random.Generator, J: int
) -> np.ndarray:
    """Sample from uniform distribution on a hyper-rectangle.

    Parameters
    ----------
    min_vec, max_vec : numpy array of length d
        Rectangle bounds; require min_vec[i] < max_vec[i] for each i.
    rng : np.random.Generator
        For reproducibility.
    J : int
        Number of samples to draw

    Returns
    -------
    (J, d) numpy array
    """
    dim = len(min_vec)
    return rng.uniform(min_vec, max_vec, size=(J, dim))


def sample_multi_beta_dist(
    min_vec: np.ndarray,
    max_vec: np.ndarray,
    a_vec: np.ndarray,
    b_vec: np.ndarray,
    rng: np.random.Generator,
    J: int,
) -> np.ndarray:
    """
    Sample from a scaled/translated Beta distributions in general dimensions.

    In each dimension i out of d dimensions, given an interval [x_i, y_i] and
    parameters (a_i, b_i), define Z_i = x_i + W_i(y_i - x_i), where W_i ~
    Beta(a_i, b_i). This function returns the tuple (Z_1, ..., Z_d), which is
    the independent coupling.

    Parameters
    ----------
    min_vec, max_vec : numpy array of length d
        Rectangle bounds; require min_vec[i] < max_vec[i] for each i.
    a_vec, b_vec : numpy array of length d
        In each dimension, store the alpha (in a_vec) and beta (in b_vec) of the
        corresponding Beta distribution. In the notation above, a_vec[i] = a_i
        and b_vec[i] = b_i.
    rng : np.random.Generator
        For reproducibility.
    J : int
        Number of samples to draw

    Returns
    -------
    (J, d) numpy array
    """
    dim = len(min_vec)
    unscaled_samples = [rng.beta(a=a_vec[i], b=b_vec[i], size=J) for i in range(dim)]
    unscaled_samples = np.column_stack(unscaled_samples)

    return unscaled_samples * (max_vec - min_vec) + min_vec
