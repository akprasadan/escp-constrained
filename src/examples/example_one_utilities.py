"""
Helper dataclasses and functions for use in src.examples.example_one.py,
which performs a synthetic constrained SCP experiment.

"""

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import pandas as pd


@dataclass
class Domain:
    """
    Characteristics of the Lambda x U space and the distributions over it.

    Attributes
    -----------
    lambda_mean : list[float]
        The mean vector of a Gaussian distribution over Lambda
    lambda_cov : list[list[float]]
        The covariance matrix of a Gaussian distribution over Lambda
    lambda_mins, lambda_maxes : np.ndarray
        A vector of minimums or maximums defining the boundaries for Lambda
    U_mean, U_sd : float
        The mean and standard deviation of a univariate Normal distribution over U.
    U_mins, U_maxes : np.ndarray
        A vector of minimums or maximums defining the boundaries for U
    lambda_u_mins, lambda_u_maxes : np.ndarray
        A vector of minimums or maximums defining the boundaries for Lambda x U (property).
    """

    lambda_mean: list[float]
    lambda_cov: list[list[float]]
    lambda_mins: np.ndarray
    lambda_maxes: np.ndarray
    U_mean: float
    U_sd: float
    U_mins: np.ndarray
    U_maxes: np.ndarray

    @property
    def lambda_u_mins(self):
        """Get a vector of minimums defining the boundaries for Lambda x U"""
        return np.concatenate([self.lambda_mins, self.U_mins])

    @property
    def lambda_u_maxes(self):
        """Get a vector of maximums defining the boundaries for Lambda x U"""
        return np.concatenate([self.lambda_maxes, self.U_maxes])


@dataclass
class Experiment:
    """
    Experimental specifications not already specified by Domain dataclass.

    Attributes
    ----------
    Q : Callable[..., np.ndarray]
        Map from Lambda x U --> D. Not the augmented version.
    num_inputs : int
        The number of lambda_i, u_i pairs to draw in order to generate q_i observations
    num_indep_u_inputs : int
        Number of additional independent copies of u_j used in Sinkhorn estimation
    sinkhorn_grid_res_U, sinkhorn_grid_res_D : int
        The number of bins to use splitting up D and U in Sinkhorn estimation
    sinkhorn_reg : float
        Regularization parameter used in Sinkhorn estimation
    num_monte_carlo_samples : int
        Number of lambda_i to draw when estimating cost function, which integrates over Lambda.
    num_joint_pairs : int
        Number of (q_i, u_i) pairs to draw from Sinkhorn solution
    SCP_prior_count : int
        Number of prior points to draw from Lambda x U in SCP algorithm, for
        reweighting via importance sampling.
    SCP_lambda_grid_res : int
        The number of bins to split each axis of Lambda x U into in the SCP algorithm.
    SCP_D_grid_res : int
        The number of bins to split each axis of D x U into in the SCP algorithm.
    """

    Q: Callable[..., np.ndarray]
    num_inputs: int
    num_indep_u_inputs: int
    sinkhorn_grid_res_U: int
    sinkhorn_grid_res_D: int
    sinkhorn_reg: float
    num_monte_carlo_samples: int
    num_joint_pairs: int
    SCP_prior_count: int
    SCP_lambda_grid_res: int
    SCP_D_grid_res: int

    def augmented_Q(self, x, y, z):
        return np.array([self.Q(x, y, z), z])


@dataclass
class PlotOptions:
    """
    Plot specifications for example one.

    Attributes
    ----------
    figsize : tuple[int, int]
        Size of Matplotlib figure
    dpi : int
        DPI
    num_points_per_scatter : int
        Number of points to use from the TGD or Sinkhorn solution for a scatterplot.
    num_points_histogram : int
        Number of points to use from D or U to plot their marginal distributions.
    expt_name : str
        Name used to distinguish experiments in filenames for saved objects.
        Involves Sinkhorn regularizing parameter, and mean/SD parameters for distribution on U.
    u_name : str
        Label used to indicate distribution of U in a subplot title
    map_name : str
        Label used to indicate map Q in a subplot title
    D_max : float
        Upper axis limit for D used in two subplots. Lower limit is always 0.
    """

    figsize: tuple[int, int]
    dpi: int
    num_points_per_scatter: int
    num_points_histogram: int
    expt_name: str
    u_name: str
    map_name: str
    D_max: float


def uniform_lambda_sampler(
    lambda_mins: np.ndarray,
    lambda_maxes: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    """Prior on Lambda."""
    return rng.uniform(lambda_mins, lambda_maxes, size=(n, 2))


def uniform_normal_sampler(
    lambda_mins: np.ndarray,
    lambda_maxes: np.ndarray,
    u_mean: float,
    u_sd: float,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    """Prior on Lambda x U an independent coupling that preserves actual U marginal."""
    lambdas = uniform_lambda_sampler(lambda_mins, lambda_maxes, rng, n)
    us = rng.normal(u_mean, u_sd, n)[:, None]
    return np.hstack((lambdas, us))


def marginalize_out_U(df: pd.DataFrame) -> pd.DataFrame:
    """Marginalize out U in probs_df and leave the first two dimensions (Lambda).

    Parameters
    ----------
    df : pd.Dataframe
        Expected have columns {x,y,z}min and {x,y,z}max and prob.
        Axes x and y correspond to Lambda while z corresponds to U.

    Returns
    -------
    marginalized_df : pd.DataFrame
        A dataframe with columns {x,y}min, {x,y}max, and prob,
        where we summed the probability values across the different
        zmin, zmax combinations for each fixed quadruple (xmin, xmax, ymin, ymax).
    """
    marginalized_df = df.groupby(
        ["xmin", "xmax", "ymin", "ymax"], as_index=False
    ).agg(prob=("prob", "sum"))

    return marginalized_df
