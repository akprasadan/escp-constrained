"""
example_two.py

Generate plots for a synthetic example with a Gaussian mixture TGD and a Gaussian prior,
truncated to a rectangular box. We take a sequence of TGDs with decaying variance,
and use a quadratic map Q applied to the TGD.

Output:
  - plots/example_two.pdf
"""

from pathlib import Path
from collections.abc import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.axes import Axes

from src.example_one import sample_gaussian_mixture as sample_gaussian_mixture
from src.estimate_functions import synthetic_scp as synthetic_scp

# Use higher quality + LaTeX fonts in Matplotlib
plt.rcParams.update({"pdf.fonttype": 42, "text.usetex": True})


# TGD configs
GRID_BOUNDS = (-5, 5, -5, 5)
MU_A, MU_B = np.array([-1, -1]), np.array([2, 2])
TGD_VARS = [1.0, 0.1, 0.01]

# Prior configs
PRIOR_COMPONENT = "B"  # 'A' or 'B' (prior mean is nudged toward the origin)
PRIOR_OFFSET = np.array([0.75, 0.75])  # how much to nudge toward the origin
PRIOR_MEAN = (MU_A + PRIOR_OFFSET) if PRIOR_COMPONENT == "A" else (MU_B - PRIOR_OFFSET)
PRIOR_VAR = 2.5

# eSCP configs
J, K, M, LAMBDA_GRID_SIZE = 100_000, 100_000, 100, 100
SEED = 20251028

# Plot options
DPI, FIGSIZE = 1200, (7.5, 7.5)
PRIOR_COLOR, TGD_COLOR = "#f4d03f", "#ff7f0e"
PURPLE_BG = (68 / 255, 1 / 255, 84 / 255)
SCATTERPLOT_OPTIONS = dict(s=7, alpha=0.95, edgecolors="black", linewidths=0.3, rasterized=False)


def Q(x, y):
    return x**2 + 3 * y**2


def gaussian_sampler(
    rng: np.random.Generator, J: int, mean: tuple[float, float] | np.ndarray, variance: float
) -> np.ndarray:
    """Example two's prior distribution sampler.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    J : int
        Number of samples to draw.
    mean : tuple[float, float] | np.ndarray
        Mean vector (mu_x, mu_y).
    variance : float
        Scalar variance (applied independently to both dimensions).

    Returns
    -------
    np.ndarray
        Array of shape (J, 2) of samples.
    """
    std = np.sqrt(variance)
    x = rng.normal(mean[0], std, size=J)
    y = rng.normal(mean[1], std, size=J)
    return np.column_stack((x, y))


def gaussian_mixture_sampler_by_var(
    mean_a: np.ndarray, mean_b: np.ndarray, var: float, weight: float = 0.5
) -> Callable[[np.random.Generator, int], np.ndarray]:
    """Wrapper function to draw from example_one.sample_gaussian_mixture with
    specified means/variance. Needed since sampler_functions.sample_distr()
    requires input of form:

        distr(rng: numpy.random.Generator, J; int) -> (J, 2) numpy array

    Parameters
    ----------
    mean_a, mean_b: (2, ) numpy array
        Mean vectors of each component of Gaussian mixture
    var : float
        Scalar variance for each independent component.
    weight : float
        Mixture weight for component A.

    Returns
    -------
    Callable[[np.random.Generator, int], np.ndarray]
        A function ``sampler(rng, J)`` that draws ``J`` samples with given distribution
    """

    return lambda rng, J: sample_gaussian_mixture(
        rng=rng, n=J, mean_a=mean_a, mean_b=mean_b, var=var, weight=weight
    )


def probs_to_mesh(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Convert tile-wise probabilities into matplotlib.pyplot.pcolormesh() inputs.

    The DataFrame df must contain columns: xmin, xmax, ymin, ymax, prob.
    Tiles are sorted by (y, x) and reshaped into a (ny, nx) grid.
    The DataFrame is the output of running one of
    estimate_functions.synthetic_scp() or
    estimate_functions.empirical_scp().

    Parameters
    ----------
    df : pd.Dataframe
        Dataframe with tile boundaries and probabilities.

    Returns
    -------
    x_edges : (nx, ) numpy array
        Sorted unique x-boundaries.
    y_edges : (ny, ) numpy array
        Sorted unique y-boundaries.
    Z : (ny - 1, nx - 1) numpy array
        Grid of probability values
    """
    x_edges = np.unique(np.concatenate([df.xmin, df.xmax]))
    y_edges = np.unique(np.concatenate([df.ymin, df.ymax]))

    # Note that in our usage ny = nx = h, since each axis of Lambda
    # is sliced into h pieces, to get h^2 rectangles.
    nx, ny = len(x_edges), len(y_edges)

    # If you swap ymin/xmin, transpose is plotted
    Z = df.sort_values(["ymin", "xmin"]).prob.to_numpy().reshape(ny - 1, nx - 1)

    return x_edges, y_edges, Z


def subsample(df: pd.DataFrame, seed: int, size: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly sample rows of a dataframe of points in R^2 for plotting purposes.

    Parameters
    ----------
    df : pd.DataFrame
        Must have x and y columns.
    seed : int
        For reproducible sampling.
    sample_size : int, default 1000
        Number of rows to sub-sample from.

    Returns
    -------
    x, y : np.ndarray
        Two arrays each of length `sample_size` taken from df.
    """
    x, y = df.x.to_numpy(), df.y.to_numpy()
    rng = np.random.default_rng(seed)

    # Pick 1000 indices from {0, 1, ..., len(x)}
    idx = rng.choice(len(x), size, replace=False)
    x, y = x[idx], y[idx]

    return x, y


def style_and_annotate(
    ax: Axes,
    bounds: tuple[float, float, float, float],
    bg: tuple[float, float, float],
    label_pos: tuple[float, float],
) -> None:
    """
    Apply plot styling and add the Lambda label.

    Parameters
    ----------
    ax : Axes
        Axis to modify.
    bounds : tuple
        (xmin, xmax, ymin, ymax) for axis limits.
    bg : color
        Background color.
    label_pos : tuple[float, float]
        (x, y) coordinate to place capital Greek letter Lambda in plot.
    """
    ax.set_facecolor(bg)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=13)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(
        label_pos[0],
        label_pos[1],
        r"$\Lambda$",
        color="white",
        fontsize=15,
        weight="bold",
    )


def main():
    """
    Run the estimator for each TGD variance and display SCP solutions.
    Results shown on 2x2 grid, where TGD points are overlaid on an SCP
    solution heatmap and the bottom-right plot is the prior.
    """
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)

    # Precompute prior samples once using a fixed seed
    _rng_prior = np.random.default_rng(SEED + 123)
    _prior_samples = gaussian_sampler(rng=_rng_prior, J=J, mean=PRIOR_MEAN, variance=PRIOR_VAR)

    # Dummy function to ensure same prior points are used
    def prior_fixed(_rng, _J):
        return _prior_samples

    # Run estimation for each variance
    fits = {}
    for i, var in enumerate(TGD_VARS):
        tgd = gaussian_mixture_sampler_by_var(mean_a=MU_A, mean_b=MU_B, var=var, weight=0.5)
        probs_df, prior_df, tgd_df = synthetic_scp(
            K=K,
            J=J,
            M=M,
            Q=Q,
            tgd=tgd,
            prior=prior_fixed,
            lambda_bounds=GRID_BOUNDS,
            lambda_grid_size=LAMBDA_GRID_SIZE,
            seed=SEED + i,
        )
        fits[var] = dict(probs=probs_df, prior=prior_df, tgd=tgd_df)

    # Color normalization across subplots
    # https://matplotlib.org/stable/users/explain/colors/colormapnorms.html#power-law
    vmax = max(fits[v]["probs"]["prob"].max() for v in TGD_VARS)
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, constrained_layout=True)
    for i, var in enumerate(TGD_VARS):
        ax = axes.ravel()[i]
        # Heatmap of eSCP solution
        ax.pcolormesh(
            *probs_to_mesh(fits[var]["probs"]),
            cmap="viridis",
            norm=norm,
            shading="auto",
            rasterized=True,
        )

        # Get TGD sample and add scatterplot
        tgd_x, tgd_y = subsample(df=fits[var]["tgd"], seed=SEED + i, size=1000)

        ax.scatter(tgd_x, tgd_y, c=TGD_COLOR, **SCATTERPLOT_OPTIONS)

        # Plot customization
        style_and_annotate(ax=ax, bounds=GRID_BOUNDS, bg=PURPLE_BG, label_pos=(4, -4.6))

    # Prior-only panel (bottom-right)
    ax_pr = axes.ravel()[3]

    # Get prior sample and make scatterplot
    prior_x, prior_y = subsample(df=fits[TGD_VARS[0]]["prior"], seed=SEED, size=1000)
    ax_pr.scatter(prior_x, prior_y, c=PRIOR_COLOR, **SCATTERPLOT_OPTIONS)

    # Prior plot customization
    style_and_annotate(ax=ax_pr, bounds=GRID_BOUNDS, bg=PURPLE_BG, label_pos=(4, -4.6))

    fig.savefig(out_dir / "example_two.pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
