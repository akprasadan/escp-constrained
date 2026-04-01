"""
example_one.py

Generates and saves two figures:
  1) 3D surface of a bivariate KDE for an unequal Gaussian mixture.
  2) 1D KDE of Q(x, y) where Q(x, y) = a*x^2 + b*y^2.

Outputs:
  - plots/example_one_3D_KDE.pdf
  - plots/example_one_1D_KDE.pdf
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

# Use higher quality + LaTeX fonts in Matplotlib
plt.rcParams.update({"pdf.fonttype": 42, "text.usetex": True})

SEED = 20251028

# Gaussian mixture configs
MEAN_A = np.array([-1, -1])
MEAN_B = np.array([2, 2])
VAR = 1.0
WEIGHT = 0.7
N_SAMPLES = 100_000

# Grid config
GRID_X_MIN, GRID_X_MAX = 0, 75  # Limits for 1D KDE curve
GRID_XY_MIN, GRID_XY_MAX = -6.0, 6.0  # Limits for 2D KDE surface
NX, NY = 150, 150  # Resolution of 2D KDE surface
NQ = 400  # Resolution of 1D KDE curve

# Map config: Q(x, y) = Q_A*x^2 + Q_B*y^2
Q_A, Q_B = 1.0, 3.0

# Plot config
PURPLE_BG = (68 / 255, 1 / 255, 84 / 255)
OUT_DIR = Path("plots/example_one")
OUT_3D = Path("3D_KDE.pdf")
OUT_1D = Path("1D_KDE.pdf")
DPI = 1200


def Q(x, y):
    return Q_A * x**2 + Q_B * y**2


def sample_gaussian_mixture(
    rng: np.random.Generator,
    n: int,
    mean_a: np.ndarray,
    mean_b: np.ndarray,
    var: float,
    weight: float,
) -> np.ndarray:
    """
    Draw samples from a 2D Gaussian mixture with two components
    and diagonal covariance var*I.

    Parameters
    -----------
    rng : np.random.Generator
        For reproducibility.
    n : int
        Number of samples
    mean_a, mean_b: (2, ) numpy array
        Mean vectors of each component of Gaussian mixture
    var : float
        Variance of each component of the Gaussian mixture
    weight : float
        The weight of the first component of the Gaussian mixture

    Returns
    -------
    x, y : (n, ) numpy array
        First and second coordinate arrays, each of length n.
    """

    std = np.sqrt(var)

    # Generate n Unif[0,1] RVs, and let those < 0.5 be True, rest False
    comp = rng.random(n) < weight

    # All the 'true' components are assigned one Gaussian distribution
    # Rest are assigned the other Gaussian distribution
    samples = np.empty((n, 2))
    samples[comp] = rng.normal(loc=mean_a, scale=std, size=(comp.sum(), 2))
    samples[~comp] = rng.normal(loc=mean_b, scale=std, size=((~comp).sum(), 2))

    return samples


def kde_2d(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    nx: int,
    y_min: float,
    y_max: float,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bivariate KDE on a regular grid. Indexing is artifact of how matplotlib.pyplot.contour()
    works.

    See also https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
    The example included is similar but uses np.mgrid instead of np.meshgrid.
    This is why it includes a transpose before reshaping.

    Parameters
    ----------
    x, y : (n, ) numpy array
        Input sample coordinates, each array of same length n
    x_min, x_max, ymin, ymax : float
        Range of the grid in the x-direction and y-direction.
    nx, ny : int
        Number of grid points along x and y directions.
    y_min, y_max : float
        Range of the grid in the y-direction.

    Returns
    -------
    X, Y : (ny, nx) numpy array
        Meshgrid arrays of points on the grid.
    Z : (ny, nx) numpy array
        KDE values evaluated on the grid, same shape as X.
    """

    # Form (2, n) array, fit Gaussian KDE
    xy_data = np.vstack([x, y])
    kernel = gaussian_kde(xy_data)

    # Form grid of points to evaluate fitted KDE at
    x_grid = np.linspace(x_min, x_max, nx)
    y_grid = np.linspace(y_min, y_max, ny)

    # Let X be ny rows of x_grid, Y be nx rows of y_grid
    X, Y = np.meshgrid(x_grid, y_grid, indexing="xy")

    # Flatten X and Y with ravel(), so, e.g., X becomes
    # a 1D vector of length ny*nx, repeating x_grid again and again.
    # Stack flattened X and Y together, yielding (2, ny*nx) array
    # which is all points we need to evaluate KDE at.
    # Lastly, put Z in same shape as X or Y, so it can be fed into plot_surface().
    Z = kernel(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

    return X, Y, Z


def kde_1d(values: np.ndarray, n_grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1D KDE on an evenly spaced grid spanning [0, max(values)].

    Parameters
    ----------
    values : (n, ) numpy array
        Input samples.
    n_grid : int
        Number of grid points.

    Returns
    -------
    grid : (n_grid, ) numpy array
        Grid points.
    dens : (n_grid, ) numpy array
        KDE values on the grid, same shape as grid.
    """
    vmax = values.max()
    grid = np.linspace(0.0, vmax, n_grid)
    dens = gaussian_kde(values)(grid)
    return grid, dens


def plot_3d_surface(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Figure:
    """3D surface of the bivariate KDE.

    Parameters
    ----------
    X, Y : (ny, nx) numpy array
        Meshgrid arrays of shape (ny, nx), where, e.g., nx is number of grid points on axis x.
    Z : (ny, nx) numpy array
        KDE values evaluated on the grid, same shape as X.

    Returns
    -------
    fig : Figure
    """
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection="3d")

    # Plot x^2 + 3y^2 = 20 as a contour in the z=0 plane
    F = Q_A * X**2 + Q_B * Y**2 - 20.0

    # https://stackoverflow.com/a/30146280
    ax.contour(X, Y, F, levels=[0], colors=PURPLE_BG, linewidths=2.2)

    # https://stackoverflow.com/a/11409882
    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)

    ax.yaxis.set_rotate_label(False)  # type: ignore

    ax.set_xlabel(r"$\lambda_1$", labelpad=-5, fontsize=10)
    ax.set_ylabel(r"$\lambda_2$", labelpad=-5, fontsize=10)

    ax.set_xlim(GRID_XY_MIN, GRID_XY_MAX)
    ax.set_ylim(GRID_XY_MIN, GRID_XY_MAX)
    ax.set_zlim(0, Z.max())
    ax.view_init(elev=35, azim=-100)

    # Put 4 ticks on each axis
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.tick_params(axis="x", pad=-3)
    ax.tick_params(axis="y", pad=-3)
    ax.tick_params(labelsize=8)

    ax.text(4, -5.2, 0, r"$\Lambda$", color="white", fontsize=12, weight="bold", zorder=5)
    # Remove gridlines + grey background + spines
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("none")  # type: ignore
        axis.pane.fill = False  # type: ignore

    return fig


def plot_1d_q_density(q_grid: np.ndarray, q_density: np.ndarray) -> Figure:
    """1D KDE plot of Q(x, y).

    Parameters
    ----------
    q_grid : (n, ) numpy array
        Array of evaluations of Q(x, y)
    q_density : (n, ) numpy array
        KDE values on the grid, same shape as grid.

    Returns
    -------
    fig : Figure
    """
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.plot(q_grid, q_density, color=PURPLE_BG, linewidth=2.2)
    ax.axvline(x=20, color="black", linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"$\mathcal{D}$", fontsize=15)
    ax.set_xlim(GRID_X_MIN, GRID_X_MAX)
    ax.set_ylim(0.0, q_density.max() * 1.2)
    ax.tick_params(labelsize=13)
    ax.grid(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    return fig


def main():
    rng = np.random.default_rng(SEED)
    tgd_data = sample_gaussian_mixture(
        rng=rng, n=N_SAMPLES, mean_a=MEAN_A, mean_b=MEAN_B, var=VAR, weight=WEIGHT
    )
    x_tgd, y_tgd = tgd_data[:, 0], tgd_data[:, 1]

    x_kde, y_kde, z_kde = kde_2d(
        x_tgd,
        y_tgd,
        x_min=GRID_XY_MIN,
        x_max=GRID_XY_MAX,
        nx=NX,
        y_min=GRID_XY_MIN,
        y_max=GRID_XY_MAX,
        ny=NY,
    )
    fig3d = plot_3d_surface(x_kde, y_kde, z_kde)

    out_path_3d = OUT_DIR / OUT_3D
    out_path_3d.parent.mkdir(parents=True, exist_ok=True)
    fig3d.savefig(out_path_3d, bbox_inches="tight", pad_inches=0.2, dpi=DPI)
    plt.close(fig3d)

    qvals = Q(x_tgd, y_tgd)
    q_grid, qdens = kde_1d(qvals, n_grid=NQ)
    fig1d = plot_1d_q_density(q_grid, qdens)

    out_path_1d = OUT_DIR / OUT_1D
    out_path_1d.parent.mkdir(parents=True, exist_ok=True)
    fig1d.savefig(out_path_1d, bbox_inches="tight", pad_inches=0.2, dpi=DPI)
    plt.close(fig1d)


if __name__ == "__main__":
    main()
