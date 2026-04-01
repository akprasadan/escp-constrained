"""
example_three.py

Run eSCP estimates on concrete data from Yeh (1998) to demonstrate continuity.
We consider a modification of a well-known relationship called Abram's Law, where the compressive
concrete strength (q) is a power law function of the water to binder ratio, which we denote R.
The binder content is the sum of cement, fly ash, and slag content. In equation:

    q = a * (R ** b)

We let Lambda be the set of (a, b) pairs defining a power law. We will use a physically
meaningful subset of values: e.g., "a" should be positive and "b" should be negative.
We show through our prior that our domain assumptions permit a broad array of curves, many of which
are clearly not physically meaningful, i.e., we are not over-assisting the algorithm.

Pipeline:
1. Subset strength values in a narrow band of R (around 0.3) and age 0-25 days.
2. Bootstrap/jitter the strength data to get a suitable sample size
3. Run eSCP on this to estimate the (a, b), using Gaussian prior.
4. Sample (a, b) pairs from both eSCP estimate and prior.
5. Plot eSCP solution, prior, and sampled power laws
   from both eSCP estimate and prior with actual (R, q) data included.
6. Repeat this but now take age to be 25-50. Show same plot, but skip prior since unchanged.

Data will be downloaded from https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength

I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial
neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998)

Outputs:
  - data/raw_concrete.csv
  - data/processed_concrete.csv
  - plots/example_three/R_0.3_age_25_50.pdf
  - plots/example_three/R_0.3_age_0_25.pdf
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.collections import QuadMesh
from matplotlib.axes import Axes
from ucimlrepo import fetch_ucirepo

from src.estimate_functions import empirical_scp as empirical_scp
from src.example_two import probs_to_mesh as probs_to_mesh

# Use higher quality + LaTeX fonts in Matplotlib
plt.rcParams.update({"pdf.fonttype": 42, "text.usetex": True})


SEED = 20251028

# Data paths
RAW_DATA_PATH = Path("data/raw_concrete.csv")
PROCESSED_DF_PATH = Path("data/processed_concrete.csv")

# Prior distribution parameters
PRIOR_A_MEAN, PRIOR_A_SD = 7, 1.5
PRIOR_B_MEAN, PRIOR_B_SD = -1.5, 1

# eSCP configs
POOL_SIZE = 100_000  # Number of bootstrapped q's
PRIOR_COUNT = 250_000  # Importance sampling algorithm sample size
A_BOUNDS, B_BOUNDS = (4, 14), (-3, -0.1)  # Bounding box for Lambda
D_BIN = 100  # Bin count to partition output space D

# Concrete data options
# Water binder ratio in R +/- Delta_R
R = 0.3
DELTA_R = 0.1

# Age in [0, 25] for experiment 1, then [25, 50] for experiment 2
AGE_MIN, AGE_MAX = 0, 25
AGE_MIN_TWO, AGE_MAX_TWO = 25, 50

# When bootstrapping strength values, add mean 0 Gaussian noise of this SD
JITTER_SD = 5

# Number of sampled power laws for visualization of solution
CURVE_COUNT = 30

# Plot features
DPI, FIGSIZE = 1200, (7.5, 7.5)
FIG_SIZE_SMALL = (FIGSIZE[0], FIGSIZE[1] / 2)
PURPLE_BG = (68 / 255, 1 / 255, 84 / 255)
PRIOR_COLOR = "#f4d03f"
RED_COLOR = "#cc4778"


# Strength map
def Q(x, y):
    return x * (R**y)


def concrete_gaussian_prior(rng: np.random.Generator, J: int) -> np.ndarray:
    """
    Gaussian prior for concrete eSCP estimate.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    J : int
        Number of samples to draw.

    Returns
    -------
    (J, 2) numpy array
    """
    x = rng.normal(PRIOR_A_MEAN, PRIOR_A_SD, size=J)
    y = rng.normal(PRIOR_B_MEAN, PRIOR_B_SD, size=J)
    return np.column_stack((x, y))


def download_concrete_data(path: Path) -> None:
    """
    Download and save dataset from UC Irvine ML Repository, only if not already present.

    Parameters
    ----------
    path : Path
        File location to store raw dataset.
    """

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        df = fetch_ucirepo(id=165).data.original  # type: ignore
        df.to_csv(path, index=False)


def process_concrete_data(input_path: Path, output_path) -> pd.DataFrame:
    """Read, compute binder and ratios, subset columns, and save to data/ directory.
    We do not perform any subsetting of age or water/binder ratios in the output.

    Parameters
    -----------
    input_path : Path
        Location of raw .csv file with concrete data downloaded from original source.
    output_path : Path
        Location of processed dataframe of concrete data.

    Returns
    --------
    df : pd.Dataframe
        Cleaned dataframe with water/binder ratio calculated.
    """
    # Download and save data only if not already present
    download_concrete_data(input_path)

    df = pd.read_csv(input_path)

    # This order is correct, but probably good to make code agnostic to order of columns.
    df.columns = [
        "cement",
        "slag",
        "fly_ash",
        "water",
        "superplasticizer",
        "course_aggregate",
        "fine_aggregate",
        "age_days",
        "strength",
    ]
    df = df.apply(pd.to_numeric)
    df["binder"] = df["cement"] + df["fly_ash"] + df["slag"]
    df["water_binder_ratio"] = df["water"] / df["binder"]

    df = df[["strength", "water", "binder", "water_binder_ratio", "age_days"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def generate_bootstrapped_strength(
    df: pd.DataFrame, age_min: float, age_max: float, count: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Bootstrap an appropriate subset of concrete strength values, and add
    Gaussian noise to resampled values.

    Parameters
    -----------
    df : pd.DataFrame
        Dataframe with columns strength, water_binder_ratio, age_days
    age_min, age_max : float
        Restriction to concrete in this age range
    count : int
        Number of resamples to draw
    rng : np.random.Generator
        Random number generator for reproducibility

    Returns
    -------
    noisy_strength_bootstraps : np.ndarray
        Array of shape (J, ) of bootstrapped/noisy strength values
    """
    # Subset dataframe in right R, age range
    mask = (
        (df["water_binder_ratio"] >= R - DELTA_R)
        & (df["water_binder_ratio"] <= R + DELTA_R)
        & (df["age_days"] <= age_max)
        & (df["age_days"] >= age_min)
    )

    # Obtain strength data and then create large bootstrapped sample,
    # jittered by Gaussian noise
    strength_vals = df.loc[mask, "strength"]

    gaussian_noise = rng.normal(0, JITTER_SD, size=count)
    bootstrapped_strength = rng.choice(strength_vals, size=count, replace=True)
    noisy_strength_bootstraps = bootstrapped_strength + gaussian_noise

    return noisy_strength_bootstraps


def plot_strength_vs_ratio(
    df: pd.DataFrame,
    ax: Axes,
    a_vector: np.ndarray,
    b_vector: np.ndarray,
    age_min: float,
    age_max: float,
) -> None:
    """
    Produce a scatter plot of strength vs. water/binder ratio.
    Superimpose a series of curves of the form y = a * x^b.
    Only shows data in a fixed age range.

    Parameters
    -----------
    df : pd.DataFrame
        Dataframe with columns 'strength' and 'water_binder_ratio'.
    ax : Axes
        Axes of some subplot.
    a_vector, b_vector : numpy array
        Arrays of values of "a" and "b", respectively, in power law y = ax^b.
        Must be of the same length.
    age_min, age_max : float
        Age range of concrete data to restrict.
    """
    df = df[(df["age_days"] <= age_max) & (df["age_days"] >= age_min)]

    ax.scatter(df["water_binder_ratio"], df["strength"], s=3)

    x_grid = np.linspace(0.1, 1, 300)
    for a, b in zip(a_vector, b_vector):
        ax.plot(x_grid, a * x_grid**b, alpha=0.2, linewidth=1)

    ax.axvspan(R - DELTA_R, R + DELTA_R, color="grey", alpha=0.15)
    ax.axvline(R - DELTA_R, color="grey", linestyle="--", alpha=0.5, linewidth=0.7)
    ax.axvline(R, color="grey", alpha=0.5, linestyle="--", linewidth=1)
    ax.axvline(R + DELTA_R, color="grey", alpha=0.5, linestyle="--", linewidth=0.7)
    ax.tick_params(labelsize=13)
    ax.set_xlabel("Water / Binder", fontsize=12)
    ax.set_ylabel("Strength (MPa)", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 85)


def plot_eSCP_estimate(
    probs_df: pd.DataFrame, ax: Axes, norm: PowerNorm, a_samples: np.ndarray, b_samples: np.ndarray
) -> QuadMesh:
    """
    Render a probability mesh from `probs_df` onto a supplied axis. Also
    plot some (a, b) pairs that are supplied.

    Expects `probs_to_mesh(probs_df)` to return:
        x_edges: array-like of x bin edges
        y_edges: array-like of y bin edges
        Z: 2D array of probabilities aligned with the mesh.

    Parameters
    ----------
    probs_df : pd.Dataframe
        Dataframe with columns xmin, xmax, ymin, ymax, prob storing eSCP solution.
    ax : Axes
        Axis to draw the mesh on.
    norm : PowerNorm
        Color scaling object for pcolormesh.
    a_samples, b_samples : (n, ) numpy array
        Array of (a, b) coefficients for the power law y = a*x^b to plot as points.

    Returns
    -------
    mesh : QuadMesh
        The matplotlib mesh object.
    """
    x_edges, y_edges, Z = probs_to_mesh(probs_df)

    mesh = ax.pcolormesh(
        x_edges, y_edges, Z, norm=norm, cmap="viridis", shading="auto", rasterized=True
    )

    ax.set_xlabel(r"$a$", fontsize=14)
    ax.set_ylabel(r"$b$", fontsize=14)

    ax.scatter(a_samples, b_samples, s=5, color=RED_COLOR)
    ax.tick_params(labelsize=13)
    ax.text(13, -2.9, r"$\Lambda$", color="white", fontsize=15, weight="bold")

    return mesh


def plot_prior(prior_sample: pd.DataFrame, ax: Axes) -> None:
    """
    Customizations for scatterplot of prior points.

    Parameters
    ----------
    prior_sample : (N, 2) numpy array
        Array of shape
    ax : Axes
        Existing axes object for plot
    """
    ax.set_facecolor(PURPLE_BG)
    ax.scatter(
        prior_sample.x,
        prior_sample.y,
        s=5,
        color=PRIOR_COLOR,
        alpha=0.95,
        edgecolors="none",
        rasterized=False,
    )
    ax.set_xlim(*A_BOUNDS)
    ax.set_ylim(*B_BOUNDS)
    ax.set_xlabel(r"$a$", fontsize=14)
    ax.set_ylabel(r"$b$", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.text(13, -2.9, r"$\Lambda$", color="white", fontsize=15, weight="bold")


def sample_from_eSCP(df: pd.DataFrame, n: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """
    Given eSCP solution, sample from the estimated heatmap.

    Expects dataframe output from empirical_scp(),
    with columns xmin, xmax, ymin, ymax, and prob. The 'prob' column is assured to sum to 1 already.
    We will treat the grid as a single long vector X = (X_1, ...., X_m) to sample from.
    Using the prob vector, compute the CDF F(X_i) over each X_i using np.cumsum() over `probs'.
    Then generate U_1, ..., U_n ~ Unif(0,1) IID, and find the "X_i" in the grid such
    that F(X_i) is approximately U_i. Then X_i will be distributed as F.
    E.g., if the cumsum vector is [0, 0.2, 0.5, 0.7, 0.9, 1], corresponding to six = 2 x 3 grid points,
    and we draw 0.3, this corresponds to picking the third (xmin, xmax, ymin, ymax) quadruplet.


    Parameters
    -----------
    df : pd.DataFrame
        First output dataframe, probs_df, from empirical_scp().
    n : int
        Number of lambda_i samples to draw.
    seed : int
        Seed used to initialize an RNG for reproducibility.

    Returns
    --------
    A tuple of 2 numpy arrays
    """
    rng = np.random.default_rng(seed)

    # Compute CDF over the flattened grid
    cdf = np.cumsum(df["prob"])
    u = rng.random(n)
    # Sample from indices using the CDF
    idx = np.searchsorted(cdf, u)

    # Sample uniformly inside the chosen rectangles
    xmin, xmax, ymin, ymax = df["xmin"], df["xmax"], df["ymin"], df["ymax"]

    x = rng.uniform(xmin[idx], xmax[idx])
    y = rng.uniform(ymin[idx], ymax[idx])

    return x, y


def main():
    # Experiment 1
    df = process_concrete_data(input_path=RAW_DATA_PATH, output_path=PROCESSED_DF_PATH)

    rng = np.random.default_rng(SEED)

    # Obtain filtered strength data (in desired R + age range) and then create large
    # bootstrapped sample jittered by Gaussian noise.
    strength_data_bootstrapped = generate_bootstrapped_strength(
        df, AGE_MIN, AGE_MAX, POOL_SIZE, rng
    )

    # Fit eSCP algorithm
    probs_df, prior_df = empirical_scp(
        q_data=strength_data_bootstrapped,
        J=PRIOR_COUNT,
        M=D_BIN,
        Q=Q,
        prior=concrete_gaussian_prior,
        lambda_bounds=(*A_BOUNDS, *B_BOUNDS),
        lambda_grid_size=50,
        seed=SEED,
    )
    # Generate samples of (a, b) pairs from fitted distribution, corresponding to power laws
    a_samples, b_samples = sample_from_eSCP(probs_df, CURVE_COUNT, SEED)

    # Prepare plot 1
    vmax = probs_df["prob"].max()
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, constrained_layout=True)
    axes = axes.ravel()
    fig.set_constrained_layout_pads(hspace=0.1)  # type: ignore
    ax_eSCP = axes[2]
    plot_eSCP_estimate(probs_df, ax_eSCP, norm, a_samples, b_samples)

    ax_strength = axes[3]
    plot_strength_vs_ratio(df, ax_strength, a_samples, b_samples, AGE_MIN, AGE_MAX)

    ax_prior = axes[0]
    # Take subset of prior data (for plotting purposes)
    prior_sample = prior_df[["x", "y"]].sample(n=1000, replace=False, random_state=SEED)
    plot_prior(prior_sample, ax_prior)

    ax_strength_prior = axes[1]
    prior_a_samples = prior_sample.x[0 : (CURVE_COUNT - 1)].to_numpy()
    prior_b_samples = prior_sample.y[0 : (CURVE_COUNT - 1)].to_numpy()
    plot_strength_vs_ratio(
        df, ax_strength_prior, prior_a_samples, prior_b_samples, AGE_MIN, AGE_MAX
    )

    out_path = Path(f"plots/example_three/R_{R}_age_{AGE_MIN}_{AGE_MAX}.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # Experiment 2, using different age group
    strength_data_bootstrapped = generate_bootstrapped_strength(
        df, AGE_MIN_TWO, AGE_MAX_TWO, POOL_SIZE, rng
    )

    # Refit eSCP estimate
    probs_df, _ = empirical_scp(
        q_data=strength_data_bootstrapped,
        J=PRIOR_COUNT,
        M=D_BIN,
        Q=Q,
        prior=concrete_gaussian_prior,
        lambda_bounds=(*A_BOUNDS, *B_BOUNDS),
        lambda_grid_size=50,
        seed=SEED,
    )

    # Sample power laws parameters from eSCP estimate
    a_samples, b_samples = sample_from_eSCP(probs_df, 30, SEED)

    # Prepare plot 2
    vmax = probs_df["prob"].max()
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_SMALL, constrained_layout=True)
    axes = axes.ravel()

    ax_eSCP = axes[0]
    plot_eSCP_estimate(probs_df, ax_eSCP, norm, a_samples, b_samples)

    ax_strength = axes[1]
    plot_strength_vs_ratio(df, ax_strength, a_samples, b_samples, AGE_MIN_TWO, AGE_MAX_TWO)

    out_path = Path(f"plots/example_three/R_{R}_age_{AGE_MIN_TWO}_{AGE_MAX_TWO}.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
