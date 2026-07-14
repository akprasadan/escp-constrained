"""
Integration test for SCP algorithm.

Ensures pushforward of estimated distribution on Lambda matches that of the
input. We draw a sample {q_1, ..., q_K}, run the empirical SCP algorithm to
generate a heatmap prob_df. Then sample lambda_1, ..., lambda_K from prob_df,
apply Q, and visually assess that the distribution of Q(lambda_1), ...,
Q(lambda_K) resembles that of q_1, ..., q_K.

Output:

  - plots/tests/integrated_test_TGD_and_estimate_and_sampled_estimate.pdf
  - plots/tests/integrated_test_pushforwards.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data_processing import probs_to_mesh
from src.estimate_functions import synthetic_scp
from src.output_functions import apply_map
from src.sampler_functions import (
    sample_from_SCP,
    sample_gaussian_mixture,
    sample_multivar_gaussian,
)

SEED_BASE = 20251028
GRID_BOUNDS = (-5, 5, -5, 5)
J, K, M, LAMBDA_GRID_SIZE = 75_000, 150_000, 100, 100


MU_A, MU_B = np.array([-1, -1]), np.array([2, 2])

PRIOR_COMPONENT = "B"  # 'A' or 'B' (prior mean is nudged toward the origin)
PRIOR_OFFSET = (0.75, 0.75)  # how much to nudge toward the origin
PRIOR_MEAN = (MU_A + PRIOR_OFFSET) if PRIOR_COMPONENT == "A" else (MU_B - PRIOR_OFFSET)
PRIOR_VAR = 2.5

PCOLORMESH_OPTIONS = {"cmap": "viridis", "shading": "auto", "rasterized": True}


def Q(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Make simple real-valued quadratic Q.

    Returns
    -------
    numpy array
    """
    return x**2 - 3 * y**2


def gaussian_mixture_tgd(rng: np.random.Generator, J: int) -> np.ndarray:
    """Generate equal mixture of Gaussians.

    Returns
    -------
    (J, 2) numpy array
    """
    return sample_gaussian_mixture(
        mean_a=MU_A,
        mean_b=MU_B,
        var_a=0.1 * np.identity(2),
        var_b=0.1 * np.identity(2),
        weight=0.5,
        rng=rng,
        J=J,
    )


def gaussian_prior(rng: np.random.Generator, J: int) -> np.ndarray:
    """Generate bivariate Gaussian at some mean with diagonal covariance.

    Returns
    -------
    (J, 2) numpy array
    """
    return sample_multivar_gaussian(
        mean=PRIOR_MEAN, cov=PRIOR_VAR * np.identity(2), rng=rng, J=J
    )


def make_pushforward_diagnostic_plots() -> None:
    """Make pushforward diagnostic plots."""
    probs_df, _, tgd_df = synthetic_scp(
        K=K,
        J=J,
        M=M,
        Q=Q,
        tgd=gaussian_mixture_tgd,
        prior=gaussian_prior,
        grid_mins=np.array([GRID_BOUNDS[0], GRID_BOUNDS[2]]),
        grid_maxes=np.array([GRID_BOUNDS[1], GRID_BOUNDS[3]]),
        lambda_grid_size=LAMBDA_GRID_SIZE,
        seed=SEED_BASE,
    )

    lambda_sample_x, lambda_sample_y = sample_from_SCP(probs_df, 50_000, SEED_BASE)

    lambda_sample = np.column_stack((lambda_sample_x, lambda_sample_y))
    pushforward_sample = apply_map(lambda_sample, Q)

    out_dir = Path("plots/tests")
    out_dir.mkdir(parents=True, exist_ok=True)
    path1 = out_dir / "integrated_test_TGD_and_estimate_and_sampled_estimate.pdf"
    path2 = out_dir / "integrated_test_pushforwards.pdf"

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 1) Sample from TGD (points -> heatmap)
    H, xedges, yedges = np.histogram2d(tgd_df["x"], tgd_df["y"], bins=100)
    axes[0, 0].pcolormesh(xedges, yedges, H.T, **PCOLORMESH_OPTIONS)
    axes[0, 0].set_title("Sample from TGD")
    axes[0, 0].set_aspect("equal")

    # 2) SCP solution (mesh already given)
    x_edges, y_edges, Z = probs_to_mesh(probs_df)
    axes[0, 1].pcolormesh(x_edges, y_edges, Z, **PCOLORMESH_OPTIONS)
    axes[0, 1].set_title("SCP Solution")
    axes[0, 1].set_aspect("equal")

    # 3) Sample from SCP solution (points -> heatmap)
    H, xedges, yedges = np.histogram2d(lambda_sample_x, lambda_sample_y, bins=100)
    axes[1, 0].pcolormesh(xedges, yedges, H.T, **PCOLORMESH_OPTIONS)
    axes[1, 0].set_title("Empirical Sample from SCP solution")
    axes[1, 0].set_aspect("equal")

    # 4) Empty
    axes[1, 1].axis("off")

    plt.tight_layout()
    fig.savefig(path1)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(tgd_df["q_x"], bins=50)
    axes[0].set_title("Actual TGD Pushforward")

    axes[1].hist(pushforward_sample, bins=50)
    axes[1].set_title("Estimated SCP Solution Pushforward")

    plt.tight_layout()
    fig.savefig(path2)
    plt.close(fig)


def test_pushforward_property() -> None:
    """Run the pushforward diagnostic check if it has not already been done."""
    paths = [
        Path("plots/tests/integrated_test_TGD_and_estimate_and_sampled_estimate.pdf"),
        Path("plots/tests/integrated_test_pushforwards.pdf"),
    ]

    if not all(p.exists() for p in paths):
        make_pushforward_diagnostic_plots()
