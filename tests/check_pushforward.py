"""
Integration test for eSCP algorithm that ensures pushforward of estimated distribution on Lambda
matches that of the input.

We draw a sample {q_1, ..., q_K}, run the empirical SCP algorithm to generate a heatmap prob_df.
Then sample lambda_1, ..., lambda_K from prob_df, apply Q, and visually assess that the distribution
of Q(lambda_1), ..., Q(lambda_K) resembles that of q_1, ..., q_K.


Output:
  - plots/tests/integrated_test_TGD_and_estimate_and_sampled_estimate.pdf
  - plots/tests/integrated_test_pushforwards.pdf
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.output_functions import apply_map as apply_map
from src.estimate_functions import synthetic_scp as synthetic_scp
from src.example_two import gaussian_mixture_sampler_by_var as mixture_sampler
from src.example_two import gaussian_sampler as gaussian_sampler
from src.example_two import probs_to_mesh as probs_to_mesh
from src.example_three import sample_from_eSCP as sample_from_eSCP


SEED_BASE = 20251028
GRID_BOUNDS = (-5, 5, -5, 5)
J, K, M, LAMBDA_GRID_SIZE = 75_000, 150_000, 100, 100


MU_A, MU_B = np.array([-1, -1]), np.array([2, 2])

PRIOR_COMPONENT = "B"  # 'A' or 'B' (prior mean is nudged toward the origin)
PRIOR_OFFSET = (0.75, 0.75)  # how much to nudge toward the origin
PRIOR_MEAN = (MU_A + PRIOR_OFFSET) if PRIOR_COMPONENT == "A" else (MU_B - PRIOR_OFFSET)
PRIOR_VAR = 2.5

PCOLORMESH_OPTIONS = dict(cmap="viridis", shading="auto", rasterized=True)

TGD = mixture_sampler(mean_a=MU_A, mean_b=MU_B, var=0.1, weight=0.5)


def Q(x, y):
    return x**2 - 3 * y**2


def prior(rng, J):
    return gaussian_sampler(
        rng=rng,
        J=J,
        mean=PRIOR_MEAN,
        variance=PRIOR_VAR,
    )


probs_df, prior_df, tgd_df = synthetic_scp(
    K=K,
    J=J,
    M=M,
    Q=Q,
    tgd=TGD,
    prior=prior,
    lambda_bounds=GRID_BOUNDS,
    lambda_grid_size=LAMBDA_GRID_SIZE,
    seed=SEED_BASE,
)


lambda_sample_x, lambda_sample_y = sample_from_eSCP(probs_df, 50_000, SEED_BASE)

lambda_sample = np.column_stack((lambda_sample_x, lambda_sample_y))

pushforward_sample = apply_map(lambda_sample, Q)


def main():

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 1) Sample from TGD (points -> heatmap)
    H, xedges, yedges = np.histogram2d(tgd_df["x"], tgd_df["y"], bins=100)
    axes[0, 0].pcolormesh(xedges, yedges, H.T, **PCOLORMESH_OPTIONS)
    axes[0, 0].set_title("Sample from TGD")
    axes[0, 0].set_aspect("equal")

    # 2) eSCP solution (mesh already given)
    x_edges, y_edges, Z = probs_to_mesh(probs_df)
    axes[0, 1].pcolormesh(x_edges, y_edges, Z, **PCOLORMESH_OPTIONS)
    axes[0, 1].set_title("eSCP Solution")
    axes[0, 1].set_aspect("equal")

    # 3) Sample from eSCP solution (points -> heatmap)
    H, xedges, yedges = np.histogram2d(lambda_sample_x, lambda_sample_y, bins=100)
    axes[1, 0].pcolormesh(xedges, yedges, H.T, **PCOLORMESH_OPTIONS)
    axes[1, 0].set_title("Empirical Sample from eSCP solution")
    axes[1, 0].set_aspect("equal")

    # 4) Empty
    axes[1, 1].axis("off")

    plt.tight_layout()
    out_path = Path("plots/tests/integrated_test_TGD_and_estimate_and_sampled_estimate.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(tgd_df["q"], bins=50)
    axes[0].set_title("Actual TGD Pushforward")

    axes[1].hist(pushforward_sample, bins=50)
    axes[1].set_title("Estimated eSCP Solution Pushforward")

    plt.tight_layout()
    out_path = Path("plots/tests/integrated_test_pushforwards.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
