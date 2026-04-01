"""
Integration test for eSCP algorithm that ensures pushforward of estimated distribution on Lambda
matches that of the input.

We draw a sample {q_1, ..., q_K}, run the empirical SCP algorithm to generate a heatmap prob_df.
Then sample lambda_1, ..., lambda_K from prob_df, apply Q, and visually assess that the distribution
of Q(lambda_1), ..., Q(lambda_K) resembles that of q_1, ..., q_K.

Data will be downloaded from https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength

I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial
neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998)

Output:
  - data/raw_concrete.csv
  - data/processed_concrete.csv
  - plots/tests/integrated_test_concrete_SCP.pdf
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.estimate_functions import empirical_scp as empirical_scp
from src.output_functions import apply_map as apply_map
from src.example_two import probs_to_mesh as probs_to_mesh
from src.example_three import download_concrete_data as download_concrete_data
from src.example_three import process_concrete_data as process_concrete_data
from src.example_three import generate_bootstrapped_strength as bootstrap_strength
from src.example_three import concrete_gaussian_prior as concrete_gaussian_prior
from src.example_three import sample_from_eSCP as sample_from_eSCP

SEED = 20251028

# Number of bootstrapped q's for eSCP run
POOL_SIZE = 100_000

# Bounding box for Lambda
A_BOUNDS = (4, 14)
B_BOUNDS = (-3, -0.1)

# Data subsetting choices
# Water binder ratio in R +/- Delta_R
# Age in [0, 25] for primary experiment
R = 0.3
DELTA_R = 0.1
AGE_MIN = 0
AGE_MAX = 25

# Prior distribution parameters
PRIOR_A_MEAN, PRIOR_A_SD = 7, 1.5
PRIOR_B_MEAN, PRIOR_B_SD = -1.5, 1

# Load concrete data
RAW_DATA_PATH = Path("data/raw_concrete.csv")
PROCESSED_DF_PATH = Path("data/processed_concrete.csv")
df = process_concrete_data(input_path=RAW_DATA_PATH, output_path=PROCESSED_DF_PATH)

# Bootstrapped strength data
rng = np.random.default_rng(SEED)
strength_data_bootstrapped = bootstrap_strength(df, AGE_MIN, AGE_MAX, POOL_SIZE, rng)

# Plot configs
PCOLORMESH_OPTIONS = dict(cmap="viridis", shading="auto", rasterized=True)


# Strength function
def Q(x, y):
    return x * (R**y)


probs_df, prior_df = empirical_scp(
    q_data=strength_data_bootstrapped,
    J=250_000,
    M=100,
    Q=Q,
    prior=concrete_gaussian_prior,
    lambda_bounds=(*A_BOUNDS, *B_BOUNDS),
    lambda_grid_size=50,
    seed=SEED,
)

a_sample, b_sample = sample_from_eSCP(probs_df, 50_000, SEED)

pushforward_sample = apply_map(np.column_stack((a_sample, b_sample)), Q)


def main():

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 1) eSCP solution
    x_edges, y_edges, Z = probs_to_mesh(probs_df)
    axes[0, 0].pcolormesh(x_edges, y_edges, Z, **PCOLORMESH_OPTIONS)
    axes[0, 0].set_title("eSCP Solution")

    # 2) Sample from eSCP solution
    H, xedges, yedges = np.histogram2d(a_sample, b_sample, bins=100)
    axes[1, 0].pcolormesh(xedges, yedges, H.T, **PCOLORMESH_OPTIONS)
    axes[1, 0].set_title("Empirical Sample from eSCP solution")

    # 3) Histogram of TGD pushforward
    axes[0, 1].hist(strength_data_bootstrapped, bins=50)
    axes[0, 1].set_title("Actual TGD Pushforward")
    axes[0, 1].set_xlim(0, 100)

    # 4) Histogram of eSCP solution pushforward
    axes[1, 1].hist(pushforward_sample, bins=50)
    axes[1, 1].set_title("Estimated eSCP Solution Pushforward")
    axes[1, 1].set_xlim(0, 100)

    plt.tight_layout()
    out_path = Path("plots/tests/integrated_test_concrete_SCP.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
