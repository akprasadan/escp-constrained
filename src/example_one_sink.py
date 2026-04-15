from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np

from src.example_two import probs_to_mesh
from src.output_functions import apply_map
from src.sinkhorn_functions import (
    augmented_scp,
    run_sinkhorn,
    sample_joint_pairs,
)

plt.rcParams.update({"pdf.fonttype": 42, "text.usetex": True})

SEED = 20251028
LAMBDA_MEAN = [4, 4]
LAMBDA_COV = [[1, 0], [0, 5]]
U_MEAN = -1
U_SD = 5
N_INPUT_SAMPLES = 10000
MC_SAMPLE_SIZE = 5000
JOINT_PAIR_SAMPLE_SIZE = 10000
DPI = 720
SINKHORN_REG = 0.001
LAMBDA_MINS = np.array([-10, -10])
LAMBDA_MAXES = np.array([10, 10])
LAMBDA_U_MINS = np.array([-10, -10, -10])
LAMBDA_U_MAXES = np.array([10, 10, 10])


def Q(x, y, z):
    return x**2 + y**2 + z


def augmented_Q(x, y, z):
    return np.array([Q(x, y, z), z])


if __name__ == "__main__":
    rng = np.random.default_rng(seed=SEED)
    lambda_data = rng.multivariate_normal(
        LAMBDA_MEAN, LAMBDA_COV, N_INPUT_SAMPLES
    )  # (N, 2)
    u_data = rng.normal(U_MEAN, U_SD, N_INPUT_SAMPLES)[:, None]  # (N, 1)
    lambda_u_data = np.hstack((lambda_data, u_data))  # (N, 3)
    q_data = apply_map(lambda_u_data, Q)  # (N, 3) --> (N, 1)

    u_data_new = rng.normal(U_MEAN, U_SD, 9000)[:, None]  # (N, 1)

    def uniform_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(LAMBDA_MINS, LAMBDA_MAXES, size=(n, 2))

    def uniform_normal_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        lambdas = uniform_sampler(rng, n)
        us = rng.normal(U_MEAN, U_SD, n)[:, None]
        return np.hstack((lambdas, us))

    sinkhorn_sol, u_grid, q_grid, empirical_u, empirical_q = run_sinkhorn(
        q_data,
        u_data_new,
        Q,
        prior=uniform_sampler,
        reg_term=SINKHORN_REG,
        lambda_mins=LAMBDA_MINS,
        lambda_maxes=LAMBDA_MAXES,
        MC_sample_size=MC_SAMPLE_SIZE,
        grid_res_U=50,
        grid_res_D=60,
        seed=SEED,
    )

    q_sink_samples, u_sink_samples = sample_joint_pairs(
        sinkhorn_sol,
        u_grid,
        q_grid,
        sample_size=JOINT_PAIR_SAMPLE_SIZE,
        seed=SEED,
    )

    prob_df, prior_df = augmented_scp(
        q_sink_samples,
        u_sink_samples,
        10_000,
        100,
        augmented_Q,
        uniform_normal_sampler,
        LAMBDA_U_MINS,
        LAMBDA_U_MAXES,
        25,
        SEED,
    )
    prob_df_marginalized = prob_df.groupby(
        ["xmin", "xmax", "ymin", "ymax"], as_index=False
    ).agg(prob=("prob", "sum"))

    x_edges, y_edges, Z = probs_to_mesh(prob_df_marginalized)

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    vmax = prob_df_marginalized["prob"].max()
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        Z,
        cmap="viridis",
        norm=norm,
        shading="auto",
        rasterized=True,
    )
    ax.set_xlabel(r"$\lambda_1$", fontsize=14)
    ax.set_ylabel(r"$\lambda_2$", fontsize=14)
    ax.tick_params(labelsize=13)

    lambda_subset = rng.choice(lambda_data, size=1000, replace=False)
    ax.scatter(lambda_subset[:, 0], lambda_subset[:, 1], s=5, color="red")
    ax.set_xlim(LAMBDA_MINS[0], LAMBDA_MAXES[0])
    ax.set_ylim(LAMBDA_MINS[1], LAMBDA_MAXES[1])

    out_path = Path(f"plots/example_sinkhorn.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
