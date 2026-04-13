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


if __name__ == "__main__":
    SEED = 20251028
    mean = [2, 2]
    cov = [[1, 0], [0, 10]]
    lambda_data = np.random.multivariate_normal(mean, cov, 100_000)  # 2D
    u_data = np.random.normal(-1, 10, 100_000)
    lambda_u_data = np.hstack((lambda_data, u_data[:, None]))  # 3D
    Q = lambda x, y, z: x**2 + y**2 + z
    augmented_Q = lambda x, y, z: np.array([Q(x, y, z), z])
    q_data = apply_map(lambda_u_data, Q)  # 3D --> 1D

    lambda_mins = lambda_data.min(axis=0)  # (2, 2)
    lambda_maxes = lambda_data.max(axis=0)
    u_mins = np.array([u_data.min(axis=0)])
    u_maxes = np.array([u_data.max(axis=0)])

    u_lambda_mins = np.concatenate([lambda_mins, u_mins])
    u_lambda_maxes = np.concatenate([lambda_maxes, u_maxes])

    def uniform_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(lambda_mins, lambda_maxes, size=(n, 2))

    def uniform_normal_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
        lambdas = uniform_sampler(rng, n)
        us = rng.normal(-1, 10, n)[:, None]
        return np.hstack((lambdas, us))

    sinkhorn_sol, u_grid, q_grid, empirical_u, empirical_q = run_sinkhorn(
        q_data,
        u_data,
        Q,
        prior=uniform_sampler,
        reg_term=0.001,
        lambda_mins=lambda_mins,
        lambda_maxes=lambda_maxes,
        MC_sample_size=100_000,
        grid_res=50,
        seed=SEED,
    )

    q_sink_samples, u_sink_samples = sample_joint_pairs(
        sinkhorn_sol, u_grid, q_grid, sample_size=100_000, seed=SEED
    )

    prob_df, prior_df = augmented_scp(
        q_sink_samples,
        u_sink_samples,
        100_000,
        100,
        augmented_Q,
        uniform_normal_sampler,
        u_lambda_mins,
        u_lambda_maxes,
        50,
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
    ax.set_xlabel(r"$\\lambda_1$", fontsize=14)
    ax.set_ylabel(r"$\\lambda_2$", fontsize=14)
    ax.tick_params(labelsize=13)

    out_path = Path(f"plots/example_sinkhorn.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=720, bbox_inches="tight")
    plt.close(fig)
