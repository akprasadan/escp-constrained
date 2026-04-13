import numpy as np

from src.output_functions import apply_map
from src.sinkhorn_functions import (
    augmented_scp,
    run_sinkhorn,
    sample_joint_pairs,
)


if __name__ == "__main__":
    SEED = 20251028
    mean = [2, 2]
    cov = [[1, 0], [0, 20]]
    lambda_data = np.random.multivariate_normal(mean, cov, 50000)  # 2D
    u_data = np.random.normal(-1, 10, 50000)
    lambda_u_data = np.hstack((lambda_data, u_data[:, None]))
    augmented_Q = lambda x, y, z: np.array([x**2 + y**2 + z, z])
    q_data = apply_map(lambda_u_data, augmented_Q)

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
        augmented_Q,
        prior=uniform_sampler,
        reg_term=0.001,
        lambda_mins=lambda_mins,
        lambda_maxes=lambda_maxes,
        MC_sample_size=10_000,
        grid_res=50,
        seed=SEED,
    )

    q_sink_samples, u_sink_samples = sample_joint_pairs(
        sinkhorn_sol, u_grid, q_grid, sample_size=50_000, seed=SEED
    )

    prob_df, prior_df = augmented_scp(
        q_sink_samples,
        u_sink_samples,
        10_000,
        100,
        augmented_Q,
        uniform_normal_sampler,
        u_lambda_mins,
        u_lambda_maxes,
        50,
        SEED,
    )
