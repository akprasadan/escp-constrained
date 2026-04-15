import numpy as np
import pytest
from src.sinkhorn_functions import run_sinkhorn, sample_joint_pairs

SEED = 20251028
RNG = np.random.default_rng(seed=SEED)
N_Q = 1000
N_U = 600
MC_SAMPLE_SIZE = 200
JOINT_SAMPLE_SIZE = 180


def uniform_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform([-10, 5], [2, 6], size=(n, 2))


fixed_params = {
    "q_data": RNG.uniform(-10, 15, size=N_Q),
    "u_data": RNG.uniform(0, 5, size=(N_U, 1)),  # Intentional reshaping
    "Q": lambda x, y, z: x + 2 * y - z,
    "prior": uniform_sampler,
    "reg_term": 0.01,
    "lambda_mins": np.array([-12, 4]),
    "lambda_maxes": np.array([3, 7]),
    "MC_sample_size": MC_SAMPLE_SIZE,
    "grid_res_U": 40,
    "grid_res_D": 60,
    "seed": SEED,
}
sinkhorn_sol, u_grid, q_grid, empirical_u, empirical_q = run_sinkhorn(
    **fixed_params
)
q_sink_samples, u_sink_samples = sample_joint_pairs(
    sinkhorn_sol, u_grid, q_grid, JOINT_SAMPLE_SIZE, seed=SEED
)


def test_sample_joint_pairs_output_size():
    """Make sure we got two (N, ) shaped arrays of the same length."""
    assert q_sink_samples.shape == (JOINT_SAMPLE_SIZE, 1)
    assert u_sink_samples.shape == (JOINT_SAMPLE_SIZE, 1)


def test_sample_joint_pairs_non_trivial():
    """Make sure we got two (N, ) shaped arrays with all different entries from uniform sampling.
    This is so even if the same index got sampled from the sinkhorn matrix, since we sample
    a random index and then sample within that corresponding tile.
    """
    assert len(np.unique(q_sink_samples)) == JOINT_SAMPLE_SIZE
    assert len(np.unique(u_sink_samples)) == JOINT_SAMPLE_SIZE


def test_sample_joint_pairs_bad_u_grid():
    """Make sure we got two (N, ) shaped arrays of the same length."""
    u_grid_bad = u_grid[:-1]
    assert u_grid_bad.shape == (len(u_grid) - 1, 1)

    q_grid_bad = q_grid[:-1]
    assert q_grid_bad.shape == (len(q_grid) - 1, 1)

    good_arguments = {
        "sinkhorn_sol": sinkhorn_sol,
        "sample_size": JOINT_SAMPLE_SIZE,
        "seed": SEED,
    }

    msg = "Sinkhorn solution or grids of wrong dimensions."
    with pytest.raises(ValueError, match=msg):
        sample_joint_pairs(u_grid=u_grid_bad, q_grid=q_grid, **good_arguments)

    with pytest.raises(ValueError, match=msg):
        sample_joint_pairs(u_grid=u_grid, q_grid=q_grid_bad, **good_arguments)
