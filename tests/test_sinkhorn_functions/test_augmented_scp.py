import numpy as np
import pandas as pd

from src.sinkhorn_functions import augmented_scp


# Put U values on [40, 50], Lambda values on [-10, 10] x [ -10, 20]
def uniform_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform([-10, -10, 40], [10, 20, 50], size=(n, 3))


SEED = 20251028
RNG = np.random.default_rng(seed=SEED)
N_SAMPLES = 100
Q = lambda x, y, z: x + y + z
augmented_Q = lambda x, y, z: np.array([Q(x, y, z), z])
INPUT_GRID_SIZE = 15
FIXED_PARAMS = {
    "q_data": RNG.uniform(-20, 20, size=N_SAMPLES),
    "u_data": RNG.uniform(40, 50, size=N_SAMPLES),
    "J": 1000,
    "M": 100,
    "augmented_Q": augmented_Q,
    "prior": uniform_sampler,
    "grid_mins": np.array([-50, -50, -50]),
    "grid_maxes": np.array([50, 50, 50]),
    "input_grid_size": INPUT_GRID_SIZE,
    "seed": SEED,
}
prob_df, prior_df = augmented_scp(**FIXED_PARAMS)


def test_augmented_SCP_prob_df_cols():
    expected_cols = {"xmin", "ymin", "zmin", "xmax", "ymax", "zmax", "prob"}
    assert isinstance(prob_df, pd.DataFrame)
    assert set(prob_df.columns) == expected_cols
    assert prob_df.shape[0] == INPUT_GRID_SIZE**3


def test_augmented_SCP_prior_df_cols():
    expected_cols = {"source", "x", "y", "z", "q_x", "q_y"}
    assert isinstance(prior_df, pd.DataFrame)
    assert set(prior_df.columns) == expected_cols
    assert (prior_df["source"] == "prior").all()
    assert prior_df.shape[0] == 1000


def test_augmented_SCP_mismatch_length(): ...
