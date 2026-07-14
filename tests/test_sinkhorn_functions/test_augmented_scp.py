"""Unit tests for src.sinkhorn_functions.py::augmented_scp()."""

import numpy as np
import pandas as pd
import pytest

from src.sinkhorn_functions import augmented_scp


def uniform_sampler(rng: np.random.Generator, n: int) -> np.ndarray:
    """Make uniform distribution over the 3D set Lambda x U.

    Uses Lambda x U = [-10, 10] x [-10, 20] x [40, 50].

    Returns
    -------
    (n, 3) numpy array
    """
    return rng.uniform([-10, -10, 40], [10, 20, 50], size=(n, 3))


SEED = 20251028
RNG = np.random.default_rng(seed=SEED)
N_SAMPLES = 100


def Q(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Make simple real-valued map.

    Returns
    -------
    numpy array
    """
    return x + y + z


def augmented_Q(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Make an augmented map from existing Q.

    Returns
    -------
    numpy array
    """
    q = Q(x, y, z)
    return np.vstack([q, z])


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
probs_df, prior_df = augmented_scp(**FIXED_PARAMS)


def test_augmented_SCP_prob_df() -> None:
    """Check columns and rows of prob_df."""
    expected_cols = {"xmin", "ymin", "zmin", "xmax", "ymax", "zmax", "prob"}
    assert isinstance(probs_df, pd.DataFrame)
    assert set(probs_df.columns) == expected_cols
    assert probs_df.shape[0] == INPUT_GRID_SIZE**3

    # Make sure xmin < xmax, ymin < ymax, zmin < zmax
    for axis in ["x", "y", "z"]:
        assert (probs_df[f"{axis}min"] < probs_df[f"{axis}max"]).all()


def test_augmented_SCP_prior_df() -> None:
    """Check columns and rows of prior_df."""
    expected_cols = {"source", "x", "y", "z", "q_x", "q_y"}
    assert isinstance(prior_df, pd.DataFrame)
    assert set(prior_df.columns) == expected_cols
    assert (prior_df["source"] == "prior").all()
    assert prior_df.shape[0] == 1000


def test_augmented_SCP_prior_correct_q_vals() -> None:
    """Check that prior_df correctly computed augmented Q."""
    expected_q_vals = np.array([
        prior_df["x"] + prior_df["y"] + prior_df["z"],
        prior_df["z"],
    ])
    q_vals = np.array([prior_df["q_x"], prior_df["q_y"]])
    assert np.allclose(expected_q_vals, q_vals)


def test_augmented_SCP_mismatch_length() -> None:
    """Both u_data and q_data must be of the same length."""
    bad_fixed_params = {
        **FIXED_PARAMS,
        "u_data": RNG.uniform(40, 50, size=N_SAMPLES - 1),
    }
    with pytest.raises(ValueError, match="mismatched lengths"):
        augmented_scp(**bad_fixed_params)

    bad_fixed_params = {
        **FIXED_PARAMS,
        "q_data": RNG.uniform(-20, 20, size=N_SAMPLES + 1),
    }
    with pytest.raises(ValueError, match="mismatched lengths"):
        augmented_scp(**bad_fixed_params)


def test_augmented_SCP_bad_dimension() -> None:
    """Both u_data and q_data must be (N, 1) or (N, ) arrays."""
    bad_fixed_params = {
        **FIXED_PARAMS,
        "u_data": RNG.uniform(40, 50, size=(N_SAMPLES, 2)),
    }
    with pytest.raises(ValueError, match="Unexpected shape for augmented Q"):
        augmented_scp(**bad_fixed_params)

    bad_fixed_params = {
        **FIXED_PARAMS,
        "q_data": RNG.uniform(-20, 20, size=(N_SAMPLES, 2)),
    }
    with pytest.raises(ValueError, match="Unexpected shape for augmented Q"):
        augmented_scp(**bad_fixed_params)
