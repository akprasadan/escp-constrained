import numpy as np
import pytest
from src.sinkhorn_functions import run_sinkhorn

SEED = 20251028
RNG = np.random.default_rng(seed=SEED)
N_Q = 1000
N_U = 600
MC_SAMPLE_SIZE = 200


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


def test_run_sinkhorn_output_shapes():
    """Check shapes of each of the outputs of the sinkhorn estimation."""

    assert u_grid.shape == (41, 1)
    assert q_grid.shape == (61, 1)
    assert sinkhorn_sol.shape == (40, 60)
    assert empirical_u.shape == (40,)
    assert empirical_q.shape == (60,)


def test_run_sinkhorn_empirical_vectors():
    """Check empirical vectors are indeed probability vector."""
    assert np.allclose(empirical_u.sum(), 1)
    assert np.allclose(empirical_q.sum(), 1)

    assert np.all(empirical_u >= 0)
    assert np.all(empirical_q >= 0)


def test_run_sinkhorn_solution():
    """Check Sinkhorn solution is a probability matrix."""
    assert np.allclose(sinkhorn_sol.sum(), 1)
    assert np.all(sinkhorn_sol >= 0)


def test_run_sinkhorn_bad_reg():
    """Zero or negative Sinkhorn regularization terms not allowed."""
    bad_fixed_params = {**fixed_params, "reg_term": -0.01}
    with pytest.raises(ValueError, match="regularization term"):
        run_sinkhorn(**bad_fixed_params)

    bad_fixed_params = {**fixed_params, "reg_term": 0}
    with pytest.raises(ValueError, match="regularization term"):
        run_sinkhorn(**bad_fixed_params)


def test_run_sinkhorn_q_or_U_dim_too_high():
    """Only 1D data allowed."""
    bad_q_data = RNG.uniform(-10, 15, size=(N_Q, 2))
    bad_fixed_params = {**fixed_params, "q_data": bad_q_data}
    with pytest.raises(ValueError, match="U and D data must be real-valued"):
        run_sinkhorn(**bad_fixed_params)

    bad_u_data = RNG.uniform(0, 5, size=(N_U, 2))
    bad_fixed_params = {**fixed_params, "u_data": bad_u_data}
    with pytest.raises(ValueError, match="U and D data must be real-valued"):
        run_sinkhorn(**bad_fixed_params)


def test_run_sinkhorn_bad_lambda_bounds():
    """
    Check that bad lambda_mins and lambda_maxes output gets caught.
    """

    bad_lambda_mins, bad_lambda_maxes = np.array([-1, -5]), np.array([0, 1, 2])

    bad_fixed_params = {
        **fixed_params,
        "lambda_mins": bad_lambda_mins,
        "lambda_maxes": bad_lambda_maxes,
    }
    with pytest.raises(ValueError, match="Incompatible lengths"):
        run_sinkhorn(**bad_fixed_params)

    bad_lambda_mins, bad_lambda_maxes = np.array([-10, -5]), np.array([0, -6])
    bad_fixed_params = {
        **fixed_params,
        "lambda_mins": bad_lambda_mins,
        "lambda_maxes": bad_lambda_maxes,
    }
    with pytest.raises(ValueError, match="grid_mins >= grid_maxes"):
        run_sinkhorn(**bad_fixed_params)


def test_run_sinkhorn_reshaped_U():
    """Passing a (N, ) array for U or D data instead of (N, 1) gives same results."""

    old_u_data = fixed_params["u_data"]
    reshaped_u_data = old_u_data.flatten()

    # Make sure this test actually does something
    assert old_u_data.ndim == 2
    assert reshaped_u_data.ndim == 1

    new_fixed_params = {**fixed_params, "u_data": reshaped_u_data}
    new_sol, new_u_grid, new_q_grid, new_emp_u, new_emp_q = run_sinkhorn(
        **new_fixed_params
    )
    assert np.allclose(sinkhorn_sol, new_sol)
    assert np.allclose(u_grid, new_u_grid)
    assert np.allclose(q_grid, new_q_grid)
    assert np.allclose(empirical_u, new_emp_u)
    assert np.allclose(empirical_q, new_emp_q)
