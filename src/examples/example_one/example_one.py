"""
example_one.py.

Perform a synthetic experiment for the constrained SCP.

See tutorial.md for more background and how to make your own example.

Set-up:
-------
- Lambda = [-10, 10]^2
- U = [0, 1]
- Q(x, y, z) = x^2 + y^2 + z
- D = Q(Lambda x U)
- P_{Lambda x U}^t = Beta(2, 8) x Beta(8, 8) x Beta(12, 3) translated/scaled to
  Lambda x U in each dimension
- Prior P_p = Unif[-10, 10]^2 x Beta(12, 3)

Output:
-------
Plots showing:

- The actual joint pushforward on D x U
- The Sinkhorn solution on D x U, along with some samples from it
- The paired constrained SCP solution for Lambda (U marginalized out), along
  with the original P_{Lambda}^t, where we used the actual joint pushforward on
  D x U
- The unpaired constrained SCP solution for Lambda (U marginalized out), along
  with the original P_{Lambda}^t, where we used the Sinkhorn pushforward on D x
  U
- A stacked heatmap that visualizes the unpaired constrained SCP solution for
  Lambda x U, a 4D plot
"""

import numpy as np
from matplotlib import pyplot as plt

from src.examples.data_generation_utilities import (
    Distributions,
    Domain,
    Experiment,
    PlotOptions,
    generate_example_data,
)
from src.examples.example_one.plotting_utilities import create_example_one_plot
from src.sampler_functions import sample_multi_beta_dist, unif_dist_hyperrect

SEED = 20251028
plt.rcParams.update({"pdf.fonttype": 42, "text.usetex": True})


def Q_example_one(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Make simple Q for example one.

    Returns
    -------
    numpy array
    """
    return x**2 + y**2 + z


def main_independent_beta() -> tuple[Domain, Distributions, Experiment, PlotOptions]:
    """Run an experiment with independent Beta TGDs for both Lambda and U.

    Returns
    -------
    tuple[Domain, Distributions, Experiment, PlotOptions]
    """
    lambda_a_vec = [2, 8]
    lambda_b_vec = [8, 8]

    u_a, u_b = 12, 3

    lambda_beta_mins, lambda_beta_maxes = np.array([-8, -8]), np.array([4, 4])
    lambda_mins, lambda_maxes = np.array([-10, -10]), np.array([10, 10])
    U_mins, U_maxes = np.array([0]), np.array([1])

    domain = Domain(lambda_mins, lambda_maxes, U_mins, U_maxes)

    def lambda_tgd(rng: np.random.Generator, J: int) -> np.ndarray:
        return sample_multi_beta_dist(
            lambda_beta_mins,
            lambda_beta_maxes,
            np.array(lambda_a_vec),
            np.array(lambda_b_vec),
            rng,
            J,
        )

    # This MUST match the joint_tgd marginal in U, which it does by construction
    def u_prior(rng: np.random.Generator, J: int) -> np.ndarray:
        return sample_multi_beta_dist(
            U_mins, U_maxes, np.array([u_a]), np.array([u_b]), rng, J
        )

    def joint_tgd(rng: np.random.Generator, J: int) -> np.ndarray:
        lambda_tgd_vals = lambda_tgd(rng, J)
        u_tgd_vals = u_prior(rng, J)
        return np.column_stack((lambda_tgd_vals, u_tgd_vals))

    def lambda_prior(rng: np.random.Generator, J: int) -> np.ndarray:
        return unif_dist_hyperrect(lambda_mins, lambda_maxes, rng, J)

    distributions = Distributions(
        joint_tgd=joint_tgd, lambda_prior=lambda_prior, u_prior=u_prior
    )

    expt = Experiment(
        Q=Q_example_one,
        num_inputs=3_000,
        num_indep_u_inputs=10_000,
        sinkhorn_grid_res_U=30,
        sinkhorn_grid_res_D=30,
        sinkhorn_reg=1,
        monte_carlo_size=15_000,
        num_joint_pairs=100_000,
        SCP_prior_count=250_000,
        SCP_lambda_grid_res=40,
        SCP_D_grid_res=30,
    )

    plot_options = PlotOptions(
        figsize=(6, 6),
        dpi=720,
        scatter_count=150,
        expt_name="quadratic_scp",
        D_min=-1,
        D_max=90,
        folder_name="one",
    )
    return domain, distributions, expt, plot_options


if __name__ == "__main__":
    domain, distributions, expt, plot_options = main_independent_beta()

    # If SCP experiment was already run, set this to be False to avoid
    # re-computing solution and only re-generate the plots.
    generate_data = False
    if generate_data:
        generate_example_data(
            domain, distributions, expt, plot_options, seed=SEED, paired=True
        )

    create_example_one_plot(domain, expt, plot_options)
