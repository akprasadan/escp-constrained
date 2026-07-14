"""
example_unconstrained.py.

Run an experiment like in example_one but do not use the constrained technique.
Instead, assume you 'lost' the U data (as with the Lambda data) and just did a
full 3D -> 1D inversion using the traditional SCP algorithm. We don't use the
augmented Q anymore but just Q itself. This experiment is chosen to demonstrate
that the U marginal is not preserved, in general. Also, we run a Sinkhorn-based
(unpaired) constrained SCP for comparison.

Set-up:
-------
- Take Lambda x U = [-10, 10]^2 x [-20, 20]. Let D = Q(Lambda x U).
- Set Q(x, y, z) = x^2 + y^2 + z^2.
- Take TGD to be an equally-weighted mixture of two Gaussians on Lambda x U,
  with the same diagonal covariances diag(1, 1, 9), and means (3, 3, 6), (-3,
  -3, -3). The U marginal, denoted U_{TGD}, is the mixture of N(6, 9) and N(-3,
  9).
- In both constrained/unconstrained version, use prior being Unif[-10, 10]^2 x
  U_{TGD}.

Procedure
----------
See example_one.py's docstring, which explains the constrained version with
Sinkhorn (i.e., unpaired) as well as the unconstrained version.

Output
------
We make a 2 x 2 grid of plots that shows:

- U marginals (TGD, constrained SCP, unconstrained SCP) to demonstrate that the
  constrained SCP matches the TGD but not the unconstrained one.
- D marginals (TGD pushforward, constrained SCP pushforward, unconstrained SCP
  pushforward). All of them should line up, since an SCP always preserves the
  pushforward in D.
- Unconstrained SCP solution, Lambda marginal only.
- Constrained SCP solution with Sinkhorn, Lambda marginal only
"""

from dataclasses import replace

import numpy as np
from matplotlib import pyplot as plt

from src.examples.data_generation_utilities import (
    Distributions,
    Domain,
    Experiment,
    PlotOptions,
    generate_example_data,
)
from src.examples.example_unconstrained.plotting_utilities import (
    Experiment_Uncon,
    create_uncon_example_plots,
    generate_example_data_uncon,
)
from src.sampler_functions import sample_gaussian_mixture, unif_dist_hyperrect

plt.rcParams.update({"pdf.fonttype": 42, "text.usetex": True})

SEED = 20251028


def Q_example_uncon(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Make simple Q to compare unconstrained/constrained.

    Unlike example one, note that the z term is squared here.

    Returns
    -------
    numpy array
    """
    return x**2 + y**2 + z**2


def main_mixture_gaussian() -> tuple[
    Domain, Distributions, Experiment_Uncon, Experiment, PlotOptions, PlotOptions
]:
    """
    Prepare dataclasses to run both a constrained and unconstrained experiment.

    We use a mixture of two Gaussian TGD on Lambda x U with diagonal covariance
    matrices. Use as prior the independent coupling of the uniform prior on
    Lambda and the actual U TGD on U. Both constrained and unconstrained will
    use this prior, since we wish to help the unconstrained do as best as it can
    while still failing as intended.

    Returns
    -------
    tuple[Domain, Distributions, Experiment_Uncon, Experiment, PlotOptions,
    PlotOptions]
    """
    lambda_mins, lambda_maxes = np.array([-10, -10]), np.array([10, 10])
    U_mins, U_maxes = np.array([-20]), np.array([20])
    domain = Domain(lambda_mins, lambda_maxes, U_mins, U_maxes)

    u_mean_A, u_mean_B = 6, -3
    tgd_mean_A = np.array([3, 3, u_mean_A])
    tgd_mean_B = np.array([-3, -3, u_mean_B])

    u_sd = 3
    tgd_cov_A = np.diag([1, 1, u_sd**2])
    tgd_cov_B = np.diag([1, 1, u_sd**2])

    def joint_tgd(rng: np.random.Generator, J: int) -> np.ndarray:
        return sample_gaussian_mixture(
            tgd_mean_A, tgd_mean_B, tgd_cov_A, tgd_cov_B, 0.5, rng, J
        )

    def lambda_prior(rng: np.random.Generator, J: int) -> np.ndarray:
        return unif_dist_hyperrect(lambda_mins, lambda_maxes, rng, J)

    # This MUST match the joint_tgd marginal in U
    def u_prior(rng: np.random.Generator, J: int) -> np.ndarray:
        return sample_gaussian_mixture(
            tgd_mean_A[-1],
            tgd_mean_B[-1],
            tgd_cov_A[2, 2],
            tgd_cov_B[2, 2],
            0.5,
            rng,
            J,
        )

    distributions = Distributions(
        joint_tgd=joint_tgd, lambda_prior=lambda_prior, u_prior=u_prior
    )
    expt_uncon = Experiment_Uncon(
        Q=Q_example_uncon,
        num_inputs=20_000,
        SCP_prior_count=250_000,
        SCP_lambda_grid_res=30,
        SCP_D_grid_res=50,
    )

    expt_con = Experiment(
        Q=Q_example_uncon,
        num_inputs=20_000,
        num_indep_u_inputs=150_000,
        sinkhorn_grid_res_D=30,
        sinkhorn_grid_res_U=30,
        sinkhorn_reg=10,
        monte_carlo_size=15_000,
        num_joint_pairs=200_000,
        SCP_prior_count=250_000,
        SCP_lambda_grid_res=30,
        SCP_D_grid_res=50,
    )

    plot_options_uncon = PlotOptions(
        figsize=(6, 6),
        dpi=720,
        scatter_count=150,
        expt_name=r"unconstrained_gaussian_mixture",
        D_min=-1,
        D_max=200,
        folder_name="unconstrained",
    )

    # Constrained plot options are identical, including the directory, specified
    # by folder_name, in each of data/ and plots/. However, the experiment name
    # must be different.
    plot_options_con = replace(
        plot_options_uncon, expt_name=r"constrained_gaussian_mixture"
    )

    return (
        domain,
        distributions,
        expt_uncon,
        expt_con,
        plot_options_uncon,
        plot_options_con,
    )


if __name__ == "__main__":
    (
        domain,
        distributions,
        expt_uncon,
        expt_con,
        plot_options_uncon,
        plot_options_con,
    ) = main_mixture_gaussian()

    # If SCP experiment was already run, set this to be False to avoid
    # re-computing solution and only re-generate the plots.
    generate_data = False

    if generate_data:
        # Get unconstrained SCP experimental data.
        generate_example_data_uncon(
            domain, distributions, expt_uncon, plot_options_uncon, seed=SEED
        )
        # Get unpaired, unconstrained SCP experimental data, using Sinkhorn.
        # Skip paired SCP.
        generate_example_data(
            domain, distributions, expt_con, plot_options_con, seed=SEED, paired=False
        )

    create_uncon_example_plots(domain, expt_uncon, plot_options_uncon, plot_options_con)
