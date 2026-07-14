"""
example_two.py.

We perform a constrained SCP experiment using a forward model that is a 2D
thin-plate convection-diffusion heat equation. Specifically, we set Lambda to be
a triplet of 2 convection parameters and a diffusion parameter, U the mean
vertical position of a Gaussian-shaped (in a spatial sense, nor random) source
term, and Q the temperature in steady state at a fixed position. We assume
Dirichlet conditions of 0 on three of the sides, but the top edge has a
non-homogeneous temperature function constraint (based on a Beta distribution).
We actually use a surrogate model (based on an XGBoost fit) rather than the
direct PDE solver to speed up computations. However, the initial generation of
synthetic observations uses the actual heat equation solver. In our setting, it
really makes no difference due to the ability to achieve near perfect
interpolation accuracy of the XGBoost model thanks to cheap computation.

Before running this file, make sure "data/example_two/xgb_model.json" already
exists, or create it by first running q_computations.py (warning: slow, 10-30
minutes) and then fit_xgb.py in the src/examples/example_two/ directory. This
file can also be slow to run, say ~10 minutes.

Unlike example one, this example has a 3D Lambda. See the tutorial.md file to
design your own such example, not necessarily using a heat equation, which
forces more preparation/processing steps. Just replace the Q with your simpler
version and otherwise call the same utility functions and use the same
dataclasses.

Set-up:
-------
- Lambda = [0, 3]^2 x [0.1, 1.5], i.e., 2 convection parameters + 1 diffusion
  parameter
- U = [0.2, 0.8], the mean vertical position of a Gaussian source term
- Heat Equation: 2D thin-plate convection-diffusion heat eq. in steady state at
  (0.5, 0.5) on unit square
- Dirichlet conditions: 0 on left/bottom/right; on top, x(1-x)^4 scaled so max
  height on [0, 1] is 1.
- Heat source term of 3 * exp(-(x-0.5)^2 / 0.1^2 + (y - u)^2 / 0.05^2), where u
  is the random parameter.
- Q = XGBoost prediction trained on heat equation evaluations
- D = Q(Lambda x U)
- TGD P_{Lambda x U}^t on Lambda x U = Beta(2, 6) x Beta(2, 2) x Beta(6, 2) x
  Beta(2, 10), translated/scaled to Lambda x U in each dimension. For example,
  if one axis is the interval [a, b], and the distribution to be scaled is
  Beta(x, y), then the marginal TGD is (b - a)*Beta(x, y) + a.
- Prior P_p = Unif[Lambda] x Beta(2, 10), translated/scaled to U

Output:
-------
- A 2 x 3 set of plots containing the unpaired and paired scenario constrained
  SCP solution marginalized over each possible pairing of Lambda dimensions,
  i.e., 3 different plots per type of SCP method. A subset of TGD points are
  included (projected onto each corresponding pair of Lambda axes).
- The D and U marginals (from TGD, Sinkhorn, unpaired SCP solution, and paired
  SCP solution), the D x U true joint distribution (TGD pushforward), and the D
  x U joint Sinkhorn estimate. Either joint plot shows samples from the
  corresponding distribution.
- A 2 x 1 plot showing the marginal 3D paired and unpaired constrained SCPs on
  Lambda (not U) as a stacked heatmap
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from xgboost import XGBRegressor

from src.examples.data_generation_utilities import (
    Distributions,
    Domain,
    Experiment,
    PlotOptions,
    generate_example_data,
)
from src.examples.example_two.plotting_utilities import create_example_two_plot
from src.examples.example_two.q_computations import heat_equation_w_source_at_pos
from src.sampler_functions import sample_multi_beta_dist, unif_dist_hyperrect

plt.rcParams.update({"pdf.fonttype": 42, "text.usetex": True})

SEED = 20251028


q_model = XGBRegressor()

q_model_path = Path("data/example_two/xgb_model.json")
if not q_model_path.exists():
    raise FileNotFoundError(
        "Missing XGBoost model fit. Make sure to run fit_xgb.py first."
    )

q_model.load_model(q_model_path)


def heat_eq_xgb_predict(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Set Q to be XGB prediction at this param combo.

    Returns
    -------
    numpy array
    """
    return q_model.predict(np.column_stack((x, y, z, w)))


def main_heat_eq_with_source() -> tuple[Domain, Distributions, Experiment, PlotOptions]:
    """Prepare example two's heat equation experiment.

    Returns
    -------
    tuple[Domain, Distributions, Experiment, PlotOptions]
    """
    # Define the domain of Lambda and U, inside of which our TGD must have
    # support
    lambda_mins, lambda_maxes = np.array([0, 0, 0.1]), np.array([3, 3, 1.5])
    u_mins, u_maxes = np.array([0.2]), np.array([0.8])

    # Define (alpha, beta) parameters in a Beta distribution we place over
    # Lambda
    lambda_a_vec = [2, 2, 6]  # alpha
    lambda_b_vec = [6, 2, 2]  # beta
    u_a, u_b = 2, 10

    # Define support of TGD on Lambda, a subset of Lambda itself The support of
    # the U TGD will already be given by that of U above (since it is known)
    lambda_tgd_mins, lambda_tgd_maxes = (
        np.array([0.25, 1, 0.5]),
        np.array([1.5, 2, 0.9]),
    )
    sinkhorn_reg = 1
    domain = Domain(lambda_mins, lambda_maxes, u_mins, u_maxes)

    def lambda_tgd(rng: np.random.Generator, J: int) -> np.ndarray:
        return sample_multi_beta_dist(
            lambda_tgd_mins,
            lambda_tgd_maxes,
            np.array(lambda_a_vec),
            np.array(lambda_b_vec),
            rng,
            J,
        )

    # This must match the joint_tgd marginal in U!
    def u_prior(rng: np.random.Generator, J: int) -> np.ndarray:
        return sample_multi_beta_dist(
            u_mins, u_maxes, np.array([u_a]), np.array([u_b]), rng, J
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
        Q=heat_eq_xgb_predict,
        non_surrogate_Q=heat_equation_w_source_at_pos,  # type: ignore
        num_inputs=25_000,
        num_indep_u_inputs=25_000,
        sinkhorn_grid_res_U=30,
        sinkhorn_grid_res_D=30,
        sinkhorn_reg=sinkhorn_reg,
        monte_carlo_size=15_000,
        num_joint_pairs=100_000,
        SCP_prior_count=300_000,
        SCP_lambda_grid_res=25,
        SCP_D_grid_res=20,
    )

    plot_options = PlotOptions(
        figsize=(6, 6),
        dpi=720,
        scatter_count=150,
        expt_name="heat_eq_scp",
        D_min=0.1,
        D_max=0.4,
        folder_name="two",
    )
    return domain, distributions, expt, plot_options


if __name__ == "__main__":
    domain, distributions, expt, plot_options = main_heat_eq_with_source()

    # If SCP experiment was already run, set this to be False to avoid
    # re-computing solution and only re-generate the plots.
    generate_data = False

    if generate_data:
        generate_example_data(
            domain, distributions, expt, plot_options, SEED, paired=True
        )

    create_example_two_plot(domain, expt, plot_options)
