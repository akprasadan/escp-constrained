"""
data_generation_utilities.py.

Utility functions and dataclasses to run the examples,
src/examples/example_one/example_one.py, examples/example_two/example_two.py.
"""

from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from src.output_functions import apply_map
from src.sampler_functions import sample_distr
from src.sinkhorn_functions import augmented_scp, run_sinkhorn, sample_joint_pairs


@dataclass
class Domain:
    """
    Characteristics of the Lambda x U space.

    Attributes
    ----------
    lambda_mins, lambda_maxes : (dim(Lambda), ) numpy array
        A vector of minimums or maximums defining the boundaries for Lambda
    U_mins, U_maxes : (dim(U), ) numpy array
        A vector of minimums or maximums defining the boundaries for U
    input_mins, input_maxes : (dim(Lambda) + dim(U), ) numpy array
        A vector of minimums or maximums defining the boundaries for Lambda x U
        (property).
    """

    lambda_mins: np.ndarray
    lambda_maxes: np.ndarray
    U_mins: np.ndarray
    U_maxes: np.ndarray

    @property
    def input_mins(self) -> np.ndarray:
        """Get a vector of minimums defining the boundaries for Lambda x U."""
        return np.concatenate([self.lambda_mins, self.U_mins])

    @property
    def input_maxes(self) -> np.ndarray:
        """Get a vector of maximums defining the boundaries for Lambda x U."""
        return np.concatenate([self.lambda_maxes, self.U_maxes])


@dataclass
class Distributions:
    """
    Characteristics of the TGD and prior on Lambda and U.

    Attributes
    ----------
    joint_tgd : Callable[[np.random.Generator, int], np.ndarray]
        A TGD on Lambda x U.
    lambda_prior : Callable[[np.random.Generator, int], np.ndarray]
        A prior distribution on Lambda
    u_prior: Callable[[np.random.Generator, int], np.ndarray]
        A prior on U. It must match the TGD marginal for U.

    Methods
    -------
    joint_prior : Callable[[np.random.Generator, int], np.ndarray]
        The independent coupling of lambda_prior and u_prior.

    Notes
    -----
    If you wish to use the independent coupling as the TGD, the following
    workflow is recommended:

    First define functions `lambda_tgd` and `u_prior`:

    >>> def lambda_tgd(rng, J) -> np.ndarray: ...

    >>> def u_prior(rng, J) -> np.ndarray: ...

    Then stack the outputs together:

    >>> def joint_tgd(rng, J) -> np.ndarray:
    ...     lambda_tgd_vals = lambda_tgd(rng, J)
    ...     u_prior_vals = u_prior(rng, J)
    ...     return np.column_stack((lambda_tgd_vals, u_prior_vals))

    Warning
    -------
    You must make sure the U prior is specified to match the joint TGD marginal,
    otherwise the example code will draw from the wrong distribution as the U
    TGD.
    """

    joint_tgd: Callable[[np.random.Generator, int], np.ndarray]
    lambda_prior: Callable[[np.random.Generator, int], np.ndarray]
    u_prior: Callable[[np.random.Generator, int], np.ndarray]

    @property
    def joint_prior(self) -> Callable[[np.random.Generator, int], np.ndarray]:
        """
        Prior distribution over Lambda x U.

        Prior is formed by independent coupling of lambda_prior and the u_prior
        (which is just the u_tgd in general).

        Returns
        -------
        Callable[[np.random.Generator, int], np.ndarray]
            Prior sampler on Lambda x U.
        """

        def _prior(rng: np.random.Generator, J: int) -> np.ndarray:
            lambda_sample = self.lambda_prior(rng, J)
            u_sample = self.u_prior(rng, J)
            return np.column_stack((lambda_sample, u_sample))

        return _prior


@dataclass
class Experiment:
    """
    Dataclass for experimental hyper-parameters for the constrained SCP.

    Specifies settings for both paired or unpaired (D, U) scenarios, that are
    not already specified by the Domain and Distribution dataclasses.

    Attributes
    ----------
    Q : Callable[..., np.ndarray]
        Map from Lambda x U --> D. Not the augmented version.
    num_inputs : int
        The number of (lambda_i, u_i) pairs to draw in order to generate the
        observed q_i
    num_indep_u_inputs : int
        Number of additional independent copies of u_j used in Sinkhorn
        estimation
    sinkhorn_grid_res_U, sinkhorn_grid_res_D : int
        The number of bins to use splitting up D and U in Sinkhorn estimation
    sinkhorn_reg : float
        Regularization parameter used in Sinkhorn estimation
    monte_carlo_size : int
        Number of lambda_i to draw when estimating cost function, which
        integrates over Lambda.
    num_joint_pairs : int
        Number of (q_i, u_i) pairs to draw from Sinkhorn solution
    SCP_prior_count : int
        Number of prior points to draw from Lambda x U in SCP algorithm, for
        reweighing via importance sampling.
    SCP_lambda_grid_res : int
        The number of bins to split each axis of Lambda x U into in the SCP
        algorithm. Scales exponentially in the dimension of Lambda x U.
    SCP_D_grid_res : int
        The number of bins to split each axis of D x U into in the SCP
        algorithm.
    """

    Q: Callable[..., np.ndarray]
    num_inputs: int
    num_indep_u_inputs: int
    sinkhorn_grid_res_U: int
    sinkhorn_grid_res_D: int
    sinkhorn_reg: float
    monte_carlo_size: int
    num_joint_pairs: int
    SCP_prior_count: int
    SCP_lambda_grid_res: int
    SCP_D_grid_res: int
    non_surrogate_Q: Callable[..., np.ndarray] | None = None

    def augmented_Q(self, *args: np.ndarray) -> np.ndarray:
        """
        Augmented version of Q.

        This uses the vectorized version of apply_map().

        Returns
        -------
        numpy array
        """
        q = self.Q(*args)
        u = np.atleast_1d(args[-1])
        return np.vstack((q, u))


@dataclass
class PlotOptions:
    """
    Dataclass for plot and file-naming specifications.

    Attributes
    ----------
    figsize : tuple[int, int]
        Size of Matplotlib figure
    dpi : int
        DPI
    scatter_count : int
        Number of points to use from the TGD or Sinkhorn solution for a
        scatterplot.
    expt_name : str
        Name used to distinguish experiments in filenames for saved objects.
        Usually contains Sinkhorn regularizing parameter or mean/SD parameters
        for distribution on U.
    D_min, D_max : float
        Lower and upper axis limit for D used in two subplots.
    folder_name : str
        The label of the example to set the directory name. So for example, if
        you use "two" as folder_name, you will store datasets in
        data/example_two/ and plots in plots/example_two/. The filename itself,
        however, is specified by expt_name.
    """

    figsize: tuple[int, int]
    dpi: int
    scatter_count: int
    expt_name: str
    D_min: float
    D_max: float
    folder_name: str


def generate_example_data(
    domain: Domain,
    distributions: Distributions,
    expt: Experiment,
    plot_options: PlotOptions,
    seed: int,
    paired: bool,
) -> None:
    """
    Generate data for the examples.

    First, generate synthetic data {q_i} from evaluations of Q(lambda_i, u_i),
    then with an independent sample {u_j}, fit a joint distribution on D x U
    using Sinkhorn algorithm. Then samples (q_i, u_i) from this joint
    distribution and feeds it into the SCP algorithm. Simultaneously, it uses
    the actual paired data (q_i, u_i) that we initially discarded to run the
    SCP, without use of Sinkhorn. Various data quantities are then saved for
    plotting, stored in the data/ folder.

    Warning (to developers)
    -----------------------
    Carefully note the use of different seeds everywhere intentionally.

    Parameters
    ----------
    domain : Domain
        Dataclass for Lambda and U specifications
    distributions : Distribution
        Dataclass for TGD and prior specifications.
    expt : Experiment
        Dataclass for experimental hyper-parameters for Sinkhorn estimation and
        constrained SCP.
    plot_options : PlotOptions
        Dataclass for plot and file-naming specifications.
    seed : int
        For reproducibility.
    paired : bool
        Whether to run the paired constrained SCP experiment too
    """

    def next_seed(_rng: np.random.Generator) -> int:
        """Use a RNG to generate another new seed.

        Returns
        -------
        int
        """
        return int(_rng.integers(0, 2**63 - 1))

    def reg_to_seed(reg: float) -> int:
        """Generate a seed from a Sinkhorn regularization term.

        Returns
        -------
        int
        """
        return round(1e3 * np.log10(reg) + 5000)

    rng = np.random.default_rng(seed + reg_to_seed(expt.sinkhorn_reg))
    tgd_seed = next_seed(rng)
    u_ind_seed = next_seed(rng)
    sinkhorn_seed = next_seed(rng)
    joint_sampler_seed = next_seed(rng)
    aug_scp_seed = next_seed(rng)
    aug_scp_true_joint_seed = next_seed(rng)

    print("Beginning unpaired constrained SCP:")

    joint_tgd_vals = sample_distr(expt.num_inputs, tgd_seed, distributions.joint_tgd)

    lambda_tgd_vals = joint_tgd_vals[:, :-1]
    u_tgd_vals = joint_tgd_vals[:, -1]

    # If you are using a surrogate Q, use the NON-surrogate Q for the initial
    # data generation. We don't use the apply_map in that case in order to
    # enable multi-threading.
    if expt.non_surrogate_Q:
        # Multi-threading enables speed-up of one order of magnitude.
        with Pool() as p:
            # WARNING: check order of arguments to non_surrogate_Q (the heat
            # equation function) It must line up with the joint_tgd values
            # defined a few lines above
            q_vals = p.starmap(expt.non_surrogate_Q, joint_tgd_vals, chunksize=1000)
            # Convert (N, ) array to (N, 1) for consistency
            q_vals = np.array(q_vals)[:, None]
    else:
        q_vals = apply_map(joint_tgd_vals, expt.Q)

    q_type_str = " using non-surrogate Q" if expt.non_surrogate_Q else ""
    print(f"1. Synthetically generating {len(q_vals):,} obs.{q_type_str} in D.")

    # Note u_prior matches the U marginal of the joint_tgd
    print(f"2. Resampling {expt.num_indep_u_inputs:,} indep. copies from U prior.")
    u_indep_vals = sample_distr(
        expt.num_indep_u_inputs, u_ind_seed, distributions.u_prior
    )

    print(
        f"3. Fitting Sinkhorn w/ {len(q_vals):,} D pts. and {len(u_indep_vals):,} U pts."
    )

    sinkhorn_sol, u_grid, q_grid, _, _ = run_sinkhorn(
        q_vals,
        u_indep_vals,  # independent u_j, NOT u_tgd_vals
        expt.Q,  # unaugmented
        distributions.lambda_prior,
        expt.sinkhorn_reg,
        domain.lambda_mins,
        domain.lambda_maxes,
        expt.monte_carlo_size,
        expt.sinkhorn_grid_res_U,
        expt.sinkhorn_grid_res_D,
        sinkhorn_seed,
    )
    print(
        f"4. Resampling {expt.num_joint_pairs:,} points from Sinkhorn solution on D x U."
    )

    q_sinkhorn, u_sinkhorn = sample_joint_pairs(
        sinkhorn_sol,
        u_grid,
        q_grid,
        sample_size=expt.num_joint_pairs,
        seed=joint_sampler_seed,
    )

    print(
        f"5. Running augmented SCP w/ {len(q_sinkhorn):,} obs., and resampling {expt.SCP_prior_count:,} prior samples."
    )

    unpaired_prob_df, _ = augmented_scp(
        q_sinkhorn,
        u_sinkhorn,
        expt.SCP_prior_count,
        expt.SCP_D_grid_res,
        expt.augmented_Q,
        distributions.joint_prior,
        domain.input_mins,
        domain.input_maxes,
        expt.SCP_lambda_grid_res,
        aug_scp_seed,
    )

    data_dir = Path(f"data/example_{plot_options.folder_name}")
    data_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        data_dir / f"{plot_options.expt_name}_arrays.npz",
        lambda_tgd_vals=lambda_tgd_vals,
        u_tgd_vals=u_tgd_vals,
        q_vals=q_vals,
        u_grid=u_grid,
        q_grid=q_grid,
        sinkhorn_sol=sinkhorn_sol,
        q_sinkhorn=q_sinkhorn,
        u_sinkhorn=u_sinkhorn,
    )

    unpaired_prob_df.to_csv(
        data_dir / f"unpaired_{plot_options.expt_name}_prob_df.csv", index=False
    )

    if paired:
        print("Beginning paired constrained SCP:")
        print(
            f"1. Running augmented SCP w/ {len(q_vals):,} obs., and resampling {expt.SCP_prior_count:,} prior samples."
        )
        paired_prob_df, _ = augmented_scp(
            q_vals,
            u_tgd_vals,  # actual u_i from TGD that generated q_i
            expt.SCP_prior_count,
            expt.SCP_D_grid_res,
            expt.augmented_Q,
            distributions.joint_prior,
            domain.input_mins,
            domain.input_maxes,
            expt.SCP_lambda_grid_res,
            aug_scp_true_joint_seed,
        )

        paired_prob_df.to_csv(
            data_dir / f"paired_{plot_options.expt_name}_prob_df.csv", index=False
        )


def create_histogram_from_SCP_U_marginal(
    marginal_prob_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make histogram data from a marginal SCP dataframe.

    Given a dataframe with rows (zmin, zmax, prob), where z corresponds to the U
    axis from an SCP solution, compute the u_values and associated scaled
    probabilities to create a histogram. U may also correspond to "w" if Lambda
    is 3D.

    Parameters
    ----------
    marginal_prob_df : pd.DataFrame
        A dataframe with columns zmin, zmax, and prob (possibly w in place of
        z).

    Returns
    -------
    u_bin_edges : numpy array
        Sorted array of U bin edges defined by the SCP dataframe.
    probs : numpy array
        The corresponding marginal probability of the corresponding interval
        specified by each bin in the u_s, but scaled to match that of a
        Matplotlib histogram with density = True.
    """
    # Find the min/max columns corresponding to U
    U_min_column = next(col for col in marginal_prob_df.columns if col.endswith("min"))
    U_max_column = U_min_column.replace("min", "max")

    # Sort the dataframe by the min value Extract both positions and associated
    # probabilities
    sorted_U_df = marginal_prob_df.sort_values([U_min_column])
    u_bin_edges = sorted_U_df[U_min_column].to_numpy()
    # Add on the last max_value so you get all bin edges
    u_bin_edges = np.append(u_bin_edges, sorted_U_df[U_max_column].iloc[-1])

    # Get probabilities probs, then scale probs to match histogram. We must find
    # p_i = f(probs) such that sum_i (delta_x) * p_i = 1. Thus, set p_i = probs
    # / delta_x.
    probs = sorted_U_df.prob.to_numpy()
    delta_u = np.diff(u_bin_edges).mean()  # All identical differences
    probs = probs / delta_u

    return u_bin_edges, probs
