"""
plotting_utilities.py.

Helper functions that create different plots, including U marginals, D
marginals, the joint distributions on D x U from the TGD pushforward or
Sinkhorn, the 2D Marginal SCP solutions in Lambda only, and 3D SCP solutions
either in Lambda if it is 3D, or Lambda x U if Lambda is 2D.

The example-specific plotting helpers are in the respective subfolder in a
module again titled `plotting_utilities.py`. For example, the function
`create_example_one_plot()` in `examples/example_one/plotting_utilities.py` puts
all of the generic functions in this module together to generate the plots for
example one.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

from src.data_processing import marginalize_out_lambda, probs_to_mesh
from src.examples.data_generation_utilities import (
    Domain,
    Experiment,
    PlotOptions,
    create_histogram_from_SCP_U_marginal,
)
from src.output_functions import apply_map
from src.sampler_functions import sample_from_SCP

SEED = 20251028
NP_HIST_OPTIONS = {"bins": 50, "density": True}
PLT_HIST_OPTIONS = {"bins": 50, "density": True, "histtype": "step"}
MESH_OPTIONS = {"shading": "auto", "rasterized": True}
SCATTERPLOT_OPTIONS = {"s": 8, "alpha": 0.9, "edgecolors": "black", "linewidths": 0.1}


def create_U_marginal_plots(
    ax: Axes,
    sinkhorn_vals: np.ndarray,
    tgd_vals: np.ndarray,
    unpaired_scp_df: pd.DataFrame,
    paired_scp_df: pd.DataFrame,
) -> None:
    """
    Plot U marginals from an SCP experiment.

    Adds to a plot a histogram or step plot of the marginal U sinkhorn values,
    the U TGD values, and both marginal SCP solutions for U, paired and
    unpaired.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis
    sinkhorn_vals : (N_1, ) numpy array
        One-dimensional datapoints on U from a Sinkhorn solution. All values
        will be used for the plot.
    tgd_vals : (N_2, ) numpy array
        One-dimensional U values from the marginal TGD. All values will be used
        for the plot.
    unpaired_scp_df : pd.DataFrame
        Unpaired, constrained SCP solution that uses Sinkhorn
    paired_scp_df : pd.DataFrame
        Paired, constrained SCP solution
    """
    unpaired_scp_u_marginal_df = marginalize_out_lambda(unpaired_scp_df)
    paired_scp_u_marginal_df = marginalize_out_lambda(paired_scp_df)

    tgd_counts, tgd_edges = np.histogram(tgd_vals, **NP_HIST_OPTIONS)
    ax.stairs(values=tgd_counts, edges=tgd_edges, label="TGD")

    sink_counts, sink_edges = np.histogram(sinkhorn_vals, **NP_HIST_OPTIONS)
    ax.stairs(values=sink_counts, edges=sink_edges, label="Sinkhorn Sol.")

    u_bin_edges, probs = create_histogram_from_SCP_U_marginal(paired_scp_u_marginal_df)
    ax.stairs(values=probs, edges=u_bin_edges, label="Paired")

    u_bin_edges, probs = create_histogram_from_SCP_U_marginal(
        unpaired_scp_u_marginal_df
    )
    ax.stairs(values=probs, edges=u_bin_edges, label="Unpaired")

    ax.set_yticks([])
    ax.set_xlabel(r"$U$", fontsize=13)
    ax.set_title(r"$U$ Marginals from TGD, Sinkhorn, and Marginalized SCP Solutions")

    custom_lines = [
        Line2D([0], [0], color="C0", lw=2),
        Line2D([0], [0], color="C1", lw=2),
        Line2D([0], [0], color="C2", lw=2),
        Line2D([0], [0], color="C3", lw=2),
    ]

    _, hist_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=custom_lines,
        labels=hist_labels,
        loc="upper left",
        handlelength=1,
        fontsize=9,
    )


def create_D_marginal_plots(
    ax: Axes,
    sinkhorn_vals: np.ndarray,
    tgd_pushforward_vals: np.ndarray,
    unpaired_scp_df: pd.DataFrame,
    paired_scp_df: pd.DataFrame,
    Q: Callable[..., np.ndarray],
    D_min: float,
    D_max: float,
) -> None:
    """
    Plot D marginals from an SCP experiment.

    Adds a step plot of the marginal D Sinkhorn values, TGD pushforward values,
    and pushforwards of both SCPs, paired and unpaired.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis
    sinkhorn_vals : (N_1, ) numpy array
        One-dimensional D values from a Sinkhorn solution. All values will be
        used for the plot.
    tgd_pushforward_vals : (N_2, ) numpy array
        One-dimensional D values from the marginal TGD. All values will be used
        for the plot.
    unpaired_scp_df : pd.DataFrame
        Unpaired, constrained SCP solution.
    paired_scp_df : pd.DataFrame
        Paired, constrained SCP solution.
    Q : Callable[..., np.ndarray]
        The non-augmented map on Lambda x U.
    D_min, D_max : float
        x-axis limits
    """
    ax.hist(tgd_pushforward_vals, **PLT_HIST_OPTIONS, label="TGD Push.")

    ax.hist(sinkhorn_vals, **PLT_HIST_OPTIONS, label="Sinkhorn Sol.")

    input_samples = np.column_stack(sample_from_SCP(paired_scp_df, 2000, SEED))
    pushforward_sample = apply_map(input_samples, Q)
    ax.hist(pushforward_sample, **PLT_HIST_OPTIONS, label="Paired Push.")

    input_samples = np.column_stack(sample_from_SCP(unpaired_scp_df, 2000, SEED))
    pushforward_sample = apply_map(input_samples, Q)
    ax.hist(pushforward_sample, **PLT_HIST_OPTIONS, label="Unpaired Push.")

    custom_lines = [
        Line2D([0], [0], color="C0", lw=2),
        Line2D([0], [0], color="C1", lw=2),
        Line2D([0], [0], color="C2", lw=2),
        Line2D([0], [0], color="C3", lw=2),
    ]

    _, hist_labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=custom_lines,
        labels=hist_labels,
        loc="upper right",
        handlelength=1,
        fontsize=9,
    )

    ax.set_xlim(D_min, D_max)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_yticks([])
    ax.set_xlabel(r"$\mathcal{D}$", fontsize=13)
    ax.set_title(r"$\mathcal{D}$ Marginals from TGD, Sinkhorn, and SCP Pushforwards")


def D_U_plot_styling(
    ax: Axes, U_min: float, U_max: float, D_min: float, D_max: float
) -> None:
    """Add some common elements to the (D, U) joint plots.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis
    U_min, U_max : float
        The y-limits of the U by D plot
    D_min, D_max : float
        The x-limits of the U by D plot
    """
    ax.set_ylabel(r"$U$", fontsize=13)
    ax.set_xlabel(r"$\mathcal{D}$", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_xlim(D_min, D_max)
    ax.set_ylim(U_min, U_max)
    ax.set_facecolor("#00204C")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))


def create_true_pushforward_plot(
    ax: Axes,
    rng: np.random.Generator,
    q_vals: np.ndarray,
    u_tgd_vals: np.ndarray,
    domain: Domain,
    expt: Experiment,
    plot_options: PlotOptions,
) -> None:
    """
    Visualize true joint pushforward on D x U, along with sample of points.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis
    rng : np.random.Generator
        To pick a subset of the TGD and q vals to compute pushforwards of.
    q_vals : (N, 1) array
        An array of values in D, paired with the associated U values from the
        TGD
    u_tgd_vals : (N, 1) array
        An array of values in U, paired with the q_vals
    domain : Domain
        Dataclass for Lambda and U specifications
    expt : Experiment
        Dataclass for experimental hyper-parameters for Sinkhorn and SCP
    plot_options : PlotOptions
        Dataclass for plot and file-naming specifications
    """
    D_U_hist, d_edges, u_edges = np.histogram2d(
        q_vals.flatten(),
        u_tgd_vals.flatten(),
        bins=[expt.sinkhorn_grid_res_D, expt.sinkhorn_grid_res_U],
        density=True,
    )
    D_U_norm = PowerNorm(gamma=1, vmin=0, vmax=D_U_hist.max())
    ax.pcolormesh(
        d_edges, u_edges, D_U_hist.T, cmap="cividis", norm=D_U_norm, **MESH_OPTIONS
    )
    # we use a subset of the (q, u) joint pairs for plotting purposes
    true_pushforward_sample = rng.choice(
        expt.num_inputs, size=plot_options.scatter_count, replace=False
    )
    u_tgd_subset = u_tgd_vals[true_pushforward_sample]
    q_vals_subset = q_vals[true_pushforward_sample]
    ax.scatter(x=q_vals_subset, y=u_tgd_subset, color="lavender", **SCATTERPLOT_OPTIONS)
    D_U_plot_styling(
        ax,
        float(domain.U_mins[0]),
        float(domain.U_maxes[0]),
        plot_options.D_min,
        plot_options.D_max,
    )


def create_D_U_sinkhorn_plot(
    ax: Axes,
    rng: np.random.Generator,
    q_grid: np.ndarray,
    u_grid: np.ndarray,
    sinkhorn_sol: np.ndarray,
    q_sinkhorn_subset: np.ndarray,
    u_sinkhorn_subset: np.ndarray,
    domain: Domain,
    plot_options: PlotOptions,
) -> None:
    """Create plot of the Sinkhorn solution and some samples from it.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis
    rng : np.random.Generator
        To pick a subset of the TGD and q vals to compute pushforwards of.
    u_grid, q_grid : (grid_res_U + 1, ) or (grid_res_D + 1, ) numpy array
        Bin edges used in the Sinkhorn algorithm for U and D.
    sinkhorn_sol : (grid_res_U, grid_res_D) numpy array
        Sinkhorn estimate, where grid_res_{U, D} were chosen in Sinkhorn
        algorithm.
    q_sinkhorn_subset, u_sinkhorn_subset : numpy array
        Paired arrays of points from D and U distributed according to the
        Sinkhorn solution.
    domain : Domain
        Dataclass for Lambda and U specifications
    plot_options : PlotOptions
        Dataclass for plot and file-naming specifications
    """
    sink_norm = PowerNorm(gamma=1, vmin=0, vmax=sinkhorn_sol.max())
    ax.pcolormesh(
        q_grid, u_grid, sinkhorn_sol, norm=sink_norm, cmap="cividis", **MESH_OPTIONS
    )
    sinkhorn_subset_sample = rng.choice(
        len(q_sinkhorn_subset), size=plot_options.scatter_count, replace=False
    )
    ax.scatter(
        x=q_sinkhorn_subset[sinkhorn_subset_sample],
        y=u_sinkhorn_subset[sinkhorn_subset_sample],
        color="lavender",
        **SCATTERPLOT_OPTIONS,
    )
    D_U_plot_styling(
        ax,
        float(domain.U_mins[0]),
        float(domain.U_maxes[0]),
        plot_options.D_min,
        plot_options.D_max,
    )


def create_SCP_Lambda_plot(
    ax: Axes,
    marginal_scp_df: pd.DataFrame,
    lambda_tgd_subset: np.ndarray,
    scp_norm: PowerNorm,
    lambda_mins: np.ndarray,
    lambda_maxes: np.ndarray,
) -> None:
    """Depict the SCP solution in Lambda with U marginalized out.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis.
    marginal_scp_df : pd.DataFrame
        Dataframe with columns of form xmin, ymin, xmax, ymax, prob, where "x"
        and "y" may be interchanged with two other variables. The "x" and "y"
        correspond to dimensions of Lambda.
    lambda_tgd_subset : (N, 2) numpy array
        A subset of points drawn from the TGD (projected onto Lambda) for
        plotting purposes.
    scp_norm : PowerNorm
        Color scaling object for heatmap.
    lambda_mins, lambda_maxes : (2, ) numpy array
        Minimums/maximums of the boundaries for two relevant axes of Lambda.
        Must be of length 2.
    """
    # Make sure we didn't accidentally feed in all the dimensions of Lambda,
    # which would apply the incorrect axis limits later.
    assert len(lambda_mins) == 2

    ax.pcolormesh(
        *probs_to_mesh(marginal_scp_df), norm=scp_norm, cmap="viridis", **MESH_OPTIONS
    )
    ax.scatter(
        lambda_tgd_subset[:, 0],
        lambda_tgd_subset[:, 1],
        color="coral",
        **SCATTERPLOT_OPTIONS,
    )

    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    ax.set_xlim(lambda_mins[0], lambda_maxes[0])
    ax.set_ylim(lambda_mins[1], lambda_maxes[1])


def make_3d_scp_plot(
    ax: Axes3D,
    df: pd.DataFrame,
    tgd_subset_dim1: np.ndarray,
    tgd_subset_dim2: np.ndarray,
    tgd_subset_dim3: np.ndarray,
    input_mins: np.ndarray,
    input_maxes: np.ndarray,
    num_slices: int,
    elev: float,
    azim: float,
    labels: list[str],
) -> None:
    """Make stacked heatmaps for a 3D SCP dataframe.

    Parameters
    ----------
    ax : Axes3D
        3D Matplotlib axis
    df : pd.DataFrame
        Dataframe with columns {x,y,z}{min, max}, and prob. Relabel the columns
        if variables other than {x, y, z} are used.
    tgd_subset_dim1, tgd_subset_dim2, tgd_subset_dim3 : (N, ) numpy array
        Each of the dimensions of a subset points drawn from the TGD
    input_mins, input_maxes : (3, ) numpy array
        Minimums/maximums of the boundaries for three chosen dimensions of
        Lambda or Lambda x U. Must be of length 3.
    num_slices : int
        Number of heatmaps to stack
    elev, azim : float
        Angle to display 3D figure at
    labels : list[str]
        List of axis labels, corresponding to the order of how TGD points are
        specified
    """
    # Get num_slices values of zmin from our dataset
    # https://stackoverflow.com/questions/50685409/select-n-evenly-spaced-out-elements-in-array-including-first-and-last
    z_vals = np.sort(df.zmin.unique())
    idx = np.unique(np.linspace(0, len(z_vals) - 1, num_slices).round().astype(int))
    z_slices = z_vals[idx]

    norm = PowerNorm(gamma=1, vmin=df.prob.min(), vmax=df.prob.max())

    for z in z_slices:
        df_z = df[df.zmin == z]

        # Get the probability heatmap for (X, Y) at this fixed value of Z
        x_edges, y_edges, Z = probs_to_mesh(df_z)
        X, Y = np.meshgrid(x_edges, y_edges)
        Z_plane = np.full_like(X, z)  # Store the constant z-value in same shape as X

        # Assign colors to according to probability value, NOT just z
        colors = plt.get_cmap("viridis")(norm(Z), alpha=0.5)

        ax.plot_surface(
            X,
            Y,
            Z_plane,
            facecolors=colors,
            linewidth=0,
            rstride=1,
            cstride=1,
            shade=False,
            rasterized=False,
            antialiased=True,
        )

    ax.text(
        0.1,
        0.1,
        z_slices[0],
        r"$O$",
        fontsize=12,
        color="black",
        ha="left",
        va="bottom",
    )
    ax.scatter(0, 0, z_slices[0], color="black", s=3, zorder=10)
    ax.scatter(
        xs=tgd_subset_dim1,
        ys=tgd_subset_dim2,
        zs=tgd_subset_dim3,  # type: ignore
        color="red",
        s=2,
        depthshade=False,
    )

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("none")  # type: ignore
        axis.pane.fill = False  # type: ignore
    ax.grid(False)

    ax.view_init(elev=elev, azim=azim)

    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(3))

    ax.set_xlim(input_mins[0], input_maxes[0])
    ax.set_ylim(input_mins[1], input_maxes[1])
    ax.set_zlim(input_mins[2], input_maxes[2])

    ax.set_xlabel(labels[0], fontsize=13)
    ax.set_ylabel(labels[1], fontsize=13)
    ax.set_zlabel(labels[2], fontsize=13)
