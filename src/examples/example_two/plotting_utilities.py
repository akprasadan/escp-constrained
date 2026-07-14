"""plotting_utilities.py for Example 2.

This module creates the analogue to
`src/examples/example_one/plotting_utilities::create_example_one_plot()` for an
experiment where Lambda is 3D instead of 2D, leading to slightly different
plotting choices. It also has a helper function to relabel a marginalized SCP
dataframe.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.ticker import MaxNLocator

from src.data_processing import marginalize_out_U, marginalize_scp_df, relabel_2D_scp_df
from src.examples.data_generation_utilities import Domain, Experiment, PlotOptions
from src.examples.plotting_utilities import (
    create_D_marginal_plots,
    create_D_U_sinkhorn_plot,
    create_SCP_Lambda_plot,
    create_true_pushforward_plot,
    create_U_marginal_plots,
    make_3d_scp_plot,
)

SEED = 20251028


def create_example_two_plot(
    domain: Domain, expt: Experiment, plot_options: PlotOptions
) -> None:
    """
    Make plot for example two (or future examples).

    Assumes experimental data already generated. This is intended for scenarios
    where Lambda is 3D and U is 1D. See
    `src/examples/example_one/plotting_utilities::create_example_one_plot()` for
    a slightly different version, where Lambda is 2D.

    - A 2 x 3 plot grid showing the paired and unpaired SCP solutions
      marginalized over each possible pairing of Lambda dimensions, i.e., 6
      different plots. A subset of TGD points are included (projected onto each
      corresponding pair of Lambda axes).
    - A 2 x 2 plot grid showing the D and U marginals (from TGD, Sinkhorn,
      unpaired SCP solution (using Sinkhorn), paired SCP solution, the D x U
      true joint distribution (TGD pushforward), and the D x U joint Sinkhorn
      estimate. Either joint plot shows samples from the corresponding
      distribution.
    - A 1 x 2 plot grid showing the marginal 3D SCP on Lambda (not U) as a
      stacked heatmap, paired and unpaired.

    Parameters
    ----------
    domain : Domain
        Dataclass for Lambda and U specifications
    expt: Experimental
        Dataclass for experimental hyper-parameters for Sinkhorn and SCP
    plot_options : PlotOptions
        Dataclass for plot and file-naming specifications.
    """
    rng = np.random.default_rng(seed=SEED)

    data_dir = Path(f"data/example_{plot_options.folder_name}")
    data = np.load(data_dir / f"{plot_options.expt_name}_arrays.npz")

    lambda_tgd_vals = data["lambda_tgd_vals"]
    lambda_tgd_subset = rng.choice(
        lambda_tgd_vals, size=plot_options.scatter_count, replace=False
    )
    u_grid, q_grid = data["u_grid"], data["q_grid"]
    q_vals = data["q_vals"]
    u_tgd_vals = data["u_tgd_vals"]
    sinkhorn_sol = data["sinkhorn_sol"]
    q_sinkhorn = data["q_sinkhorn"]
    u_sinkhorn = data["u_sinkhorn"]

    # we save a subset of the (q, u) joint pairs for plotting purposes
    joint_sample_idx = rng.choice(expt.num_joint_pairs, size=5000, replace=True)
    q_sinkhorn_subset = q_sinkhorn[joint_sample_idx]
    u_sinkhorn_subset = u_sinkhorn[joint_sample_idx]

    unpaired_scp_df = pd.read_csv(
        data_dir / f"unpaired_{plot_options.expt_name}_prob_df.csv"
    )
    paired_scp_df = pd.read_csv(
        data_dir / f"paired_{plot_options.expt_name}_prob_df.csv"
    )

    fig, axes = plt.subplots(2, 2, figsize=plot_options.figsize, layout="constrained")
    ax_U, ax_D, ax_D_U, ax_D_U_sink = axes.ravel()

    u_q_tgd_sample = rng.choice(expt.num_inputs, size=5000, replace=True)

    create_U_marginal_plots(
        ax_U,
        u_sinkhorn_subset,
        u_tgd_vals[u_q_tgd_sample],
        unpaired_scp_df,
        paired_scp_df,
    )
    ax_U.tick_params(labelsize=11)
    ax_U.get_legend().set_loc("upper right")

    create_D_marginal_plots(
        ax_D,
        q_sinkhorn_subset,
        q_vals[u_q_tgd_sample],
        unpaired_scp_df,
        paired_scp_df,
        expt.Q,
        plot_options.D_min,
        plot_options.D_max,
    )
    ax_D.set_xlim(0.16, 0.4)
    ax_D.tick_params(labelsize=11)

    create_true_pushforward_plot(
        ax_D_U, rng, q_vals, u_tgd_vals, domain, expt, plot_options
    )
    # override axis limits
    ax_D_U.set_xlim(0.16, 0.32)

    create_D_U_sinkhorn_plot(
        ax_D_U_sink,
        rng,
        q_grid,
        u_grid,
        sinkhorn_sol,
        q_sinkhorn_subset,
        u_sinkhorn_subset,
        domain,
        plot_options,
    )
    # override axis limits
    ax_D_U_sink.set_xlim(0.16, 0.32)
    # Turn off titles for publication purposes
    for ax in axes.ravel():
        ax.set_title("")

    out_path = Path(
        f"plots/example_{plot_options.folder_name}/{plot_options.expt_name}_D_U_plots.pdf"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=plot_options.dpi)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(8, 8 * 2 / 3), layout="constrained")

    fig.set_constrained_layout_pads(rect=(0.05, 0, 0.93, 1))  # type: ignore

    (
        paired_conv1_conv2_ax,
        paired_conv1_diff_ax,
        paired_conv2_diff_ax,
        unpaired_conv1_conv2_ax,
        unpaired_conv1_diff_ax,
        unpaired_conv2_diff_ax,
    ) = axes.ravel()

    # Get Lambda marginal SCP solutions
    paired_scp_lambda_marg_conv1_conv2 = marginalize_scp_df(paired_scp_df, ["x", "y"])
    paired_scp_lambda_marg_conv1_diff = marginalize_scp_df(paired_scp_df, ["x", "z"])
    paired_scp_lambda_marg_conv2_diff = marginalize_scp_df(paired_scp_df, ["y", "z"])

    unpaired_scp_lambda_marg_conv1_conv2 = marginalize_scp_df(
        unpaired_scp_df, ["x", "y"]
    )
    unpaired_scp_lambda_marg_conv1_diff = marginalize_scp_df(
        unpaired_scp_df, ["x", "z"]
    )
    unpaired_scp_lambda_marg_conv2_diff = marginalize_scp_df(
        unpaired_scp_df, ["y", "z"]
    )

    scp_norm = PowerNorm(
        gamma=1,
        vmin=0,
        vmax=max(
            paired_scp_lambda_marg_conv1_conv2["prob"].max(),
            paired_scp_lambda_marg_conv1_diff["prob"].max(),
            paired_scp_lambda_marg_conv2_diff["prob"].max(),
            unpaired_scp_lambda_marg_conv1_conv2["prob"].max(),
            unpaired_scp_lambda_marg_conv1_diff["prob"].max(),
            unpaired_scp_lambda_marg_conv2_diff["prob"].max(),
        ),
    )

    create_SCP_Lambda_plot(
        paired_conv1_conv2_ax,
        relabel_2D_scp_df(
            paired_scp_lambda_marg_conv1_conv2, "x", "y"
        ),  # no need to relabel
        lambda_tgd_subset[:, [0, 1]],
        scp_norm,
        domain.lambda_mins[[0, 1]],
        domain.lambda_maxes[[0, 1]],
    )
    paired_conv1_conv2_ax.set_title(r"Conv. $y$ vs. Conv. $x$", fontsize=14)

    create_SCP_Lambda_plot(
        paired_conv1_diff_ax,
        relabel_2D_scp_df(paired_scp_lambda_marg_conv1_diff, "x", "z"),
        lambda_tgd_subset[:, [0, 2]],
        scp_norm,
        domain.lambda_mins[[0, 2]],
        domain.lambda_maxes[[0, 2]],
    )

    paired_conv1_diff_ax.set_title(r"Diffusion vs. Conv. $x$", fontsize=14)
    create_SCP_Lambda_plot(
        paired_conv2_diff_ax,
        relabel_2D_scp_df(paired_scp_lambda_marg_conv2_diff, "y", "z"),
        lambda_tgd_subset[:, [1, 2]],
        scp_norm,
        domain.lambda_mins[[1, 2]],
        domain.lambda_maxes[[1, 2]],
    )
    paired_conv2_diff_ax.set_title(r"Diffusion vs. Conv. $y$", fontsize=13)

    create_SCP_Lambda_plot(
        unpaired_conv1_conv2_ax,
        relabel_2D_scp_df(
            unpaired_scp_lambda_marg_conv1_conv2, "x", "y"
        ),  # no need to relabel
        lambda_tgd_subset[:, [0, 1]],
        scp_norm,
        domain.lambda_mins[[0, 1]],
        domain.lambda_maxes[[0, 1]],
    )
    unpaired_conv1_conv2_ax.set_title(r"Conv. $y$ vs. Conv. $x$", fontsize=14)

    create_SCP_Lambda_plot(
        unpaired_conv1_diff_ax,
        relabel_2D_scp_df(unpaired_scp_lambda_marg_conv1_diff, "x", "z"),
        lambda_tgd_subset[:, [0, 2]],
        scp_norm,
        domain.lambda_mins[[0, 2]],
        domain.lambda_maxes[[0, 2]],
    )
    unpaired_conv1_diff_ax.set_title(r"Diffusion vs. Conv. $x$", fontsize=14)

    create_SCP_Lambda_plot(
        unpaired_conv2_diff_ax,
        relabel_2D_scp_df(unpaired_scp_lambda_marg_conv2_diff, "y", "z"),
        lambda_tgd_subset[:, [1, 2]],
        scp_norm,
        domain.lambda_mins[[1, 2]],
        domain.lambda_maxes[[1, 2]],
    )
    unpaired_conv2_diff_ax.set_title(r"Diffusion vs. Conv. $y$", fontsize=14)

    for ax in axes.ravel():
        ax.tick_params(labelsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))

    text_options = {
        "fontsize": 16,
        "fontweight": "bold",
        "va": "center",
        "rotation": 90,
    }
    fig.text(0.01, 0.75, "Paired", **text_options)
    fig.text(0.01, 0.25, "Unpaired", **text_options)

    out_path = Path(
        f"plots/example_{plot_options.folder_name}/{plot_options.expt_name}_2d_plots.pdf"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=plot_options.dpi)
    plt.close(fig)

    unpaired_scp_marg_lambda = marginalize_out_U(unpaired_scp_df)
    paired_scp_marg_lambda = marginalize_out_U(paired_scp_df)

    fig, axes = plt.subplots(
        1, 2, figsize=(6, 9), subplot_kw={"projection": "3d"}, layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=0, h_pad=0, wspace=0, hspace=0)  # type: ignore

    paired_ax, unpaired_ax = axes
    for ax in axes:
        ax.set_box_aspect([1, 1, 2])
        ax.tick_params(labelsize=9)
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
        ax.zaxis.label.set_fontsize(10)

    make_3d_scp_plot(
        paired_ax,
        paired_scp_marg_lambda,
        lambda_tgd_subset[:, 0],
        lambda_tgd_subset[:, 1],
        lambda_tgd_subset[:, 2],
        domain.lambda_mins,
        domain.lambda_maxes,
        num_slices=10,
        elev=15,
        azim=230,
        labels=[r"Conv. $x$", r"Conv. $y$", r"Diffusion"],
    )

    make_3d_scp_plot(
        unpaired_ax,
        unpaired_scp_marg_lambda,
        lambda_tgd_subset[:, 0],
        lambda_tgd_subset[:, 1],
        lambda_tgd_subset[:, 2],
        domain.lambda_mins,
        domain.lambda_maxes,
        num_slices=10,
        elev=15,
        azim=230,
        labels=[r"Conv. $x$", r"Conv. $y$", r"Diffusion"],
    )

    out_path = Path(
        f"plots/example_{plot_options.folder_name}/{plot_options.expt_name}_3d_plots.pdf"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=plot_options.dpi, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
