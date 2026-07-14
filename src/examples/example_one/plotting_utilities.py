"""
plotting_utilities.py for Example 1.

Helper functions to make the plots described in
src/examples/example_one/example_one.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm

from src.data_processing import marginalize_out_U
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


def create_example_one_plot(
    domain: Domain, expt: Experiment, plot_options: PlotOptions
) -> None:
    """
    Make plots for example one (or future examples).

    Assumes data has been generated. This is intended for scenarios where Lambda
    is 2D and U is 1D. See
    `src/examples/example_two/plotting_utilities::create_example_two_plot()` for
    a slightly different version, due to Lambda being 3D.

    The following plots are saved in the plots/ folder.

    - A 2 x 1 plot grid showing the unpaired and paired SCP solutions
      marginalized to the 2D Lambda space. A subset of TGD points are included
      (projected onto Lambda).
    - A 2 x 2 plot grid showing the D and U marginals (from TGD, Sinkhorn,
      unpaired SCP solution, and paired SCP solution), the D x U true joint
      distribution (TGD pushforward), and the D x U joint Sinkhorn estimate.
      Either joint plot shows samples from the corresponding distribution.
    - A single plot showing the full 3D SCP (unpaired) on Lambda x U as a
      stacked heatmap.

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
    u_tgd_vals = data["u_tgd_vals"]

    tgd_subset_idx = rng.choice(
        expt.num_inputs, size=plot_options.scatter_count, replace=False
    )
    lambda_tgd_subset = lambda_tgd_vals[tgd_subset_idx]
    u_tgd_subset = u_tgd_vals[tgd_subset_idx]

    u_grid, q_grid = data["u_grid"], data["q_grid"]
    q_vals = data["q_vals"]

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

    # A different sample from tgd_subset_idx, since this is for making a
    # histogram, not directly plotting
    u_q_tgd_sample = rng.choice(expt.num_inputs, size=5000, replace=True)

    create_U_marginal_plots(
        ax_U,
        u_sinkhorn_subset,
        u_tgd_vals[u_q_tgd_sample],
        unpaired_scp_df,
        paired_scp_df,
    )
    ax_U.tick_params(labelsize=11)
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
    ax_D.tick_params(labelsize=11)
    ax_D.set_xlim(-1, 120)

    create_true_pushforward_plot(
        ax_D_U, rng, q_vals, u_tgd_vals, domain, expt, plot_options
    )

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

    # Turn off titles for publication purposes
    for ax in axes.ravel():
        ax.set_title("")

    out_path = Path(
        f"plots/example_{plot_options.folder_name}/{plot_options.expt_name}_D_U_plots.pdf"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=plot_options.dpi)
    plt.close(fig)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(plot_options.figsize[0] / 2, plot_options.figsize[1]),
        layout="constrained",
    )
    paired_scp_ax, unpaired_scp_ax = axes.ravel()

    # Get Lambda marginal SCP solutions
    unpaired_scp_lambda_marginal_df = marginalize_out_U(unpaired_scp_df)
    paired_scp_lambda_marginal_df = marginalize_out_U(paired_scp_df)

    scp_norm = PowerNorm(
        gamma=1,
        vmin=0,
        vmax=max(
            paired_scp_lambda_marginal_df["prob"].max(),
            unpaired_scp_lambda_marginal_df["prob"].max(),
        ),
    )
    create_SCP_Lambda_plot(
        paired_scp_ax,
        paired_scp_lambda_marginal_df,
        lambda_tgd_subset,
        scp_norm,
        domain.lambda_mins,
        domain.lambda_maxes,
    )
    paired_scp_ax.set_title(
        r"SCP Sol. $\hat P_{\Lambda}^t$, Paired Data ($U$ Marginalized Out)"
    )

    create_SCP_Lambda_plot(
        unpaired_scp_ax,
        unpaired_scp_lambda_marginal_df,
        lambda_tgd_subset,
        scp_norm,
        domain.lambda_mins,
        domain.lambda_maxes,
    )
    unpaired_scp_ax.set_title(
        r"SCP Sol. $\hat P_{\Lambda}^t$, Unpaired Data ($U$ marginalized out)"
    )

    # Turn off titles, set param size, add Lambda label
    for ax in axes.ravel():
        ax.tick_params(labelsize=12)
        ax.set_title("")
        ax.text(8.5, -9, r"$\Lambda$", color="white", fontsize=15, weight="bold")

    out_path = Path(
        f"plots/example_{plot_options.folder_name}/{plot_options.expt_name}_2d_plots.pdf"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=plot_options.dpi)
    plt.close(fig)

    fig = plt.figure(figsize=(3, 9), layout="constrained")

    ax = fig.add_subplot(111, projection="3d")
    fig.get_layout_engine().set(w_pad=0, h_pad=0, wspace=0, hspace=0)  # type: ignore

    ax.set_box_aspect([1, 1, 2])

    make_3d_scp_plot(
        ax,
        unpaired_scp_df,
        lambda_tgd_subset[:, 0],
        lambda_tgd_subset[:, 1],
        u_tgd_subset,
        domain.input_mins,  # Use both lambda + u for this example
        domain.input_maxes,
        num_slices=min(len(np.sort(unpaired_scp_df.zmin.unique())), 10),
        elev=15,
        azim=230,
        labels=[r"$\lambda_1$", r"$\lambda_2$", r"$U$"],
    )

    ax.tick_params(labelsize=9)

    ax.xaxis.label.set_fontsize(10)
    ax.yaxis.label.set_fontsize(10)
    ax.zaxis.label.set_fontsize(10)

    out_path = Path(
        f"plots/example_{plot_options.folder_name}/{plot_options.expt_name}_3d_plots.pdf"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=plot_options.dpi, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
