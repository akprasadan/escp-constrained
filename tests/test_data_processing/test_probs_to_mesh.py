"""
Unit/integration (visual) tests for `src.data_processing.py::probs_to_mesh()`.

This function takes in the dataframe with columns (xmin, xmax, ymin, ymax, prob)
corresponding to the SCP solution heatmap.

It returns
    - x_edges: array-like of x bin edges
    - y_edges: array-like of y bin edges
    - Z: 2D array of probabilities aligned with the mesh.

These arrays are ultimately feed into matplotlib.axes.Axes.pcolormesh().

This tests that the output is as expected, so that the plots are correct. We do
so both with a unit test (run with the usual pytest), as well as a plot where we
vary where the max probability weight in Lambda goes, verifying that
pcolormesh() when fed the results of probs_to_mesh() yields the right result.

Output:
  - plots/tests/integrated_test_probs_to_mesh.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import PowerNorm
from pandas.testing import assert_frame_equal

from src.data_processing import probs_to_mesh

PURPLE_BG = (68 / 255, 1 / 255, 84 / 255)
DPI = 720


def style_and_annotate(
    ax: Axes,
    bounds: tuple[float, float, float, float],
    bg: tuple[float, float, float],
    label_pos: tuple[float, float],
) -> None:
    """
    Apply plot styling and add the Lambda label.

    Parameters
    ----------
    ax : Axes
        Matplotlib axis.
    bounds : tuple[float, float, float, float]
        (xmin, xmax, ymin, ymax) for axis limits.
    bg : tuple[float, float, float]
        Background color.
    label_pos : tuple[float, float]
        (x, y) coordinate to place capital Greek letter Lambda in plot.
    """
    ax.set_facecolor(bg)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=13)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(
        label_pos[0],
        label_pos[1],
        r"$\Lambda$",
        color="white",
        fontsize=15,
        weight="bold",
    )


def test_probs_to_mesh() -> None:
    """
    Test probs_to_mesh().

    We grid up [0,3] x [0,2] by making a partition of the x-axis into [0, 1],
    [1, 2], [2,3], and the y-axis into [0,1], [1,2]. We then assign a
    'probability' to each (x,y) pair.

    Technically our probability estimation functions will always have the same
    number of sub-intervals for the x and y directions, namely, h, giving rise
    to h^2 rectangles comprising Lambda.
    """
    df_test = pd.DataFrame({
        "xmin": [0, 1, 2, 0, 1, 2],
        "xmax": [1, 2, 3, 1, 2, 3],
        "ymin": [0, 0, 0, 1, 1, 1],
        "ymax": [1, 1, 1, 2, 2, 2],
        "prob": [10, 11, 12, 20, 21, 22],
    })
    xe, ye, Z = probs_to_mesh(df_test)
    assert np.array_equal(xe, np.array([0, 1, 2, 3]))
    assert np.array_equal(ye, np.array([0, 1, 2]))
    assert Z.shape == (len(ye) - 1, len(xe) - 1)
    assert np.array_equal(Z, np.array([[10, 11, 12], [20, 21, 22]]))


def test_visual_check_probs_to_mesh() -> None:
    """
    Test probs_to_mesh visually.

    Perform visual check of the function probs_to_mesh() and its usage with
    matplotlib.axes.Axes.pcolormesh().

    Take Lambda = [0, 3] x [0, 2], split the x range into 3 equal-length bins,
    and the y-range into 2. Then for a range of squares in the rectangle, place
    most of the probability mass in that square. Then run probs_to_mesh() on
    each corresponding dataframe and feed into the pcolormesh() function. Assess
    that each plot has the right orientation, i.e., longer on the x-axis than
    y-axis. Each plot has a title indicating where most mass should be, and this
    should be visually clear from the pcolormesh heatmap. For additional
    robustness, each dataframe has its rows permuted to ensure the results are
    not sensitive to how the probability estimate dataframes are sorted.
    """
    out_dir = Path("plots/tests")
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "integrated_test_probs_to_mesh.pdf"

    if path.exists():
        return

    lambda_grid = {
        "xmin": [0, 1, 2, 0, 1, 2],
        "xmax": [1, 2, 3, 1, 2, 3],
        "ymin": [0, 0, 0, 1, 1, 1],
        "ymax": [1, 1, 1, 2, 2, 2],
    }

    # The 10 is where the max weight is placed
    prob_options = [
        [10, 1, 1, 1, 1, 1],  # [0,1] x [0,1]
        [1, 10, 1, 1, 1, 1],  # [1,2] x [0,1]
        [1, 1, 10, 1, 1, 1],  # [2,3] x [0,1]
        [1, 1, 1, 10, 1, 1],  # [0,1] x [1,2]
    ]
    prob_labels = ["[0,1] x [0,1]", "[1,2] x [0,1]", "[2,3] x [0,1]", "[0,1] x [1,2]"]

    results = []
    dfs = []

    for probs in prob_options:
        # Form dataframe as if output by our SCP functions
        df_old = pd.DataFrame({**lambda_grid, "prob": probs})

        # Permute the rows some way to ensure the code works no matter how the
        # dataframe was put together in probability_functions.prob_over_grid().
        permutation = [3, 4, 5, 0, 1, 2]
        df = df_old.iloc[permutation].reset_index(drop=True)

        # Ensure the permutation actually worked If the permutation was simply
        # [0, 1, .., 6], this would fail
        with pytest.raises(AssertionError):
            assert_frame_equal(df_old, df, check_exact=True)

        dfs.append(df)
        results.append(probs_to_mesh(df))

    fig, axes = plt.subplots(2, 2, figsize=(6, 6), constrained_layout=True)
    vmax = max(df_swapped["prob"].max() for df_swapped in dfs)
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)

    for i in range(4):
        ax = axes.ravel()[i]
        x_edges, y_edges, Z = results[i]
        ax.pcolormesh(
            x_edges,
            y_edges,
            Z,
            cmap="viridis",
            norm=norm,
            shading="auto",
            rasterized=True,
        )
        ax.set_title(f"Max weight on {prob_labels[i]}")
        style_and_annotate(
            ax=ax, bounds=(0, 3, 0, 2), bg=PURPLE_BG, label_pos=(2.7, 1.7)
        )

    fig.savefig(path, dpi=DPI, bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    test_visual_check_probs_to_mesh()
