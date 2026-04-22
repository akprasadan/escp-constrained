import numpy as np
import pandas as pd


def probs_to_mesh(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Convert tile-wise probabilities into matplotlib.pyplot.pcolormesh() inputs.

    The DataFrame df must contain columns: xmin, xmax, ymin, ymax, prob.
    Tiles are sorted by (y, x) and reshaped into a (ny, nx) grid.
    The DataFrame is the output of running one of
    estimate_functions.synthetic_scp() or
    estimate_functions.empirical_scp().

    Parameters
    ----------
    df : pd.Dataframe
        Dataframe with tile boundaries and probabilities.

    Returns
    -------
    x_edges : (nx, ) numpy array
        Sorted unique x-boundaries.
    y_edges : (ny, ) numpy array
        Sorted unique y-boundaries.
    Z : (ny - 1, nx - 1) numpy array
        Grid of probability values
    """
    x_edges = np.unique(np.concatenate([df.xmin, df.xmax]))
    y_edges = np.unique(np.concatenate([df.ymin, df.ymax]))

    # Note that in our usage ny = nx = h, since each axis of Lambda
    # is sliced into h pieces, to get h^2 rectangles.
    nx, ny = len(x_edges), len(y_edges)

    # If you swap ymin/xmin, transpose is plotted
    Z = df.sort_values(["ymin", "xmin"]).prob.to_numpy().reshape(ny - 1, nx - 1)

    return x_edges, y_edges, Z
