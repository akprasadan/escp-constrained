"""data_processing.py.

Helper functions to process the output of the SCP algorithm, typically the
probs_df pd.DataFrame.
"""

import numpy as np
import pandas as pd


def simplify_low_dim_dfs(
    df: pd.DataFrame, dim: int, column_name: str, axes_names: list[str]
) -> pd.DataFrame:
    """
    Convert SCP solution dataframe to more readable format.

    If the dataframe has column comprised of dim-many sub-columns, where dim <=
    4, we split them up. For example, if lambda is a column of 3D values in
    Lambda, split it into "x, y, z", and if "q" is a column of 2D values in D,
    split it into "q_x, q_y". But do not do anything if the dimension is > 3.
    This will be applied to prior_df or tgd_df. Similar logic for probs_df will
    be included in src.probability_functions.py::estimate_grid_probs_df()
    directly.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a column 'column_name' comprised of lists of length dim.
    column_name : str
        Column of df with multivariate values of length dim.
    axes_names : list[str]
        List of possible new column names, of length at least larger than dim.

    Returns
    -------
    pd.DataFrame
        New dataframe with split up columns, and original multivariate columns
        removed.
    """
    if dim > 4:
        return df

    needed_axis_names = axes_names[:dim]
    df[needed_axis_names] = df[column_name].to_list()
    return df.drop(columns=column_name)


def marginalize_scp_df(df: pd.DataFrame, dims_to_keep: list[str]) -> pd.DataFrame:
    """Marginalize out all but the given columns in a SCP solution.

    Parameters
    ----------
    df : pd.DataFrame
        Expected to have columns {x, y, ...}min and {x, y, ...}max and prob. The
        number of "min" (or "max") columns is the dimension of Lambda + U, and
        the last most one corresponds to U. In our examples, U will either
        correspond to zmin/zmax or wmin/wmax depending on the dimension.
    dims_to_keep : list[str]
        List of axis labels, like ["x", "y"] that we should get marginals for.
        I.e., marginalize out all other variables.
    """
    min_cols = [f"{dim}min" for dim in dims_to_keep]
    max_cols = [f"{dim}max" for dim in dims_to_keep]
    # Put them together ordered as [xmin, xmax, ymin, ymax, ...]
    cols_to_keep = [
        col for pair in zip(min_cols, max_cols, strict=True) for col in pair
    ]
    return df.groupby(cols_to_keep, as_index=False).agg(prob=("prob", "sum"))


def marginalize_out_U(df: pd.DataFrame) -> pd.DataFrame:
    """Marginalize out U in an SCP solution.

    Resulting solution has a density in the first two/three dimensions (Lambda).

    Parameters
    ----------
    df : pd.DataFrame
        Expected to have columns {x, y, ...}min and {x, y, ...}max and prob, in
        that order. The number of "min" (or "max") columns is the dimension of
        Lambda + U, and the last most one corresponds to U. In our examples, U
        will either correspond to zmin/zmax or wmin/wmax depending on the
        dimension.

    Returns
    -------
    pd.DataFrame
        A dataframe with columns {x,y, ...}min, {x,y, ...}max, and prob,
        corresponding to Lambda. If the U axis was z, only the x and y columns
        would remain, and the "prob" value is the summed probability values
        across the different zmin, zmax combinations for each fixed quadruple
        (xmin, xmax, ymin, ymax).

    Warning
    -------
    This requires the U column to be the last one when ordered alphabetically
    (x, y, z, w), where we redefine w to come after z. So if the columns are x,
    y, z, then U should be the z column, and if they are x, y, z, w, then U
    should be the w column.
    """
    # Task: Extract the Lambda only columns, e.g., ["xmin", "xmax", "ymin",
    # "ymax"]
    possible_axes = ["x", "y", "z", "w"]
    dimension_lambda_and_U = (len(df.columns) - 1) // 2
    dimension_lambda = dimension_lambda_and_U - 1
    lambda_axes = possible_axes[:dimension_lambda]
    return marginalize_scp_df(df, lambda_axes)


def marginalize_out_lambda(df: pd.DataFrame) -> pd.DataFrame:
    """Marginalize out Lambda in probs_df and have a density in U.

    Parameters
    ----------
    df : pd.DataFrame
        Expected to have columns {x, y, ...}min and {x, y, ...}max and prob. The
        number of "min" (or "max") columns is the dimension of Lambda + U, and
        the last most one corresponds to U. In our examples, U will either
        correspond to zmin/zmax or wmin/wmax depending on the dimension.

    Returns
    -------
    pd.DataFrame
        A dataframe with columns for the min/max values of U and prob. If U was
        "z", then it would have "zmin", "zmax", and "prob". The "prob" column is
        then formed by summing probability values across different (xmin, xmax,
        ymin, ymax) combinations for each fixed (zmin, zmax) pair.

    Warning
    -------
    This requires the U column to be the last one when ordered alphabetically
    (x, y, z, w), where we redefine w to come after z. So if the columns are x,
    y, z, then U should be the z column, and if they are x, y, z, w, then U
    should be the w column.
    """
    # Get the U_columns, "zmin" or "zmax" for instance.
    possible_axes = ["x", "y", "z", "w"]
    dimension_lambda_and_U = (len(df.columns) - 1) // 2
    U_axis = possible_axes[dimension_lambda_and_U - 1]
    return marginalize_scp_df(df, [U_axis])


def probs_to_mesh(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert tile-wise probabilities into matplotlib.pyplot.pcolormesh() inputs.

    The DataFrame df must contain columns: xmin, xmax, ymin, ymax, prob. The
    DataFrame is the output of running one of estimate_functions.synthetic_scp()
    or estimate_functions.empirical_scp(). If the columns use variables other
    than x and y, like w and z, rename them to x and y before calling this
    function. Tiles, corresponding to rows of the dataframe, are then sorted by
    (y, x) and reshaped into a (ny, nx) grid.

    Parameters
    ----------
    df : pd.DataFrame
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

    # Note that in our usage ny = nx = h, since each axis of Lambda is sliced
    # into h pieces, to get h^2 rectangles.
    nx, ny = len(x_edges), len(y_edges)

    # If you swap ymin/xmin, transpose is plotted
    Z = df.sort_values(["ymin", "xmin"]).prob.to_numpy().reshape(ny - 1, nx - 1)

    return x_edges, y_edges, Z


def relabel_2D_scp_df(df: pd.DataFrame, axis_one: str, axis_two: str) -> pd.DataFrame:
    """Relabel an SCP solution dataframe with 2 dimensions.

    If dataframe has 4 columns of the form {x,y,z}{min, max}, e.g., xmin, xmax,
    zmin, zmax, automatically relabel them to xmin, xmax, ymin, ymax. This is so
    we can then pass the dataframe into src/data_processing.py::probs_to_mesh.
    If the columns are already xmin, ymin, zmin, zmax, this function has no
    effect on the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with two columns of the form {x, y, z}min, for distinct
        initial letters, and then the corresponding 2 {x, y, z}max columns.
    axis_one, axis_two : str
        A single character string indicating which variable appearing in the
        columns should be relabeled as x and y, respectively.

    Returns
    -------
    pd.DataFrame
    """
    # Check input is valid
    df_cols = list(df.columns)
    if len(df_cols) != 5:
        raise ValueError("Too many columns in df.")
    if df_cols[-1] != "prob":
        raise ValueError("Missing 'probs' column in df.")

    min_col_count = sum("min" in col for col in df_cols)
    max_col_count = sum("max" in col for col in df_cols)

    if min_col_count != 2 or max_col_count != 2:
        raise ValueError("df should have 2 'min' columns and 2 'max' columns")
    if f"{axis_one}min" not in df_cols:
        raise ValueError(f"df missing {axis_one} column.")
    if f"{axis_two}min" not in df_cols:
        raise ValueError(f"df missing {axis_one} column.")

    return df.rename(
        columns={
            f"{axis_one}min": "xmin",
            f"{axis_one}max": "xmax",
            f"{axis_two}min": "ymin",
            f"{axis_two}max": "ymax",
        }
    )
