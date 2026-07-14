"""
fit_xgb.py.

After heat equation computations of src/examples/example_two/q_computations.py
are done, fit an XGBoost model. This is very fast even for a large dataset
(hundreds of thousands of rows). Also, produce a calibration plot as a
diagnostic check, and print out some error metrics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.examples.example_two.q_computations import heat_equation_w_source_at_pos


def train_xgb() -> None:
    """
    Train XGBoost regression model on heat equation data.

    Read in dataset of heat equation evaluations as a function of parameters and
    train XGBoost regression model. Save results to data/example_two/ folder.
    """
    heat_eq_computation_path = Path("data/example_two/surrogate_training_data.csv")

    if not heat_eq_computation_path.exists():
        raise FileNotFoundError(
            "Missing heat equation computations. Make sure you ran q_computations.py first."
        )

    df = pd.read_csv(heat_eq_computation_path)

    # WARNING: Use the ordering you want the surrogate Q (prediction by the XGB
    # model) and SCP to use.
    X = df[["conv1", "conv2", "diff", "source_pos_y"]].to_numpy(dtype=float)
    y = df["q"].to_numpy(dtype=float)

    model = XGBRegressor(n_estimators=600, learning_rate=0.02, subsample=0.9)
    model.fit(X, y)

    path = Path("data/example_two/xgb_model.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(path)


def check_model_fit() -> None:
    """
    Get XGBoost metrics.

    Assesses quality of XGBoost regression by drawing points from a tiling of
    the domain, evaluating the PDE directly, and computing test error. Make a
    calibration plot and display the error results.
    """
    q_model = XGBRegressor()
    q_model.load_model("data/example_two/xgb_model.json")

    res = 5

    CONV_1_BDS = (0.05, 5)
    CONV_2_BDS = (0.05, 5)
    DIFF_BDS = (0.05, 1)
    SOURCE_POS_Y_BDS = (0.05, 0.95)

    c1 = np.linspace(*CONV_1_BDS, res)
    c2 = np.linspace(*CONV_2_BDS, res)
    d = np.linspace(*DIFF_BDS, res)
    y = np.linspace(*SOURCE_POS_Y_BDS, res)

    C1, C2, D, Y = np.meshgrid(c1, c2, d, y, indexing="ij")
    X = np.column_stack((C1.ravel(), C2.ravel(), D.ravel(), Y.ravel()))

    y_true = np.array([
        heat_equation_w_source_at_pos(c1, c2, d, y) for c1, c2, d, y in X
    ])
    y_pred = q_model.predict(X)

    abs_err = np.abs(y_pred - y_true)
    sq_err = np.square(y_pred - y_true)
    mean_abs = abs_err.mean()
    max_abs = abs_err.max()
    mean_sq = sq_err.mean()
    metrics_message = (
        f"MAE: = {mean_abs:.3f}, MSE = {mean_sq:.3f}, Max abs. error: = {max_abs:.3f}"
    )
    print(metrics_message)
    min_y_val = min(y_true.min(), y_pred.min())
    max_y_val = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.4, s=10)

    # Add diagonal line to calibration plot
    plt.plot([min_y_val, max_y_val], [min_y_val, max_y_val], "--")

    plt.xscale("log")
    plt.yscale("log")

    # more ticks
    plt.gca().xaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1, 2, 5]))
    plt.gca().yaxis.set_major_locator(mticker.LogLocator(base=10, subs=[1, 2, 5]))

    # readable labels
    formatter = mticker.FuncFormatter(lambda x, _: f"{x:.4f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.xlabel("Simulated Temp. (PDE)")
    plt.ylabel("Predicted Temp. (XGB)")
    plt.title("Calibration Plot for XGB Surrogate Model of Heat Equation")

    plt.figtext(0.05, -0.05, metrics_message, fontsize=9)

    path = Path("plots/example_two/surrogate_fit.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    train_xgb()
    check_model_fit()
