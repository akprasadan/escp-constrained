"""pde_benchmarking.py.

Compare run-time of evaluating PDE from example_two.py using FiPy, then with
multi-threading, and then using the XGBoost prediction operation. Runs 10 trials
of 3000 evaluations of each method and stores results in
data/example_two/benchmarking.csv.

Below are the results on a 2024 MacBook Pro with an Apple M4 Pro Chip with 14
cores, 24 GB Memory running the operating system Tahoe 26.5.1:

- FiPy with Multithreading: 6.3 milliseconds (ms) per evaluation
- FiPy w/o Multithreading: 15.4 ms per evaluation
- XGBoost Prediction: 0.000931 ms per evaluation
"""

import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src.examples.example_two.example_two import heat_eq_xgb_predict
from src.examples.example_two.q_computations import heat_equation_w_source_at_pos
from src.output_functions import apply_map
from src.sampler_functions import sample_distr

CONV_1_BDS = (0.05, 5)
CONV_2_BDS = (0.05, 5)
DIFF_BDS = (0.05, 1)
SOURCE_POS_Y_BDS = (0.05, 0.95)
MIN_VEC = np.array([CONV_1_BDS[0], CONV_2_BDS[0], DIFF_BDS[0], SOURCE_POS_Y_BDS[0]])
MAX_VEC = np.array([CONV_1_BDS[1], CONV_2_BDS[1], DIFF_BDS[1], SOURCE_POS_Y_BDS[1]])


def test_point_distribution(rng: np.random.Generator, J: int) -> np.ndarray:
    """
    Sample points from the 4D heat equation parameter space.

    Parameters
    ----------
    rng : np.random.Generator
        For reproducibility.
    J : int
        Number of samples to draw

    Returns
    -------
    (J, 4) numpy array drawn uniformly
    """
    return rng.uniform(MIN_VEC, MAX_VEC, size=(J, 4))


def run_profiling(N: int, n_trials: int) -> None:
    """Benchmark FiPy and XGBoost model for PDE computations.

    Parameters
    ----------
    N : int
        Number of parameters to try per trial for each method.
    n_trials : int
        Number of times to repeat each method's computations.
    """
    rows = []

    for i in range(n_trials):
        print(f"Starting Trial {i + 1} / {n_trials}")

        points = sample_distr(N, seed=42 + i, distr=test_point_distribution)

        t0 = time.perf_counter()
        with Pool() as p:
            p.starmap(heat_equation_w_source_at_pos, points, chunksize=1000)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        apply_map(points, heat_equation_w_source_at_pos)  # type: ignore
        t3 = time.perf_counter()

        t4 = time.perf_counter()
        # Relies on q_model XGBoost model saved in example_two.py
        apply_map(points, heat_eq_xgb_predict)
        t5 = time.perf_counter()

        new_row = {
            "trial": i,
            "n_samples": N,
            "fipy_mt": t1 - t0,
            "fipy_no_mt": t3 - t2,
            "xgb": t5 - t4,
            "total": t5 - t0,
        }
        rows.append(new_row)

    df = pd.DataFrame(rows)
    df.to_csv("data/example_two/benchmarking.csv", index=False)


if __name__ == "__main__":
    run_profiling(N=3000, n_trials=10)
