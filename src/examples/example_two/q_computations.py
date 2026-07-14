"""
q_computations.py.

Perform one time computation of heat equation evaluations to create a surrogate
model (an XGBoost regression) for the heat equation detailed in
src.examples.example_two.example_two.py.

We pick a dense scattering of points in a 4D hyper-rectangle of parameters over
which to evaluate Q, the heat equation solver. Computation is relatively cheap
so we use the following generous budget: Draw a large (50K) sample of points on
a uniform distribution on the 4D domain, and supplement this with a smaller
sampling of ~10K points that concentrate on the edges/corners of the 4D domain.
Assess whether additional sampling is needed for better interpolation accuracy
by checking the diagnostic output from fit_xgb.py.

Warning: This can be slow (20-25 minutes on a 2024 MacBook Pro with an M4 Pro
chip). To speed up, consider reducing the resolution of the PDE solver, set with
NX and NY, or reducing the tolerance in DefaultAsymmetricSolver. Alternatively,
reduce the sampling budget or use a more efficient sampling technique.

Output is stored in data/example_two/heat_equation_with_source.csv. If that file
already exists, running this function will append new observations to that fil,
while removing duplicates. But if you must change the experiment, be sure delete
that .csv file to avoid combining solutions to different heat equations in the
same output.
"""

import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from fipy import CellVariable, DiffusionTerm, ExponentialConvectionTerm, Grid2D
from fipy.solvers import DefaultAsymmetricSolver
from fipy.tools import numerix

# Note that NX, NY, and MESH get hardcoded in heat_equation_solver_with_source.
# Use caution if re-importing from this module
NX = NY = 20
MESH = Grid2D(nx=NX, ny=NY, dx=1 / NX, dy=1 / NY)
SEED = 20251028

CONV_1_BDS = (0.05, 5)
CONV_2_BDS = (0.05, 5)
DIFF_BDS = (0.05, 1)
SOURCE_POS_Y_BDS = (0.05, 0.95)


def heat_equation_solver_with_source(
    convCoeff1: float, convCoeff2: float, diffCoeff: float, source_pos_y: float
) -> CellVariable:
    """
    Solve example two's PDE.

    Solves the 2D thin-plate steady state diffusion-convection heat equation,
    with Dirichlet conditions of 0 on three of the edges and a Beta distribution
    constraint on the top edge, and a Gaussian source term centered at (0.5,
    source_pos_y).

    Parameters
    ----------
    convCoeff1, convCoeff2 : float
        The two convection parameters
    diffCoeff : float
        A single diffusion parameter
    source_pos_y : float
        The vertical location of a source term.

    Returns
    -------
    CellVariable
        A FiPy object containing the solution.

    Note
    -----
    Be careful with the argument ordering.
    """
    x, y = MESH.cellCenters
    sourceTerm = CellVariable(
        name="source",
        mesh=MESH,
        value=3 * numerix.exp(-((x - 0.5) ** 2 / 0.1 + (y - source_pos_y) ** 2 / 0.05)),
    )

    x_face, _ = MESH.faceCenters
    top_face_func = (
        x_face * (1 - x_face) ** 4 / ((1 / 5) * (4 / 5) ** 4)
    )  # A Beta(2, 5) PDF, scaled so max height is 1

    var = CellVariable(name="temperature", mesh=MESH, value=0.0)

    var.constrain(0.0, where=MESH.facesLeft)  # Constant Dirichlet conditions
    var.constrain(0.0, where=MESH.facesRight)
    var.constrain(0.0, where=MESH.facesBottom)
    var.constrain(top_face_func, where=MESH.facesTop)  # Non-constant Dirichlet cond.

    eq = (
        DiffusionTerm(coeff=diffCoeff)
        + ExponentialConvectionTerm(coeff=(convCoeff1, convCoeff2))  # type: ignore
        == -sourceTerm
    )

    eq.solve(var=var, solver=DefaultAsymmetricSolver(tolerance=1e-5))  # type: ignore

    return var


def heat_equation_w_source_at_pos(
    convCoeff1: float, convCoeff2: float, diffCoeff: float, source_pos_y: float
) -> float:
    """
    Solve example two's PDE and evaluate at position.

    Evaluate at a fixed position the 2D thin-plate steady state
    diffusion-convection heat equation, with Dirichlet conditions of 0 on three
    of the edges and a Beta distribution constraint on the top edge, and a
    Gaussian source term centered at (0.5, source_pos_y).

    Parameters
    ----------
    convCoeff1, convCoeff2 : float
        The two convection parameters
    diffCoeff : float
        A single diffusion parameter
    source_pos_y : float
        The vertical location of a source term.

    Returns
    -------
    CellVariable
        A FiPy object containing the solution.

    Note
    -----
    Be careful with the argument ordering.
    """
    solution = heat_equation_solver_with_source(
        convCoeff1=convCoeff1,
        convCoeff2=convCoeff2,
        diffCoeff=diffCoeff,
        source_pos_y=source_pos_y,
    )
    return solution((0.5, 0.5)).item()


if __name__ == "__main__":
    rng = np.random.default_rng(seed=SEED)
    n_samples_main = 50_000
    n_samples_edges = 10_000

    # Main sampling run (uniform distributions)
    df_main = pd.DataFrame({
        "conv1": rng.uniform(*CONV_1_BDS, n_samples_main),
        "conv2": rng.uniform(*CONV_2_BDS, n_samples_main),
        "diff": rng.uniform(*DIFF_BDS, n_samples_main),
        "source_pos_y": rng.uniform(*SOURCE_POS_Y_BDS, n_samples_main),
    })

    # Supplemental sampling (to better sample near edges/corners) using a
    # multi-dimensional and scaled version of a beta distribution that has sharp
    # peaks near 0 and 1.
    def scaled_beta(bds: tuple[float, float], beta_param: float = 0.4) -> np.ndarray:
        """Scale a Beta(0.4, 0.4) dist to an arbitrary interval.

        Returns
        -------
        (n_samples_edges, ) numpy array
        """
        beta_samples = rng.beta(beta_param, beta_param, n_samples_edges)
        return bds[0] + (bds[1] - bds[0]) * beta_samples

    df_edges = pd.DataFrame({
        "conv1": scaled_beta(CONV_1_BDS),
        "conv2": scaled_beta(CONV_2_BDS),
        "diff": scaled_beta(DIFF_BDS),
        "source_pos_y": scaled_beta(SOURCE_POS_Y_BDS),
    })
    df = pd.concat([df_main, df_edges], ignore_index=True)

    # WARNING: Make sure order of covariates lines up with
    # heat_equation_w_source_at_pos()
    points = df[["conv1", "conv2", "diff", "source_pos_y"]].values

    start = time.perf_counter()
    # Use multi-threading for computation; 10-20x speed-up.
    with Pool() as p:
        # WARNING: check order of arguments to heat_equation_w_source_at_pos It
        # must line up with the points defined a few lines above
        df["q"] = p.starmap(heat_equation_w_source_at_pos, points, chunksize=1000)
    end = time.perf_counter()

    elapsed = end - start
    print(
        f"Forward Model Computations: {elapsed:.3f} s ({elapsed / len(points):.6f} s per eval)"
    )

    path = Path("data/example_two/surrogate_training_data.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined_df = pd.concat([existing_df, df], ignore_index=True)

    combined_df = combined_df.drop_duplicates(
        subset=["conv1", "conv2", "diff", "source_pos_y"]
    )
    combined_df.to_csv(path, index=False)
