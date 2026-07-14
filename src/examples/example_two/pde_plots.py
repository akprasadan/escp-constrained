"""
pde_plots.py.

Module to generate some sample solutions to the PDE of example two. Saves the
resulting PDE solutions in plots/examples/example_two/.
"""

from pathlib import Path

import numpy as np
from fipy import CellVariable, DiffusionTerm, ExponentialConvectionTerm, Grid2D
from fipy.solvers import DefaultAsymmetricSolver
from fipy.tools import numerix
from matplotlib import pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "text.usetex": True})


def solve_example_heat_eqs(
    convCoeff1: float,
    convCoeff2: float,
    diffCoeff: float,
    source_pos_y: float,
    nx: int,
    ny: int,
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
    nx, ny : int
        The resolution to solve the PDE with.

    Returns
    -------
    CellVariable
        A FiPy object containing the solution.
    """
    mesh = Grid2D(nx=nx, ny=ny, dx=1 / nx, dy=1 / ny)
    x, y = mesh.cellCenters
    sourceTerm = CellVariable(
        name="source",
        mesh=mesh,
        value=3 * numerix.exp(-((x - 0.5) ** 2 / 0.1 + (y - source_pos_y) ** 2 / 0.05)),
    )

    x_face, _ = mesh.faceCenters
    top_face_func = (
        x_face * (1 - x_face) ** 4 / ((1 / 5) * (4 / 5) ** 4)
    )  # A Beta(2, 5) PDF, scaled so max height is 1

    var = CellVariable(name="temperature", mesh=mesh, value=0.0)

    var.constrain(0.0, where=mesh.facesLeft)  # Constant Dirichlet conditions
    var.constrain(0.0, where=mesh.facesRight)
    var.constrain(0.0, where=mesh.facesBottom)
    var.constrain(top_face_func, where=mesh.facesTop)  # Non-constant Dirichlet cond.

    eq = (
        DiffusionTerm(coeff=diffCoeff)
        + ExponentialConvectionTerm(coeff=(convCoeff1, convCoeff2))  # type: ignore
        == -sourceTerm
    )

    eq.solve(var=var, solver=DefaultAsymmetricSolver(tolerance=1e-5))  # type: ignore

    return var


def extract_geometry_from_solution(
    solution: CellVariable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get X, Y, and Z grid for use by Axes.pcolormesh().

    Parameters
    ----------
    solution : CellVariable
        A FiPy object storing results of a PDE solution.
    """
    solution_mesh = solution.mesh
    nx, ny = solution_mesh.nx, solution_mesh.ny
    x_vals, y_vals = solution_mesh.cellCenters
    X = x_vals.reshape((ny, nx))
    Y = y_vals.reshape((ny, nx))
    Z = solution.value.reshape((ny, nx))
    return X, Y, Z


if __name__ == "__main__":
    # PDE parameters: Convection, diffusion, and vertical-mean of source term
    # position
    param_combos = [(0.5, 2, 1, 0.25), (2, 0.5, 0.1, 0.25), (0.5, 2, 0.1, 0.25)]

    # Grid resolution to solve PDE on we intentionally rewrite a new function
    # for the heat equation solver since the one we already used in our examples
    # hardcodes the NX = NY = 20 and the associated mesh.
    NX = NY = 100

    fig, axes = plt.subplots(
        1, 3, figsize=(8, 8 / 3), sharex=True, sharey=True, layout="constrained"
    )

    for i, ax in enumerate(axes.ravel()):
        params = param_combos[i]
        solution = solve_example_heat_eqs(*params, nx=NX, ny=NY)
        X, Y, Z = extract_geometry_from_solution(solution)

        ax.pcolormesh(X, Y, Z, cmap="inferno", shading="auto", rasterized=True)

        ax.scatter([0.5, 0.5], [0.25, 0.5], color="white", s=2)
        label_offset = 0.015

        ax.text(
            0.5 + label_offset, 0.25 + label_offset, "S", color="white", fontsize=13
        )
        ax.text(0.5 + label_offset, 0.5 + label_offset, "M", color="white", fontsize=13)

        ax.set_title(
            rf"$(\vec{{\lambda}}, u)=({params[0]}, {params[1]}, {params[2]}, {params[3]})$",
            fontsize=14,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    out_path = Path("plots/example_two/pde_example.pdf")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=720)
    plt.close(fig)
