This repository implements an algorithm given in [^1] to solve a so-called
'Statistical Calibration Problem' (SCP), and then extends it to handle an SCP
with a partially known stochastic input space, which we also call a 'constrained
SCP' for short, as opposed to the original 'unconstrained SCP'.  The constrained
SCP is developed in the paper[^2] associated with this repository. The code for
the SCP generalizes the code of [another
repository](https://github.com/akprasadan/escp-continuity.git) authored by
myself for the purpose of another related paper[^3]. The code now permits
arbitrary input and output dimensions, although for the constrained SCP, there
are some additional restrictions. The repo then includes several examples needed
to reproduce all experiments highlighted in the forthcoming paper[^2].

[^1]: Shi, H., Yang, L., Chi, J., Butler, T., Wang, H., Bingham, D., & Estep, D.
    (2026). Nonparametric Bayesian Calibration of Computer Models. *arXiv
    preprint arXiv:2509.22597.*

[^2]: Prasadan, A., Basu, S., Yazdi, F., Bingham, D., and Estep, D., 2026.
    Non-Parametric Model Calibration with  Stochastic Control Parameters. *In preparation*.

[^3]: Prasadan, A., Estep, D. and Bingham, D., 2026. Continuity of the Solution
    of a Non-Parametric Bayesian Statistical Calibration Procedure. *arXiv
    preprint arXiv:2603.20665.* In revision with the SIAM/ASA Journal of
    Uncertainty Quantification.

## Getting Started

Feel free to use any package manager of your choice, but I believe the most
convenient one is **uv**, a popular package/project manager. I will therefore
provide instructions assuming you are using **uv**. You can find installation
instructions for **uv**
[here](https://docs.astral.sh/uv/getting-started/installation/).

1. Clone the repository and navigate to it.

    ```bash
    git clone https://github.com/akprasadan/escp-constrained.git
    cd escp-constrained
    ```

2. Install the project and its dependencies with `uv sync`.

## Usage

Now you can run modules with **uv** in a virtual environment. There are three
main computational examples in the paper. The first two are run as follows:

```bash
uv run python -m src.examples.example_unconstrained.example_unconstrained
uv run python -m src.examples.example_one.example_one
```

For the heat equation example, run the following commands. Warning: Some scripts
might take some time.

```bash
uv run python -m src.examples.example_two.pde_plots
uv run python -m src.examples.example_two.q_computations # warning: slow
uv run python -m src.examples.example_two.fit_xgb
uv run python -m src.examples.example_two.example_two # warning: slow
```

You can run some benchmarking for the heat equation example with:

```bash
uv run python -m src.examples.example_two.pde_benchmarking # warning: slow
```

To run all tests, use `uv run pytest`. It should take around a minute to run the
first time you do so.

## Directory Structure

- The core functions are stored in the modules at the top level of
  [`src/`](src). 
- The examples are stored in [`src/examples/`](src/examples/).
- Unit and integrated tests appear in [`tests/`](tests). 
- Results from running the examples or certain tests are stored in
  [`data/`](data) and [`plots/`](plots).

## Create Custom Experiments

If you wish to easily run experiments using my existing examples, consult
[`tutorial.md`](tutorial.md). That document recaps how the constrained SCP
algorithm works, introduces each of the examples, and explains how to modify
them for your own purposes.


## License

[MIT](https://choosealicense.com/licenses/mit/)

## Issues

If you spot any bugs or have questions, feel free to raise a GitHub Issue or
contact me via email.
