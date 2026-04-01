This repository implements the eSCP algorithm given in [^1] for an up to 3-dimensional input space
and up to 2-dimensional output space.

[^1]: Shi, H., Yang, L., Chi, J., Butler, T., Wang, H., Bingham, D., & Estep, D. (2025). Nonparametric Bayesian Calibration of Computer Models. *arXiv preprint arXiv:2509.22597.*

## Getting Started
 
1. Install **uv**, a package manager. [See instructions here](https://docs.astral.sh/uv/getting-started/installation/).

2. Clone the repository and navigate to it.

        git clone https://github.com/akprasadan/escp-constrained.git

        cd escp-continuity

3. Install the project and its dependencies.

        uv sync

## Usage

Now you can run modules with **uv** in a virtual environment, e.g., 

        uv run python -m src.estimate_fucntions

You can also run all unit tests.

        uv run pytest

## Directory Structure

- The core functions and examples are stored in `src/`. 
- Results from running the examples or certain tests are stored in `plots/`.
- The code is tested in `tests/`. 


## License

[MIT](https://choosealicense.com/licenses/mit/)


