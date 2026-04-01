This repository implements the eSCP algorithm given in [^1] for the purpose of running several experiments detailed in a paper[^2] that studies the continuity of this algorithm. The three modules 'src/example_{one, two, three}.py' reproduce the plots from [^2]. Details of the examples are explained in the paper. The algorithm is currently written for 2-dimensions only, and it is a future task to extend it to more dimensions.

[^1]: Shi, H., Yang, L., Chi, J., Butler, T., Wang, H., Bingham, D., & Estep, D. (2025). Nonparametric Bayesian Calibration of Computer Models. *arXiv preprint arXiv:2509.22597.*

[^2]: Prasadan A., Bingham, D., & Estep, D. (2026). Continuity of the Solution of a Non-Parametric Bayesian Statistical Calibration Procedure. *arXiv preprint arXiv:2603.20665.*

## Getting Started
 
1. Install **uv**, a package manager. [See instructions here](https://docs.astral.sh/uv/getting-started/installation/).

2. Clone the repository and navigate to it.

        git clone https://github.com/akprasadan/escp-continuity.git

        cd escp-continuity

3. Install the project and its dependencies.

        uv sync

## Usage

Now you can run modules with **uv** in a virtual environment, e.g., 

        uv run python -m src.example_one

        uv run python -m tests.check_pushforward

You can also run all unit tests.

        uv run pytest

## Directory Structure

- The core functions and examples are stored in `src/`. You only need to run each of the `src/example_{i}.py` files to obtain all the figures from the paper. Use the functions in `src/estimate_functions.py` to create your own examples. The rest of the modules in `src/` just contain helper functions.
- Necessary datasets will be stored in `data/` after running either `src/example_three.py` or `tests/check_concrete_pushforward.py`. These modules will download a concrete dataset and store the raw version in `data/raw_concrete.csv` and a processed version in  `data/processed_concrete.csv`. The original dataset is due to [^3] and can also be downloaded from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength).
- Results from running the examples or certain tests are stored in `plots/`.
- The code is tested in `tests/`. Three tests, `tests/check_pushforward.py`, `tests/check_concrete_pushforward.py`, `tests/test_probs_to_mesh.py` include visual tests and must be manually run. Their results are stored in `plots/tests/`.

[^3]: Yeh, I. C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. *Cement and Concrete research, 28*(12), 1797-1808.


## License

[MIT](https://choosealicense.com/licenses/mit/)


