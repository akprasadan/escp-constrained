The following is a brief description of how the constrained SCP works, then what
the included examples do, and how to create your own custom experiment.

# Intro to the Constrained SCP

In a constrained SCP experiment, we start with some unknown trial-generating
distribution (TGD) $P_{\Lambda\times U}^t$ on a set $\Lambda\times U$, where the
TGD has a known marginal $P_U^t$ in $U$. Note that the TGD is not required to be
an independent coupling. Also, U may be more than one dimension. This TGD is
used to generate data $q_i=Q(\lambda_i, u_i)$ for some computer model mapping
$Q\colon\Lambda\times U\to\mathcal{D}\subset\mathbb{R}^m$. We then use the
$\lbrace q_i\rbrace$ and information about $P_U^t$ to generate a distribution
$\hat P_{\Lambda\times U}$ with the same pushforward as the TGD. A prior $P_p$
on $\Lambda \times U$ assists this procedure, where the marginal of $P_p$ in $U$
is again $P_U^t$. The method relies on an *augmented* map $\tilde
Q\colon\Lambda\times U\to\mathcal{D}\times U$, where we stack $Q(\lambda, u)$
and $u$. This ensures the solution has the right prior $P_U^t$ on $U$. 

## Paired Case

Suppose we have access to pairs $\lbrace(q_i, u_i)\rbrace$ where
$q_i=Q(\lambda_i, u_i)$ for $(\lambda_i, u_i)\sim P_{\Lambda\times U}^t$. Here
the $u_i$ need not be one-dimensional.

1. Draw $N$ pairs $\lbrace(\lambda_i, u_i)\rbrace$ from $P_{\Lambda \times
   U}^t$.
2. Compute $N$ observations of $q_i = Q(\lambda_i, u_i)$. Discard
   $\lbrace\lambda_i\rbrace$ for experimental purposes, though retain it for
   plotting/diagnostics. Keep the $\lbrace u_i\rbrace$.
3. Run the SCP using the augmented map $Q$ and these pairs $\lbrace(q_i,
   u_i)\rbrace$ which have distribution from the pushforward $\tilde Q
   P_{\Lambda\times U}^t$. Use the independent coupling of the priors on
   $\Lambda$ and $U$ as the prior for the SCP.
4. Marginalize out $U$ in the SCP solution to get an estimated distribution over
   $\Lambda$.

## Unpaired Case

Suppose now we have data $\lbrace q_i\rbrace$ where $q_i=Q(\lambda_i,u_i)$ for
$(\lambda_i, u_i)\sim P_{\Lambda\times U}^t$. We also can draw $u_j\sim
P_{U}^t$. But we do not have access to pairs $\lbrace(q_i, u_i)\rbrace$ as in
the paired case. Thus, we now rely on optimal-transport techniques, namely,
Sinkhorn's algorithm, to generate paired $\lbrace(q_k, u_k)\rbrace$ drawn from
an estimate of the joint distribution of $(Q(\lambda_k, u_k), u_k)$. For the
purpose of this paper and code, we will assume that $U$ is one-dimensional for
the unpaired case.

1. Draw $N$ pairs $\lbrace(\lambda_i, u_i)\rbrace$ from $P_{\Lambda \times
   U}^t$.
2. Compute $N$ observations of $q_i = Q(\lambda_i, u_i)$. Discard $\lbrace
   (\lambda_i, u_i) \rbrace$ for experimental purposes, though retain it for
   plotting/diagnostics.
3. Draw $N'$ independent copies $\lbrace u_j\rbrace$ from $P_U^t$.
4. Using the (unpaired) $\lbrace q_i\rbrace$ and $\lbrace u_j\rbrace$ data, run
   the Sinkhorn algorithm to obtain a joint distribution $\hat\Pi$ on
   $\mathcal{D} \times U$. The Sinkhorn algorithm requires use of a prior on
   $\Lambda$ in order to compute the cost function, as well as a hyper-parameter
   $\varepsilon$.
5. Sample $N''$ pairs $\lbrace(q_k, u_k)\rbrace$ from this joint distribution
   $\hat\Pi$.
6. Run the SCP using the augmented map $Q$ and the $N''$ pairs $\lbrace(q_k,
   u_k)\rbrace\sim\hat\Pi$. Use the independent coupling of the priors on
   $\Lambda$ and $U$ as the prior for the SCP.
7. Marginalize out $U$ in the SCP solution to get an estimated distribution over
   $\Lambda$.


## Unconstrained Case
For comparison purposes, we explain how an unconstrained SCP would work, i.e.,
not using the methods of this paper but only an earlier methodology. The main
difference is we do not use an augmented map $Q$, and only invert on data
$\lbrace q_i\rbrace$. The marginal $P_U^t$ may not be preserved in the solution.
We do not require $U$ to be one-dimensional.

1. Draw $N$ pairs $\lbrace(\lambda_i, u_i)\rbrace$ from $P_{\Lambda \times
   U}^t$.
2. Compute $N$ observations of $q_i = Q(\lambda_i, u_i)$. Discard
   $\lbrace(\lambda_i, u_i)\rbrace$ for experimental purposes, though retain it
   for plotting/diagnostics.
3. Run the SCP using the non-augmented map $Q$ and the data $\lbrace
   q_i\rbrace$. Use the independent coupling of the priors on $\Lambda$ and $U$
   as the prior for the SCP.
4. Marginalize out $U$ in the SCP solution to get an estimated distribution over
   $\Lambda$.

# Examples

Each of the following examples uses hyper-rectangles as the domain
$\Lambda\times U$. The $\Lambda$ space will always be 2 or 3 dimensional, and
$U$ will always be 1-dimensional. $Q$ will always be a real-valued map on
$\Lambda\times U$. The specific choices will be documented in the corresponding
module. Example one and two run both paired and unpaired scenarios for the
$\lbrace(q_i, u_i)\rbrace$ data. The unconstrained example also runs the
unpaired scenario in addition to the unconstrained algorithm.

## Example One

We consider a simple quadratic $Q$ with added noise, $Q(\lambda_1,
\lambda_2,u)=\lambda_1^2+\lambda_2^2+u$. The TGD will be the coupling of
independent Beta distributions on both $\Lambda$ and $U$.

The following plots are created:
- The actual joint pushforward on $\mathcal{D} \times U$ with some samples from
  it
- The Sinkhorn solution $\hat\Pi$ on $\mathcal{D} \times U$ with some samples
  from it
- The paired SCP solution for $\Lambda$ ($U$ marginalized out), with projected
  samples from  the TGD
- The unpaired SCP solution (which uses Sinkhorn) for $\Lambda$ ($U$
  marginalized out), with projected samples from the TGD
- A stacked heatmap that visualizes the unpaired SCP solution for $\Lambda
  \times U$, a 4D visual

Run it with:
```bash
uv run python -m src.examples.example_one.example_one
```

## Example Two

This experiment is slightly different because it uses a 3D $\Lambda$, leading to
different plots. Moreover, it requires some extra steps because we use an
emulator for $Q$. But the set-up is quite similar, and the main difference is a
different plotting function gets called. 

We set $Q$ to be an XGBoost trained model that emulates the steady-state
solution at a fixed point for a thin-plate 2D convection-diffusion heat equation
with a stochastic heat source term and some variable Dirichlet conditions. Our
$\Lambda$ is a pair of convection parameters and a single diffusion parameter.
The $U$ is the vertical position of the heat source. We run both a paired and
unpaired constrained SCP. However, we do not use the surrogate model for the
synthetic data generation; only in the subsequent calls of $Q$ inside the
algorithm. That way, the inversion is on 'genuine' data.


The following plots are created:
- A 2 x 3 set of plots containing the unpaired and paired scenario constrained
  SCP solution marginalized over each possible pairing of $\Lambda$ dimensions,
  i.e., 3 different plots per type of SCP method. A subset of TGD points are
  included (projected onto each corresponding pair of $\Lambda$ axes).
- The $\mathcal{D}$ and $U$ marginals (from TGD, Sinkhorn, paired SCP solution,
  and unpaired SCP solution), the $\mathcal{D} \times U$ true joint distribution
  (TGD pushforward), and the $\mathcal{D}\times U$ joint Sinkhorn estimate.
  Either joint plot shows samples from the corresponding distribution.
- The marginal 3D paired and unpaired SCP solutions on $\Lambda$ (no $U$) as a
  stacked heatmap.

```bash
uv run python -m src.examples.example_two.q_computations # warning: slow
uv run python -m src.examples.example_two.fit_xgb
uv run python -m src.examples.example_two.example_two # warning: slow
```

To generate some examples PDE solutions:

```bash
uv run python -m src.examples.example_two.pde_plots
```

## Unconstrained Example

Set $Q(\lambda_1,\lambda_2,u)=\lambda_1^2+\lambda_2^2+u^2$. Use a mixture of
diagonal tri-variate Gaussians as the TGD on $\Lambda\times U$. We run both the
unpaired experiment and the fully unconstrained algorithm. A plot is created
showing the $U$ marginals of the TGD and both SCP methods, the $\mathcal{D}$
pushforward of the TGD and both SCPs, and then both of the SCP solutions on
$\Lambda$ with $U$ marginalized out.

Run it with
```bash
uv run python -m src.examples.example_unconstrained.example_unconstrained
```

# Create a custom example

We will go through the creation of an example experiment with 2-dimensional
$\Lambda$, one-dimensional $U$, and Gaussian TGDs. The logic is similar if
$\Lambda$ is 3-dimensional, we'll just swap out a plotting function later
(explained where relevant). An unconstrained example is a bit more complicated
but also briefly explained.

Create a file `src/examples/custom_example.py`. The following imports are
required. We'll assume you use the plotting function for example one.

```python
import numpy as np

from src.examples.data_generation_utilities import (
    Distributions,
    Domain,
    Experiment,
    PlotOptions,
    generate_example_data,
)
from src.examples.example_one.plotting_utilities import create_example_one_plot

## If doing a 3D Lambda experiment:
# from src.examples.example_two.plotting_utilities import create_example_two_plot

## If doing an unconstrained experiment
# from src.examples.example_unconstrained.plotting_utilities import (
#    Experiment_Uncon,
#    create_uncon_example_plots,
#    generate_example_data_uncon,
# )
```

Define a computer model $Q$ and also set a seed.

```python
def Q(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return x + y + z


SEED = 42
```

Create a function that instantiates 4 dataclasses storing experimental
configuration settings:

- `Domain`: specifying the box corresponding to $\Lambda$ and $U$. You just need
  to pass in a vector of minimums and maximums for each of $\Lambda$ and $U$,
  and a @dataclass property will compute the combined min/max vectors for
  $\Lambda \times U$.
- `Distributions`: specifying the TGD and prior. You must specify a joint TGD on
  $\Lambda\times U$, a prior on $\Lambda$ only, and a prior on $U$ only. Another
  property will construct the resulting prior on $\Lambda \times U$ formed by
  the independent coupling. Importantly, the prior on $U$ must match the
  marginal prior of the joint TGD, otherwise some of the example code will be
  incorrect. Some wrappers for various distributions are available in
  [`src/sampler_functions.py`](src/sampler_functions.py), including mixtures of
  multivariate Gaussians or uniform distributions on hyper-rectangles.
- `Experiment`: Configuration settings for both paired and unpaired constrained
  SCP experiments. 
    - Use caution with `SCP_lambda_grid_res` as this scales exponentially with
      the dimension. Also, `monte_carlo_size` may slow down the Sinkhorn
      estimation step. 
    - For the heat equation example, we use a surrogate Q for the algorithm but
      non-surrogate Q for the initial data generation. Thus, pass in an
      additional parameter `non_surrogate_Q` in the `Experiment` dataclass. In
      our example, we set `non_surrogate_Q=heat_equation_w_source_at_pos`, where
      we call the original heat equation solver used to train the XGBoost model. 
    - If you are doing an unconstrained SCP experiment, you will also use the
      `Experiment_Uncon` dataclass defined in
      [`src/examples/example_unconstrained/plotting_utilities.py`](src/examples/example_unconstrained/plotting_utilities.py).
      The other dataclasses are still fine for this setting. 
    - You may have to adjust the regularization term here if you get warnings
      later from the Python Optimal Transport (POT) library about convergence.
- `PlotOptions`: Settings for plot-use and file-naming. The `folder_name`
  attribute will ensure the data and plot outputs of the example get stored in
  their own appropriately named directory `example_{folder_name}/`. The
  `expt_name` attribute sets the actual file-name in `example_{folder_name}/`.

```python
def gaussian_experiment() -> tuple[Domain, Distributions, Experiment, PlotOptions]:
    """Run an experiment with Gaussian TGDs."""

    lambda_mins, lambda_maxes = np.array([-10, -10]), np.array([10, 10])
    U_mins, U_maxes = np.array([-10]), np.array([10])
    domain = Domain(lambda_mins, lambda_maxes, U_mins, U_maxes)

    tgd_mean = np.array([1, 1, -1])
    tgd_cov = np.diag([1, 9, 4])

    def joint_tgd(rng: np.random.Generator, J: int) -> np.ndarray:
        return rng.multivariate_normal(tgd_mean, tgd_cov, J)

    def lambda_prior(rng: np.random.Generator, J: int) -> np.ndarray:
        dim = len(lambda_mins)
        return rng.uniform(lambda_mins, lambda_maxes, size=(J, dim))

    # This MUST match the joint_tgd marginal in U
    def u_prior(rng: np.random.Generator, J: int) -> np.ndarray:
        return rng.normal(-1, 2, J)

    distributions = Distributions(
        joint_tgd=joint_tgd, lambda_prior=lambda_prior, u_prior=u_prior
    )

    expt = Experiment(
        Q=Q,
        num_inputs=50_000,
        num_indep_u_inputs=50_000,
        sinkhorn_grid_res_U=30,
        sinkhorn_grid_res_D=30,
        sinkhorn_reg=1,
        monte_carlo_size=15_000,
        num_joint_pairs=50_000,
        SCP_prior_count=100_000,
        SCP_lambda_grid_res=20,
        SCP_D_grid_res=30,
    )

    plot_options = PlotOptions(
        figsize=(10, 10),
        dpi=720,
        scatter_count=250,
        expt_name=rf"tutorial_gaussian",
        D_min=0,
        D_max=60,
        folder_name="tutorial",
    )
    
    return domain, distributions, expt, plot_options
```

Now, we actually run the experiment. We'll call a function
[`src/examples/data_generation_utilities.py::generate_example_data`](src/examples/data_generation_utilities.py)
that populates a folder in [`data/`](data) with the results, and then
[`src/examples/example_one/plotting_utilities.py::create_example_one_plot`](src/examples/example_one/plotting_utilities.py)
that populates a folder in [`plots/`](plots) with figures. In both
[`data/`](data) and [`plots/`](plots), look for the directory
`example_{folder_name}/` where `folder_name` was specified by `PlotOptions`.

The data generation function includes an argument for whether you want to run
the paired-case constrained SCP or not. The unpaired case is always run. If you
want to do an example with 3-dimensional $\Lambda$, then you will instead want
to use a plotting function from
[`src/examples/example_two/plotting_utilities.py::create_example_two_plot`](src/examples/example_two/plotting_utilities.py),
but the data generation function is the same.

```python
if __name__ == "__main__":
    domain, distributions, expt, plot_options = gaussian_experiment()

    generate_example_data(
        domain, distributions, expt, plot_options, seed=SEED, paired=True
    )

    create_example_one_plot(domain, expt, plot_options)

    ## For 3D Lambda, use this function instead
    # create_example_two_plot(domain, expt, plot_options)
```

To repeat my included unconstrained experiment, slightly more work is needed. In
the dataclass creation function, define a single `Domain` and `Distributions`
dataclass, define an `Experiment` dataclass, an `Experiment_Uncon`dataclass, and
make a `PlotOptions` class for both the unconstrained and constrained setting.
Make sure the all of the `Experiment_Uncon` attributes match the corresponding
entry in `Experiment` so that the experiment is comparable. The `PlotOptions`
attributes for both constrained/unconstrained will be identical except for the
`expt_name`. The `folder_name` is set to `unconstrained` for both `PlotOptions`
instances, meaning a directory called `example_unconstrained` will be made in
each of [`data/`](data) and [`plots/`](plots) to store the results.

```python
def make_unconstrained_experiment() -> tuple[
    Domain, Distributions, Experiment_Uncon, Experiment, PlotOptions, PlotOptions
]:
    # Not shown: define necessary min/max vectors

    domain = Domain(...)

    # Not shown: define TGD and priors

    distributions = Distributions(...)

    expt_uncon = Experiment_Uncon(
        Q=Q_example_uncon,
        num_inputs=20_000,
        SCP_prior_count=250_000,
        SCP_lambda_grid_res=30,
        SCP_D_grid_res=50,
    )

    # Note the expt_uncon attributes match the corresponding subset of expt_con ones
    expt_con = Experiment(
        Q=Q_example_uncon,
        num_inputs=20_000,
        num_indep_u_inputs=150_000,
        sinkhorn_grid_res_D=30,
        sinkhorn_grid_res_U=30,
        sinkhorn_reg=10,
        monte_carlo_size=15_000,
        num_joint_pairs=200_000,
        SCP_prior_count=250_000,
        SCP_lambda_grid_res=30,
        SCP_D_grid_res=50,
    )

    plot_options_uncon = PlotOptions(
        figsize=(6, 6),
        dpi=720,
        scatter_count=150,
        expt_name=r"unconstrained_gaussian_mixture",
        D_min=-1,
        D_max=200,
        folder_name="unconstrained",
    )

    # Just change the expt_name
    plot_options_con = replace(plot_options_uncon, expt_name=rf"constrained")

    return (
        domain,
        distributions,
        expt_uncon,
        expt_con,
        plot_options_uncon,
        plot_options_con,
    )
```

Now run the experiment as before. You'll need to also call
`generate_example_data_uncon` and `create_uncon_example_plots` from
[`src/examples/example_unconstrained/plotting_utilities.py`](src/examples/example_unconstrained/plotting_utilities.py),
in addition to the data generation utility function from earlier,
[`src/examples/data_generation_utilities.py::generate_example_data`](src/examples/data_generation_utilities.py).

```python
if __name__ == "__main__":
    (
        domain,
        distributions,
        expt_uncon,
        expt_con,
        plot_options_uncon,
        plot_options_con,
    ) = make_unconstrained_experiment()

    # Get unconstrained SCP experimental data.
    generate_example_data_uncon(
        domain, distributions, expt_uncon, plot_options_uncon, seed=SEED
    )
    # Get unpaired constrained SCP experimental data, which uses Sinkhorn. Skip the paired data case.
    generate_example_data(
        domain, distributions, expt_con, plot_options_con, seed=SEED, paired=False
    )

    create_uncon_example_plots(domain, expt_uncon, plot_options_uncon, plot_options_con)
```

# Further Plot Customization

The main functions above that create the plots, `create_example_one_plot`,
`create_example_two_plot`, and `create_uncon_example_plots` rely on many
plotting helper functions (mostly in
[`src/examples/plotting_utilities.py`](src/examples/plotting_utilities.py)).
They usually have a subset of the following workflow:

1. Load in the data already created by a function like `generate_example_data`
2. Generate the diagnostic plots (all the plots focused on $\mathcal{D}$ and
   $U$, that is)
3. Generate the 2D heatmaps
4. Generate 3D stacked heatmaps

For item 2, for example, it instantiates a Matplotlib Figure and then calls
`create_U_marginals`, `create_D_marginals`, `create_true_pushforward_plot`, and
`create_D_U_sinkhorn_plot`. The first of these is a helper that takes in an Axes
and relevant data, and then plots the $U$ marginal of the TGD, Sinkhorn
solution, and both paired and unpaired SCP solutions. 

If you want to modify which plots get generated at all: just delete the
corresponding figure creation in the main plotting function. If you want to
change what plots get shown in which figures, just rearrange when each helper
gets called. If you want to change the actual plotting choices made in each
sub-plot, modify the helper function itself, e.g., `create_U_marginals`. If you
need to make new plots entirely, you may have to modify what gets saved in
`generate_example_data`. For example, I don't save the prior-samples in the
importance sampling, but you can certainly save it and then load it when making
the plots.

If the desired changes are more extensive than that, I suggest ignoring my
example code entirely and writing it from scratch. You can rely on the core
modules at the top level of [`src/`](src). For example, you can just directly
call `augmented_scp` in
[`src/sinkhorn_functions.py`](src/sinkhorn_functions.py), which returns the SCP
solutions, and make your plot from that. My examples, in contrast to the core
modules, are written for the purpose of a paper I am writing, not use by others.

# Higher dimensional $U$

If you want to run an experiment with higher dimensional $U$, you can do so in
the paired case constrained SCP or fully unconstrained SCP. For a constrained
SCP, you will need to augment the data. Do not directly use the aforementioned
`augmented_scp`, which is designed for univariate $U$ and $\mathcal{D}$.
Instead, directly invoke the `empirical_scp` function in
[`src/estimate_functions`](src/estimate_functions.py) with the augmented $Q$.
The augmentation now adds a vector of $U$ values rather than just a single
number. Use the following recipe to augment an existing $Q$.

```python
def Q(a, b, ..., z) -> np.ndarray:
    ...

def augmented_Q(a, b, ..., y, z, u_1, u_2, ..., u_k) -> np.ndarray:
    q = Q(a, b, ..., z)
    return np.vstack((q, u_1, ..., u_k))
```

The input data you feed into the `empirical_scp` will be a $(N, d + k)$ numpy
array, where $d$ is the dimension of $\mathcal{D}$ and $k$ is the dimension of
$U$.