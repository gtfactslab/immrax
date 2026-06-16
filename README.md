# immrax

`immrax` is a tool for interval analysis and mixed monotone reachability analysis in JAX.

Inclusion function transformations are composable with existing JAX transformations, allowing the use of Automatic Differentiation to learn relationships between inputs and outputs, as well as parallelization and GPU capabilities for quick, accurate reachable set estimation.

For more information, please see the full [documentation](https://immrax.readthedocs.io).

## Installation

### System Dependencies

`immrax` depends on the library `pypoman`, which internally uses `pycddlib` as a wrapper around [the cdd library](https://people.inf.ethz.ch/fukudak/cdd_home/).  For this wrapper to function properly, you must install `cdd` to your system.  On Ubuntu, the relevant packages can be installed with

```bash
apt-get install -y libcdd-dev libgmp-dev
```

On Arch linux, you can use

```bash
pacman -S cddlib
```

### Package

`immrax` is available as a package on PyPI and can be installed with `pip`.

```shell
pip install immrax
```

We recommend you use a python virtual environment (e.g. through `pip`, `conda`, `uv`, etc.).

### Hardware Acceleration

`immrax` supports hardware acceleration through JAX.  You can install the `cuda12` or `cuda13` extras it supplies directly, or manually choose a different JAX accelerator. (Our development machines are all currently CUDA based; other methods are untested.)

```shell
pip install "jax[rocm7-local]"
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for further details.

## Development

Simply clone this repo to get started with `immrax` development.  We manage dependencies with [`uv`](https://docs.astral.sh/uv/); to create a virtual environment with everything you need run

```shell
uv sync
```

The above comments on [system dependencies](#system-dependencies) still apply.

### Examples

To test if the installation process worked, run the `compare.py` example.  This should return the outputs of different inclusion functions as well as their runtimes.

```shell
uv run examples/compare.py
```

The additional `examples` extra contains some dependencies needed for the more complex examples; be sure to also install it if you want to run the others.

### Other Tooling

We have a useful (though not yet comprehensive) test suite that can be run with

```shell
uv run pytest
```

Automatic linting and formatting is done with `ruff`

```shell
uv run ruff check # lint all project files, report only
uv run ruff format # autoformat all project files
```

This repo also ships a tracked git pre-commit hook that runs `ruff check --fix` and `ruff format` on staged Python files automatically. Enable it once per clone with

```shell
git config core.hooksPath .githooks
```

The hook is non-blocking: it applies safe fixes and formatting, warns if any unfixable lint issues remain, and lets the commit proceed.

## Citation

If you find this library useful, please cite our paper with the following bibtex entry.

```
@article{immrax,
title = {immrax: A Parallelizable and Differentiable Toolbox for Interval Analysis and Mixed Monotone Reachability in {JAX}},
journal = {IFAC-PapersOnLine},
volume = {58},
number = {11},
pages = {75-80},
year = {2024},
note = {8th IFAC Conference on Analysis and Design of Hybrid Systems ADHS 2024},
issn = {2405-8963},
doi = {https://doi.org/10.1016/j.ifacol.2024.07.428},
url = {https://www.sciencedirect.com/science/article/pii/S2405896324005275},
author = {Akash Harapanahalli and Saber Jafarpour and Samuel Coogan},
keywords = {Interval analysis, Reachability analysis, Automatic differentiation, Parallel computation, Computational tools, Optimal control, Robust control},
abstract = {We present an implementation of interval analysis and mixed monotone interval reachability analysis as function transforms in Python, fully composable with the computational framework JAX. The resulting toolbox inherits several key features from JAX, including computational efficiency through Just-In-Time Compilation, GPU acceleration for quick parallelized computations, and Automatic Differentiability We demonstrate the toolbox’s performance on several case studies, including a reachability problem on a vehicle model controlled by a neural network, and a robust closed-loop optimal control problem for a swinging pendulum.}
}
```
