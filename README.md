# immrax

`immrax` is a tool for interval analysis and mixed monotone reachability analysis in JAX.

Inclusion function transformations are composable with existing JAX transformations, allowing the use of Automatic Differentiation to learn relationships between inputs and outputs, as well as parallelization and GPU capabilities for quick, accurate reachable set estimation.

For more information, please see the full [documentation](https://immrax.readthedocs.io).

# Installation

## Setting up a `conda` environment

We recommend installing JAX and `immrax` into a `conda` environment ([miniconda](https://docs.conda.io/projects/miniconda/en/latest/)).

```shell
conda create -n immrax python=3.11
conda activate immrax
```

## Installing immrax

A stable version of `immrax` is available on PyPi, and can be installed with `pip` as usual.

```shell
pip install --upgrade pip
pip install immrax
```

If you have cuda-enabled hardware you wish to utilize, please install the `cuda` optional dependency group.

```shell
pip install --upgrade pip
pip install immrax[cuda]
```

To test if the installation process worked, run the `compare.py` example.

```shell
cd examples
python compare.py
```

This should return the outputs of different inclusion functions as well as their runtimes.

## Installing `cyipopt` and `coinhsl` (optional)

If you would like to run the [pendulum optimal control example](examples/pendulum/pendulum.ipynb), you need to install IPOPT and the MA57 linear solver from HSL.

First, install `cyipopt` (more instructions [here](https://cyipopt.readthedocs.io/en/stable/install.html)).

```shell
conda install -c conda-forge cyipopt
```

This command can take a while to fully resolve.

To use the MA57 solver, you'll first need to acquire a package from [HSL](https://www.hsl.rl.ac.uk/). While there are instructions [here](https://cyipopt.readthedocs.io/en/stable/install.html#conda-forge-binaries-with-hsl), we highly recommend to instead use [ThirdParty-HSL](https://github.com/coin-or-tools/ThirdParty-HSL) to install HSL globally.
Then, use a symbolic link to help the `conda` environment locate it.

```shell
ln -s /usr/local/lib/libcoinhsl.so $CONDA_PREFIX/lib/libcoinhsl.so
```

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
