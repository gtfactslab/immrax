(Getting Started)=

# Getting Started

## Dependencies

`immrax` depends on the library `pypoman`, which internally uses `pycddlib` as a wrapper around [the cdd library](https://people.inf.ethz.ch/fukudak/cdd_home/). For this wrapper to function properly, you must install `cdd` to your system. On Ubuntu, the relevant packages can be installed with 

```bash
apt-get install -y libcdd-dev libgmp-dev
```

On Arch linux, you can use 

```bash
pacman -S cddlib
```


## Installation

### Setting up a `conda` environment

We recommend installing JAX and `immrax` into a `conda` environment ([miniconda](https://docs.conda.io/projects/miniconda/en/latest/)).

```shell
conda create -n immrax python=3.12
conda activate immrax
```

### Installing immrax

For now, manually clone the Github repository and `pip install` it. We plan to release a stable version on PyPi soon.

```shell
git clone https://github.com/gtfactslab/immrax.git
cd immrax
pip install .
```


If you have cuda-enabled hardware you wish to utilize, please install the `cuda` optional dependency group.

```shell
...
pip install .[cuda]
```

To test if the installation process worked, run the `compare.py` example.

```shell
cd examples
python compare.py
```

This should return the outputs of different inclusion functions as well as their runtimes.


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
