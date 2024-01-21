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

## Installing JAX

Follow the instructions from the [JAX documentation](https://jax.readthedocs.io/en/latest/installation.html). For GPU support, the easiest will likely be to install the CUDA/CUDNN libraries using pip, instead of a local installation. 

For a full installation of CUDA into the `conda` environment using `pip`,
```shell
pip install --upgrade pip
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

If you just want CPU support, the install is much simpler. Just run
```shell
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```

## Installing `immrax`

For now, manually clone the Github repository and `pip install` it. We plan to release a stable version on PyPi soon.

```shell
git clone https://github.com/gtfactslab/immrax.git
cd immrax
pip install .
```

To test if the installation process worked, run the `compare.py` example.

```shell
cd examples
python compare.py
```

This should return the outputs of different inclusion functions as well as their runtimes.

## Installing `cyipopt` and `coinhsl` (optional)

If you would like to run the [pendulum optimal control example](Pendulum), you need to install IPOPT and the MA57 linear solver from HSL.

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
