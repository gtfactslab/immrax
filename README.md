# immrax
Interval Analysis and Mixed Monotone Reachability in JAX

## Clone the repository

## Create a conda environment (recommended)
```shell
conda create -n immrax python=3.12
conda activate immrax
```

## Install Jax
Follow instructions from [https://jax.readthedocs.io/en/latest/installation.html](https://jax.readthedocs.io/en/latest/installation.html). 
For a local CUDA installation, this link may be helpful [https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202](https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202).

For a full installation of CUDA into the conda environment,
```shell
pip install --upgrade pip

# CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

With a local CUDA installation, 
```shell
pip install --upgrade pip

# Installs the wheel compatible with CUDA 12 and cuDNN 8.9 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Installs the wheel compatible with CUDA 11 and cuDNN 8.6 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Install jax_verify (optional, for immrax.neural module)


## Install cyipopt and HSL (optional, for pendulum planning example)
```shell
conda install -c conda-forge cyipopt
```
To use the MA57 linear solver, you'll need to install HSL. You'll need to acquire a package from [HSL](https://www.hsl.rl.ac.uk/).

While there are instructions [here](https://cyipopt.readthedocs.io/en/stable/install.html#conda-forge-binaries-with-hsl), we recommend to instead use [ThirdParty-HSL](https://github.com/coin-or-tools/ThirdParty-HSL) to install HSL globally. Then, use a symbolic link to the `conda` environment:
```shell
ln -s /usr/local/lib/libcoinhsl.so $CONDA_PREFIX/lib/libcoinhsl.so
```

## To use ssh instead of https for submodules (for pushing code)
```shell
cd jax_verify
git config url."ssh://git@".insteadOf https://
```

## To build the docs:
```shell
sphinx-build -M html docs/source docs/build
```
