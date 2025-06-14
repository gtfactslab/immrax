# Pendulum Optimal Control

## `Immrax` Set-Up

Please ensure you have read and completed the set-up instructions in the [main project README](README.md).

## Installing `cyipopt` and `coinhsl`

If you would like to run this example, you need to install IPOPT and the MA57 linear solver from HSL.

To use the MA57 solver, you'll first need to acquire a package from [HSL](https://www.hsl.rl.ac.uk/). While there are instructions [here](https://cyipopt.readthedocs.io/en/stable/install.html#conda-forge-binaries-with-hsl), we highly recommend to instead use [ThirdParty-HSL](https://github.com/coin-or-tools/ThirdParty-HSL) to install HSL globally.
Then, use a symbolic link to help the `conda` environment locate it.

```shell
ln -s /usr/local/lib/libcoinhsl.so $CONDA_PREFIX/lib/libcoinhsl.so
```

Finally, install `cyipopt` to your `immrax` conda environment (more instructions [here](https://cyipopt.readthedocs.io/en/stable/install.html)).

```shell
conda install -c conda-forge cyipopt
```

This command can take a while to fully resolve. 
