import sympy as sp
import immrax as irx
import jax.numpy as jnp
from matplotlib import pyplot as plt

def f(x) :
    # return 0.0*(x**2 - 1.)
    return 1

def g(x) :
    return x

class lienard (irx.System) :
    """A system of """
    def __init__ (self, f, g) :
        self.evolution = 'continuous'
        self.xlen = 2
        self._f = f
        self._g = g

    def f(self, t, x) :
        return jnp.array([
            x[1],
            -self._g(x[0]) - self._f(x[0])*x[1]
        ])

sys = lienard (f, g)
x0 = jnp.array([0.01,0.]) 
traj = sys.compute_trajectory(0., 100., x0, dt=0.05)

tfinite = jnp.isfinite(traj.ts)
print(tfinite)

plt.plot(traj.ys[tfinite,0], traj.ys[tfinite,1])
plt.show()
