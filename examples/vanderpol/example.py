from typing import Union
from jax import Array
from jaxtyping import Float, Integer
import sympy as sp
import immrax as irx
import jax.numpy as jnp
from matplotlib import pyplot as plt

# t, x1, y1, x2, y2, b = (x_vars := sp.symbols('t, x1, y1, x2, y2, b'))
# mu = 1
# f_eqn = [
#     y1,
#     mu*(1 - x1**2)*y1 + b*(x2 - x1) - x1,
#     y2,
#     mu*(1 - x2**2)*y1 + b*(x2 - x1) - x2,
#     0,
# ]

# def phi (x) :

t, x1, x2, u, w = (x_vars := sp.symbols('t, x1, x2, u, w'))

class vanderpol (irx.System) :
    def __init__(self, mu = 1.) -> None:
        self.evolution = 'continuous'
        self.xlen = 2
        self.mu = mu
    def f(self, t, x) :
        x1, x2 = x.ravel()
        return jnp.array([
            x2, -x1  - self.mu*x2*x1**2 + self.mu*x2
        ])

sys = vanderpol()
x0 = jnp.array([0.001,0.]) 
traj = sys.compute_trajectory(0., 100., x0, dt=0.05)

tfinite = jnp.isfinite(traj.ts)
print(tfinite)

plt.plot(traj.ys[tfinite,0], traj.ys[tfinite,1])

plt.figure()
plt.plot(traj.ts[tfinite], (traj.ys[tfinite,0]**2 + traj.ys[tfinite,1]**2))

plt.show()

