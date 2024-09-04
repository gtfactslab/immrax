import immrax as irx
from immrax.embedding import AuxVarEmbedding, TransformEmbedding
from immrax.utils import draw_iarray
import jax.numpy as jnp
import matplotlib.pyplot as plt

# TODO: What is the uncertainty in x[2] in the initial condition?
x0_aux = irx.icentpert(jnp.array([1, 0, 1]), jnp.array([0.1, 0.1, 0.2]))
x0 = irx.icentpert(jnp.array([1, 0]), jnp.array([0.1, 0.1]))
H = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


class HarmOsc(irx.System):
    def __init__(self) -> None:
        self.evolution = "continuous"
        self.xlen = 2

    def f(self, t, x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = x.ravel()
        return jnp.array([-x2, x1])


# Create UT embedded system
osc = HarmOsc()
embsys = AuxVarEmbedding(osc, H)
embsys2 = TransformEmbedding(osc)

# Compute trajectory in embedding
traj = embsys.compute_trajectory(
    0.0,
    1.56,
    irx.i2ut(x0_aux),
)

# Clean up and display results
tfinite = jnp.where(jnp.isfinite(traj.ts))
ts_clean = traj.ts[tfinite]
ys_clean = traj.ys[tfinite]

ax = plt.gca()
y_int = [irx.ut2i(y) for y in ys_clean]
for timestep, bound in zip(ts_clean, y_int):
    draw_iarray(ax, bound)

plt.show()
