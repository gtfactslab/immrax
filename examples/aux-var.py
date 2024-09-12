import immrax as irx
from immrax.embedding import AuxVarEmbedding, TransformEmbedding
from immrax.inclusion import interval
from immrax.utils import draw_iarray
import jax 
import jax.numpy as jnp
import matplotlib.pyplot as plt


aux_vars = jnp.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
x0 = jnp.array([1.0, 0.0])
uncertainties = jnp.array([0.1, 0.1])
x0_int = irx.icentpert(x0, uncertainties)

fig, axs = plt.subplots(2, 3, figsize=(5, 5))
axs = axs.reshape(-1)


class HarmOsc(irx.System):
    def __init__(self) -> None:
        self.evolution = "continuous"
        self.xlen = 2

    def f(self, t, x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = x.ravel()
        return jnp.array([-x2, x1])


# Trajectory of unrefined system
osc = HarmOsc()
embsys = TransformEmbedding(osc)
traj = embsys.compute_trajectory(
    0.0,
    1.56,
    irx.i2ut(x0_int),
)

# Clean up and display results
tfinite = jnp.where(jnp.isfinite(traj.ts))
ts_clean = traj.ts[tfinite]
ys_clean = traj.ys[tfinite]

y_int = [irx.ut2i(y) for y in ys_clean]
for timestep, bound in zip(ts_clean, y_int):
    draw_iarray(axs[0], bound)

# Plot refined trajectories
x0_aux = x0
H = jnp.array([[1.0, 0.0], [0.0, 1.0]])
for i in range(len(aux_vars)):
    # Add new refinement
    print(f"Adding auxillary variable {aux_vars[i]}")
    x0_aux = jnp.append(x0_aux, jnp.dot(x0, aux_vars[i]))
    H = jnp.append(H, jnp.array([aux_vars[i]]), axis=0)
    uncertainties = jnp.append(
        uncertainties, 0.2
    )  # HACK: what should uncertainty be here?
    x0_aux_int = irx.icentpert(x0_aux, uncertainties)

    # Compute new refined trajectory
    auxsys = AuxVarEmbedding(osc, H, num_samples=30)
    traj = auxsys.compute_trajectory(
        0.0,
        1.56,
        irx.i2ut(x0_aux_int),
    )

    # Clean up and display results
    tfinite = jnp.where(jnp.isfinite(traj.ts))
    ts_clean = traj.ts[tfinite]
    ys_clean = traj.ys[tfinite]

    y_int = [irx.ut2i(y) for y in ys_clean]
    for timestep, bound in zip(ts_clean, y_int):
        draw_iarray(axs[i + 1], bound)

plt.show()
