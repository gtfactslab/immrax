import immrax as irx
from immrax.embedding import AuxVarEmbedding, TransformEmbedding
from immrax.utils import draw_iarray
from immrax.inclusion import interval
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Check lots of aux vars. With enough samples 
# Code up gradient descent steps in dual var 
# Read papers about contraction and stability

# Number of subdivisions of [0, pi] to make aux vars for
N = 9
aux_vars = jnp.array([[1.0, jnp.tan(n * jnp.pi / N)] for n in range(1, N)])
x0 = jnp.array([1.0, 0.0])
uncertainties = jnp.array([0.1, 0.1])
x0_int = irx.icentpert(x0, uncertainties)

fig, axs = plt.subplots(int(jnp.ceil(N / 3)), 3, figsize=(5, 5))
fig.suptitle("Harmonic Oscillator with Uncertainty")
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
axs[0].set_title("No aux vars")

# Plot refined trajectories
H = jnp.array([[1.0, 0.0], [0.0, 1.0]])
for i in range(len(aux_vars)):
    # Add new refinement
    print(f"Adding auxillary variable {aux_vars[i]}")
    H = jnp.append(H, jnp.array([aux_vars[i]]), axis=0)
    lifted_x0_int = interval(H) @ x0_int

    # Compute new refined trajectory
    auxsys = AuxVarEmbedding(osc, H, num_samples=1000)
    traj = auxsys.compute_trajectory(
        0.0,
        1.56,
        irx.i2ut(lifted_x0_int),
    )

    # Clean up and display results
    tfinite = jnp.where(jnp.isfinite(traj.ts))
    ts_clean = traj.ts[tfinite]
    ys_clean = traj.ys[tfinite]

    y_int = [irx.ut2i(y) for y in ys_clean]
    for timestep, bound in zip(ts_clean, y_int):
        draw_iarray(axs[i + 1], bound)
    axs[i+1].set_title(rf"$\theta = {i+1} \frac{{\pi}}{{{N}}}$")
    print(f"\tFinal bound:\n{irx.ut2i(ys_clean[-1])[:2]}")

plt.show()
