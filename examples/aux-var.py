import immrax as irx
from immrax.utils import I_refine, draw_iarray, null_space
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from immutabledict import immutabledict

x0 = irx.icentpert(jnp.array([1, 0, 1]), jnp.array([0.1, 0.1, 0.2]))

# Add auxiliary variable for interval refinement
H = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
N = null_space(H.T)
A = N.T
Hp = jnp.linalg.pinv(H) + A

# Test if columns of A are in null space of H
# print(jnp.all(jnp.isclose(A @ H, 0.0, atol=1e-5)))

IH = jax.jit(I_refine(A))
# print(I_refine(A)(irx.icentpert(jnp.array([1, 0, 1]), 0.1)))
# print(H.shape)
print(Hp @ jnp.array([1.0, 0.0, 1.0]))


class HarmOsc(irx.System):
    def __init__(self) -> None:
        self.evolution = "continuous"
        self.xlen = 2

    def f(self, t, x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = x.ravel()
        return jnp.array([-x2, x1])


# Create UT embedded system
# FIXME: mjacif doesn't work here, I think it should
olsys = HarmOsc()
liftsys = irx.LiftedSystem(olsys, H, Hp)
Fnat = jax.jit(irx.jacif(liftsys.f))
embsys = irx.ifemb(liftsys, Fnat)

# Compute trajectory in embedding
traj = embsys.compute_trajectory(
    0.0,
    1.56,
    irx.i2ut(x0),
    f_kwargs=immutabledict({"refine": IH}),
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
