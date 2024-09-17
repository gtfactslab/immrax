import immrax as irx
from immrax.embedding import AuxVarEmbedding, TransformEmbedding
from immrax.utils import draw_iarray
from immrax.inclusion import interval
import jax.numpy as jnp
import numpy as onp
from scipy.spatial import HalfspaceIntersection
import matplotlib.pyplot as plt
from pypoman import plot_polygon

from immrax.utils import linprog_refine
from immrax.inclusion import natif
# Code up gradient descent steps in dual var 
# Read papers about contraction and stability

H = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
Hp = jnp.linalg.pinv(H)
A: jnp.array = jnp.array([[0.0, -1.0], [1.0, 0.0]]) # row major
x = interval(H) @ irx.icentpert(jnp.array([1.0, 0.0]), jnp.array([0.1, 0.1]))

dt = 0.01
sim_len = int(1.56 / dt)
bounds = [None] * sim_len
bounds[0] = x

def update(x: irx.Interval, *args) -> irx.Interval:
    sys_upd = lambda x: A @ x
    lifted_upd = lambda x: H @ sys_upd(Hp @ x)
    emb_upd = natif(lifted_upd)

    Fkwargs = lambda x: emb_upd(linprog_refine(H)(x))

    n = H.shape[0] 
    _x = x.lower
    x_ = x.upper

    # Computing F on the faces of the hyperrectangle
    _X = interval(
        jnp.tile(_x, (n, 1)), jnp.where(jnp.eye(n), _x, jnp.tile(x_, (n, 1)))
    )
    _E_lower = [None] * len(_X)
    for i in range(len(_X)):
        fx = Fkwargs(_X[i])
        _E_lower[i] = fx.lower
    _E = irx.interval(_E_lower, _E_lower)

    X_ = interval(
        jnp.where(jnp.eye(n), x_, jnp.tile(_x, (n, 1))), jnp.tile(x_, (n, 1))
    )
    E__upper = [None] * len(_X)
    for i in range(len(X_)):
        fx = Fkwargs(X_[i])
        E__upper[i] = fx.upper
    E_ = irx.interval(E__upper, E__upper)

    return irx.interval(jnp.diag(_E.lower), jnp.diag(E_.upper))

for i in range(1, sim_len):
    bounds[i] = bounds[i-1] + interval(dt) * update(bounds[i-1])

print(f"Final Bound:\n{bounds[-1][:2]}")

plt.figure()
ax = plt.gca()
for b in bounds:
    draw_iarray(ax, b, alpha=0.4)

plt.show()
quit()

# Number of subdivisions of [0, pi] to make aux vars for
# Certain values of this are not good choices, as they will generate angles theta=0 or theta=pi/2
N = 6
# aux_vars = jnp.array([[1.0, 1.0]])
aux_vars = jnp.array([[jnp.cos(n * jnp.pi / N), jnp.sin(n * jnp.pi / N)] for n in range(1, N)])
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
    draw_iarray(axs[0], bound, alpha=0.4)
axs[0].set_title("No aux vars")

# Plot refined trajectories
H = jnp.array([[1.0, 0.0], [0.0, 1.0]])
for i in range(len(aux_vars)):
    # Add new refinement
    print(f"Adding auxillary variable {aux_vars[i]}")
    H = jnp.append(H, jnp.array([aux_vars[i]]), axis=0)
    lifted_x0_int = interval(H) @ x0_int

    # Compute new refined trajectory
    auxsys = AuxVarEmbedding(osc, H, num_samples=10**(i+1))
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
    plt.sca(axs[i+1])
    axs[i+1].set_title(rf"$\theta = {i+1} \frac{{\pi}}{{{N}}}$")
    for timestep, bound in zip(ts_clean, y_int):
        cons = onp.hstack((onp.vstack((-H, H)), onp.concatenate((bound.lower, -bound.upper)).reshape(-1, 1)))
        hs = HalfspaceIntersection(cons, bound.center[0:2])
        vertices = hs.intersections

        plot_polygon(vertices, fill=False, resize=True, color="tab:blue")

    print(f"\tFinal bound:\n{irx.ut2i(ys_clean[-1])[:2]}")

plt.show()
