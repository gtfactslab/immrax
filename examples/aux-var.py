from typing import List

import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, AbstractSolver, Solution
import matplotlib.pyplot as plt
import numpy as onp
from pypoman import plot_polygon
from scipy.spatial import HalfspaceIntersection


import immrax as irx
from immrax.embedding import AuxVarEmbedding, TransformEmbedding
from immrax.inclusion import Interval, interval, natif
from immrax.utils import draw_iarray, linprog_refine, run_times

# Code up gradient descent steps in dual var
# Read papers about contraction and stability

A: jax.Array = jnp.array([[0.0, -1.0], [1.0, 0.0]])  # row major
sim_len = 1.56


def linprog_traj(A, x0, H, t0, tf, dt=0.01):
    sim_len = int((tf - t0) / dt)
    Hp = jnp.linalg.pinv(H)

    def update(x: irx.Interval, *args) -> irx.Interval:
        sys_upd = lambda x: A @ x
        lifted_upd = lambda x: H @ sys_upd(Hp @ x)
        emb_upd = natif(lifted_upd)

        Fkwargs = lambda i, x: emb_upd(linprog_refine(H, collapsed_row=i)(x))

        n = H.shape[0]
        _x = x.lower
        x_ = x.upper

        # Computing F on the faces of the hyperrectangle
        _X = interval(
            jnp.tile(_x, (n, 1)), jnp.where(jnp.eye(n), _x, jnp.tile(x_, (n, 1)))
        )
        _E = jax.vmap(Fkwargs, in_axes=(0, 0))(jnp.arange(len(_X)), _X)

        X_ = interval(
            jnp.where(jnp.eye(n), x_, jnp.tile(_x, (n, 1))), jnp.tile(x_, (n, 1))
        )
        E_ = jax.vmap(Fkwargs, in_axes=(0, 0))(jnp.arange(len(X_)), X_)

        return irx.interval(jnp.diag(_E.lower), jnp.diag(E_.upper))

    def func(t, x, args): 
        return update(x)

    term = ODETerm(func)
    solver = Tsit5()
    # saveat = SaveAt(ts=jnp.linspace(t0, tf, sim_len, endpoint=False))
    saveat = SaveAt(t0=True, t1=True, steps=True)
    sol = diffeqsolve(term, solver, t0, tf, dt, x0, saveat=saveat)
    tfinite = jnp.where(jnp.isfinite(traj.ts))

    return sol.ys[tfinite]


class HarmOsc(irx.System):
    def __init__(self) -> None:
        self.evolution = "continuous"
        self.xlen = 2

    def f(self, t, x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = x.ravel()
        return jnp.array([-x2, x1])


# Number of subdivisions of [0, pi] to make aux vars for
# Certain values of this are not good choices, as they will generate angles theta=0 or theta=pi/2
# This will introduce dependence in the aux vars, causing problems with the JAX LP solver
N = 2
aux_vars = jnp.array(
    [
        [jnp.cos(n * jnp.pi / (N + 1)), jnp.sin(n * jnp.pi / (N + 1))]
        for n in range(1, N + 1)
    ]
)
x0_int = irx.icentpert(jnp.array([1.0, 0.0]), jnp.array([0.1, 0.1]))

# Trajectory of unrefined system
osc = HarmOsc()
embsys = TransformEmbedding(osc)
traj = embsys.compute_trajectory(
    0.0,
    sim_len,
    irx.i2ut(x0_int),
)

# Clean up and display results
tfinite = jnp.where(jnp.isfinite(traj.ts))
ts_clean = traj.ts[tfinite]
ys_clean = traj.ys[tfinite]

plt.rcParams.update({"text.usetex": True, "font.family": "CMU Serif", "font.size": 14})
plt.figure()

y_int = [irx.ut2i(y) for y in ys_clean]
for timestep, bound in zip(ts_clean, y_int):
    draw_iarray(plt.gca(), bound, alpha=0.4)
plt.gcf().suptitle("Harmonic Oscillator with Uncertainty (No Refinement)")

# Plot refined trajectories
fig, axs = plt.subplots(int(jnp.ceil(N / 3)), 3, figsize=(5, 5))
fig.suptitle("Harmonic Oscillator with Uncertainty (Sampled Refinement)")
axs = axs.reshape(-1)

fig_lp, axs_lp = plt.subplots(int(jnp.ceil(N / 3)), 3, figsize=(5, 5))
fig_lp.suptitle("Harmonic Oscillator with Uncertainty (LP Refinement)")
axs_lp = axs_lp.reshape(-1)

H = jnp.array([[1.0, 0.0], [0.0, 1.0]])
for i in range(len(aux_vars)):
    # Add new refinement
    print(f"Adding auxillary variable {aux_vars[i]}")
    H = jnp.append(H, jnp.array([aux_vars[i]]), axis=0)
    lifted_x0_int = interval(H) @ x0_int

    # Compute new refined trajectory
    auxsys = AuxVarEmbedding(osc, H, num_samples=10 ** (i + 1))
    traj, time = run_times(
        1,
        auxsys.compute_trajectory,
        0.0,
        sim_len,
        irx.i2ut(lifted_x0_int),
    )
    tfinite = jnp.where(jnp.isfinite(traj.ts))
    ts_clean = traj.ts[tfinite]
    ys_clean = traj.ys[tfinite]
    print(f"\tSample for {i+1} aux vars took: {time}")
    print(f"\tFinal bound: \n{irx.ut2i(ys_clean[-1])[:2]}")

    lp_traj, time = run_times(1, linprog_traj, A, lifted_x0_int, H, 0.0, sim_len)
    print(f"\tLinprog for {i+1} aux vars took: {time}")
    print(f"\tFinal bound: \n{lp_traj[-1][:2]}")

    # Clean up and display results
    y_int = [irx.ut2i(y) for y in ys_clean]
    plt.sca(axs[i])
    axs[i].set_title(rf"$\theta = {i+1} \frac{{\pi}}{{{N+1}}}$")
    for timestep, bound in zip(ts_clean, y_int):
        cons = onp.hstack(
            (
                onp.vstack((-H, H)),
                onp.concatenate((bound.lower, -bound.upper)).reshape(-1, 1),
            )
        )
        hs = HalfspaceIntersection(cons, bound.center[0:2])
        vertices = hs.intersections

        plot_polygon(vertices, fill=False, resize=True, color="tab:blue")

    plt.sca(axs_lp[i])
    axs_lp[i].set_title(rf"$\theta = {i+1} \frac{{\pi}}{{{N+1}}}$")
    for timestep in range(len(lp_traj)):
        bound = lp_traj[timestep]
        cons = onp.hstack(
            (
                onp.vstack((-H, H)),
                onp.concatenate((bound.lower, -bound.upper)).reshape(-1, 1),
            )
        )
        hs = HalfspaceIntersection(cons, bound.center[0:2])
        vertices = hs.intersections

        plot_polygon(vertices, fill=False, resize=True, color="tab:blue")

print("Plotting finished")
plt.show()
