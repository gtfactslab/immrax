import pickle
from typing import Literal, Tuple
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import immrax as irx
from immrax.embedding import AuxVarEmbedding
from immrax.system import Trajectory
from immrax.utils import (
    angular_sweep,
    run_times,
    draw_trajectory_2d,
    draw_refined_trajectory_2d,
    gen_ics,
)


class HarmOsc(irx.System):
    def __init__(self) -> None:
        self.evolution = "continuous"
        self.xlen = 2
        self.name = "Harmonic Oscillator"

    def f(self, t, x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = x.ravel()
        return jnp.array([-x2, x1])


class VanDerPolOsc(irx.System):
    def __init__(self, mu: float = 1) -> None:
        self.evolution = "continuous"
        self.xlen = 2
        self.name = "Van der Pol Oscillator"
        self.mu = mu

    def f(self, t, x: jnp.ndarray) -> jnp.ndarray:
        x1, x2 = x.ravel()
        return jnp.array([self.mu * (x1 - 1 / 3 * x1**3 - x2), x1 / self.mu])


def angular_refined_trajectory(
    num_aux_vars: int, mode: Literal["sample", "linprog"], save: bool = False
) -> Tuple[Trajectory, jax.Array, Trajectory]:
    # Generate angular sweep aux vars
    # Odd num_aux_var is not a good choice, as it will generate angle theta=pi/2, which is redundant with the actual state vars
    aux_vars = angular_sweep(num_aux_vars)
    H = jnp.vstack([jnp.eye(2), aux_vars])
    lifted_x0_int = irx.interval(H) @ x0_int

    # Compute refined trajectory
    auxsys = AuxVarEmbedding(sys, H, mode=mode)
    print("Compiling...")
    start = time.time()
    get_traj = jax.jit(
        lambda t0, tf, x0: auxsys.compute_trajectory(t0, tf, x0, solver="euler"),
        backend="cpu",
    )
    get_traj(0.0, 0.01, irx.i2ut(lifted_x0_int))
    print(f"Compilation took: {time.time() - start:.4g}s")
    print("Compiled.\nComputing trajectory...")
    traj, comp_time = run_times(
        10,
        get_traj,
        0.0,
        sim_len,
        irx.i2ut(lifted_x0_int),
    )
    print(
        f"Computing trajectory with {mode} refinement for {num_aux_vars} aux vars took: {comp_time.mean():.4g} ± {comp_time.std():.4g}s"
    )

    ys_int = [irx.ut2i(y) for y in traj.ys]
    final_bound = ys_int[-1][2:]
    final_bound_size = (final_bound[0].upper - final_bound[0].lower) * (
        final_bound[1].upper - final_bound[1].lower
    )
    print(f"Final bound: \n{final_bound}, size: {final_bound_size}")

    if save:
        pickle.dump(ys_int, open(f"{mode}_traj_{num_aux_vars}.pkl", "wb"))

    mc_x0s = gen_ics(x0_int, 30)
    mc_traj = jax.vmap(
        lambda x0: sys.compute_trajectory(0, sim_len, x0, solver="euler"),
    )(mc_x0s)

    return traj, H, mc_traj


def plot_angular_refined_trajectory(traj: Trajectory, H: jax.Array):
    fig = plt.figure()
    # fig, axs = plt.subplots(int(jnp.ceil(N / 3)), 3, figsize=(5, 5))
    fig.suptitle(f"Reachable Sets of the {sys.name}")
    plt.gca().set_xlabel(r"$x_1$")
    plt.gca().set_ylabel(r"$x_2$")
    # axs = axs.reshape(-1)

    draw_refined_trajectory_2d(traj, H)


x0_int = irx.icentpert(jnp.array([1.0, 0.0]), jnp.array([0.1, 0.1]))
sim_len = 2 * jnp.pi

plt.rcParams.update({"text.usetex": True, "font.family": "CMU Serif", "font.size": 14})
# plt.figure()


# Trajectory of unrefined system
sys = VanDerPolOsc()  # Can use an arbitrary system here
embsys = irx.mjacemb(sys)
traj = embsys.compute_trajectory(
    0.0,
    sim_len,
    irx.i2ut(x0_int),
)
plt.gcf().suptitle(f"{sys.name} with Uncertainty (No Refinement)")
draw_trajectory_2d(traj)


traj_s, H, mc_traj = angular_refined_trajectory(6, "sample")
traj_lp, H, mc_traj = angular_refined_trajectory(6, "linprog")
plot_angular_refined_trajectory(traj_lp, H)
fig = plt.gcf()
x = mc_traj.ys[:, :, 0].T
y = mc_traj.ys[:, :, 1].T
plt.plot(x, y, alpha=0.5, color="gray", linewidth=0.5)

blue_rectangle = mpatches.Patch(
    edgecolor="tab:blue", facecolor="none", alpha=0.4, label="Reachable Set Bounds"
)
gray_line = mlines.Line2D(
    [], [], color="gray", alpha=0.5, label="Monte Carlo Trajectories"
)
plt.legend(handles=[blue_rectangle, gray_line], loc="lower left")

plt.show()
