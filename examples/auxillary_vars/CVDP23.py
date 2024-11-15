import jax.numpy as jnp
import numpy as onp
from pypoman import plot_polygon
from scipy.spatial import HalfspaceIntersection
import matplotlib.pyplot as plt
import immrax as irx
from immrax.utils import angular_sweep, run_times
from immrax.embedding import AuxVarEmbedding


class CVDP(irx.System):
    def __init__(self, mu: float = 1) -> None:
        self.evolution = "continuous"
        self.xlen = 5
        self.mu = mu

    def f(self, t, x: jnp.ndarray) -> jnp.ndarray:
        x1, y1, x2, y2, b = x.ravel()
        return jnp.array(
            [
                y1,
                self.mu * (1 - x1**2) * y1 + b * (x2 - x1) - x1,
                y2,
                self.mu * (1 - x2**2) * y2 - b * (x2 - x1) - x2,
                0,
            ]
        )


x0 = irx.interval(
    jnp.array([1.25, 2.35, 1.25, 2.35, 1]), jnp.array([1.55, 2.45, 1.55, 2.45, 3])
)
N = 10
sweep = angular_sweep(N)
couplings = [(0, 1), (2, 3)]
H = jnp.eye(5)
for coupling in couplings:
    permuted_sweep = jnp.zeros([N, len(x0)])
    for var, idx in enumerate(coupling):
        permuted_sweep = permuted_sweep.at[:, idx].set(sweep[:, var])
    H = jnp.vstack([H, permuted_sweep])
# H = jnp.vstack(
#     [
#         H,
#         jnp.array(
#             [
#                 [1, 1, 0, 0, 0],
#                 [1, -1, 0, 0, 0],
#                 [0, 0, 1, 1, 0],
#                 [0, 0, 1, -1, 0],
#                 [1, 1, 1, 0, 1],
#                 [1, 1, -1, 0, 1],
#                 [1, 0, 1, 1, -1],
#                 [1, 0, 1, -1, -1],
#             ]
#         ),
#     ]
# )

x0_lifted = irx.interval(H) @ x0
t0 = 0.0
tf = 2.0

sys = CVDP()
embsys = AuxVarEmbedding(sys, H)
print("Compiling...")
traj = embsys.compute_trajectory(t0, 0.01, irx.i2ut(x0_lifted))
print("Compiled.\nComputing trajectory...")
traj, time = run_times(1, embsys.compute_trajectory, t0, tf, irx.i2ut(x0_lifted))
print(
    f"Computing trajectory took {time.item():.4g} s, {((tf - t0)/0.01/time).item():.4g} it/s"
)
ys_int = [irx.ut2i(y) for y in traj.ys]
print(f"Final bound:\n{ys_int[-1][:5]}")

# Plot the trajectory
axs = []
for var_pair in range(2):
    plt.figure()
    plt.axhline(y=2.75, color="red", linestyle="--")
    plt.xlabel(f"x{var_pair+1}")
    plt.ylabel(f"y{var_pair+1}")
    axs.append(plt.gca())

for bound in ys_int:
    cons = onp.hstack(
        (
            onp.vstack((-H, H)),
            onp.concatenate((bound.lower, -bound.upper)).reshape(-1, 1),
        )
    )
    hs = HalfspaceIntersection(cons, bound.center[0:5])
    vertices = hs.intersections

    plt.sca(axs[0])
    plot_polygon(vertices[:, 0:2], fill=False, resize=True, color="tab:blue")
    plt.sca(axs[1])
    plot_polygon(vertices[:, 2:4], fill=False, resize=True, color="tab:blue")

plt.show()
