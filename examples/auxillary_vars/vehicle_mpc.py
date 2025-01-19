import jax.numpy as jnp

import immrax as irx
from immrax.embedding import AuxVarEmbedding
from immrax.utils import draw_refined_trajectory_2d, angular_sweep

from casadi import MX, Function, Opti, pi, arctan, cos, sin, tan
import numpy as onp

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataclasses import dataclass


@dataclass
class Vehicle:
    lf: float = 1
    lr: float = 1


class VehicleSys(irx.System):
    def __init__(self, vehicle: Vehicle = Vehicle()):
        self.vehicle = vehicle
        self.evolution = "continuous"
        self.xlen = 4

    def f(self, t, x, u, w, x_nom):
        u_fdbk = jnp.array([0, 0])
        u_fdbk = jnp.array([0, jnp.atan(2 * jnp.tan(jnp.arcsin(x_nom[2] - x[2])))])
        u_fdbk = K*(x - x_nom)
        u1, u2 = u + u_fdbk + w
        _, _, psi, v = x.ravel()
        beta = jnp.atan(
            (self.vehicle.lr * jnp.tan(u2)) / (self.vehicle.lf + self.vehicle.lr)
        )
        return jnp.array(
            [
                v * jnp.cos(psi + beta),
                v * jnp.sin(psi + beta),
                v * jnp.sin(beta) / self.vehicle.lr,
                u1,
            ]
        )


class VehicleControl:
    def __init__(
        self, vehicle: Vehicle = Vehicle(), n_horizon=20, u_step=0.25, euler_steps=10
    ):
        self.vehicle = vehicle
        self.n_horizon = n_horizon
        self.u_step = u_step
        self.euler_steps = euler_steps

        x = MX.sym("x", 4, 1)  # [x, y, psi, v]
        psi = x[2]
        v = x[3]
        u = MX.sym("u", 2, 1)  # [turning, acceleration]
        u1 = u[0]
        u2 = u[1]

        xdot = MX(4, 1)
        beta = arctan((self.vehicle.lr * tan(u2)) / (self.vehicle.lf + self.vehicle.lr))
        xdot[0] = v * cos(psi + beta)
        xdot[1] = v * sin(psi + beta)
        xdot[2] = v * sin(beta) / self.vehicle.lr
        xdot[3] = u1

        f = Function("f", [x, u], [xdot], ["x", "u"], ["xdot"])

        N = self.n_horizon

        def euler_integ(x, u):
            step = self.u_step / self.euler_steps
            for t in onp.arange(0, self.u_step, step):
                x = x + step * f(x, u)
            return x

        euler_res = euler_integ(x, u)
        F = Function("F", [x, u], [euler_res], ["x", "u"], ["x_next"])

        self.opti = Opti()
        self.xx = self.opti.variable(4, N + 1)
        self.uu = self.opti.variable(2, N)
        self.x0 = self.opti.parameter(4, 1)
        self.slack = self.opti.variable(1, N)

        self.opti.subject_to(self.xx[:, 0] == self.x0)
        J = 0
        for n in range(N):
            # Dynamics constraints
            self.opti.subject_to(self.xx[:, n + 1] == F(self.xx[:, n], self.uu[:, n]))

            # Penalize state and control effort
            J += (
                self.xx[0, n] ** 2
                + self.xx[1, n] ** 2
                + 0.1 * self.uu[0, n] ** 2
                + 15 * self.uu[1, n] ** 2
            )
            if n > 0:
                # Penalize large control changes
                J += 5e-3 * (self.uu[0, n] - self.uu[0, n - 1]) ** 2 + 5 * (
                    self.uu[1, n] - self.uu[1, n - 1]
                )
            J += 1e5 * self.slack[0, n] ** 2

            # Obstacle constraints
            self.opti.subject_to(
                (self.xx[0, n] - 4) ** 2 + (self.xx[1, n] - 4) ** 2
                >= 3**2 - self.slack[0, n]
            )
            self.opti.subject_to(
                (self.xx[0, n] + 4) ** 2 + (self.xx[1, n] - 4) ** 2
                >= 3**2 - self.slack[0, n]
            )

        # Final state penalty
        J += (
            100 * self.xx[0, N] ** 2 + 100 * self.xx[1, N] ** 2 + 1 * self.xx[3, N] ** 2
        )
        self.opti.minimize(J)

        # Actuation constraints
        self.opti.subject_to(self.opti.bounded(-20, self.xx[0, :], 20))
        self.opti.subject_to(self.opti.bounded(-20, self.xx[1, :], 20))
        self.opti.subject_to(self.opti.bounded(-10, self.xx[3, :], 10))
        self.opti.subject_to(self.opti.bounded(-20, self.uu[0, :], 20))
        self.opti.subject_to(self.opti.bounded(-pi / 3, self.uu[1, :], pi / 3))

        self.opti.solver(
            "ipopt",
            {"print_time": 0},
            {
                # "linear_solver": "mumps",
                "print_level": 0,
                "sb": "yes",
                "max_iter": 100000,
            },
        )

    def u(self, t, x):
        self.opti.set_value(self.x0, x)
        for n in range(self.n_horizon + 1):
            self.opti.set_initial(self.xx[:, n], x)
        sol = self.opti.solve()
        # print(sol.value(self.slack))
        return sol.value(self.uu[:, 0]), sol.value(self.xx[:, 1])

    def make_trajectory(self, t0, tf, x0):
        ts = onp.arange(t0, tf, self.u_step)
        us = onp.zeros((len(ts), 2))
        xs = onp.zeros((len(ts) + 1, x0.shape[0]))
        xs[0] = x0

        for i, t in enumerate(ts):
            us[i], xs[i + 1] = self.u(t, xs[i])

        return ts, xs, us


# Compute optimal control
t0 = 0
tf = 2
dt = 0.05
x0 = onp.array([8, 7, -2 * pi / 3, 2])
try:
    with open("vehicle_mpc.pkl", "rb") as f:
        ts = onp.load(f)
        xs = onp.load(f)
        us = onp.load(f)
except FileNotFoundError:
    ts, xs, us = VehicleControl(u_step=dt).make_trajectory(t0, tf, x0)
    with open("vehicle_mpc.pkl", "wb") as f:
        onp.save(f, ts)
        onp.save(f, xs)
        onp.save(f, us)
xs = jnp.asarray(xs)
us = jnp.asarray(us)

# Intervals around optimal
num_pairings = 4
H = jnp.eye(4)
pairings = angular_sweep(num_pairings)
x_psi = (
    jnp.zeros((num_pairings, H.shape[1]))
    .at[:, 0]
    .set(pairings[:, 0])
    .at[:, 2]
    .set(pairings[:, 1])
)
y_psi = (
    jnp.zeros((num_pairings, H.shape[1]))
    .at[:, 1]
    .set(pairings[:, 0])
    .at[:, 2]
    .set(pairings[:, 1])
)
x_v = (
    jnp.zeros((num_pairings, H.shape[1]))
    .at[:, 0]
    .set(pairings[:, 0])
    .at[:, 3]
    .set(pairings[:, 1])
)
y_v = (
    jnp.zeros((num_pairings, H.shape[1]))
    .at[:, 1]
    .set(pairings[:, 0])
    .at[:, 3]
    .set(pairings[:, 1])
)
# H = jnp.vstack((H, x_psi, y_psi))
# H = jnp.vstack((H, x_v, y_v))
H = jnp.vstack((H, H))

esys = AuxVarEmbedding(VehicleSys(), H, mode="sample")
ix0 = irx.icentpert(jnp.asarray(x0), jnp.array([0.05, 0.05, 0.01, 0.01]))
lx0 = irx.interval(H) @ ix0


def get_control_input(t, x):
    return us[jnp.array(t / dt, dtype=int), :]


def get_disturbance(t, x):
    return irx.icentpert(jnp.zeros(2), jnp.array([1e-3, 1e-2]))


def get_x_nom(t, x):
    return xs[jnp.array(t / dt, dtype=int), :]


traj = esys.compute_trajectory(
    t0,
    tf,
    irx.i2ut(lx0),
    (get_control_input, get_disturbance, get_x_nom),
    dt=dt / 10,
    solver="euler",
)

fig, ax = plt.subplots()
ax.plot(xs[:, 0], xs[:, 1])  # MPC trajectory
circle = patches.Circle((4, 4), 3 - 1e-2, facecolor="salmon", label="Obstacle")
ax.add_patch(circle)  # Obstacle
draw_refined_trajectory_2d(traj, H)  # interval trajectory
ax.set_xlabel("x position")
ax.set_ylabel("y position")
ax.set_title("Vehicle Trajectory")
ax.grid(True)
plt.show()
