import jax
import jax.numpy as jnp
import pytest
from diffrax import Solution
from immrax import System, Trajectory

A = jnp.array([[0, 1], [0, 0]])
B = jnp.array([[0], [1]])


class LinearSys(System):
    def __init__(self) -> None:
        self.evolution = "continuous"
        self.xlen = 2
        self.name = "Double Integrator"

    def f(self, t, x: jax.Array, u: jax.Array) -> jax.Array:
        return A @ x + B @ u


class HarmOsc(System):
    def __init__(self) -> None:
        self.evolution = "continuous"
        self.xlen = 2
        self.name = "Harmonic Oscillator"

    def f(self, t, x: jax.Array) -> jax.Array:
        x1, x2 = x.ravel()
        return jnp.array([-x2, x1])


class VanDerPolOsc(System):
    def __init__(self, mu: float = 1) -> None:
        self.evolution = "continuous"
        self.xlen = 2
        self.name = "Van der Pol Oscillator"
        self.mu = mu

    def f(self, t, x: jax.Array) -> jax.Array:
        x1, x2 = x.ravel()
        return jnp.array([self.mu * (x1 - 1 / 3 * x1**3 - x2), x1 / self.mu])


class Vehicle(System):
    def __init__(self) -> None:
        self.evolution = "continuous"
        self.xlen = 4

    def f(self, t: jax.Array, x: jax.Array, u: jax.Array, w: jax.Array) -> jax.Array:
        px, py, psi, v = x.ravel()
        u1, u2 = u.ravel()
        beta = jnp.arctan(jnp.tan(u2) / 2)
        return jnp.array(
            [v * jnp.cos(psi + beta), v * jnp.sin(psi + beta), v * jnp.sin(beta), u1]
        )


# --- Helper Functions ---


def validate_trajectory(traj_diffrax: Solution):
    """Validates the properties of a computed trajectory."""
    assert isinstance(traj_diffrax, Solution)
    assert traj_diffrax is not None

    t_finite = jnp.isfinite(traj_diffrax.ts)
    computed_ys = traj_diffrax.ys[jnp.where(t_finite)]
    padding_ys = traj_diffrax.ys[jnp.where(~t_finite)]

    assert jnp.isfinite(computed_ys).all()
    assert jnp.isinf(padding_ys).all()
    assert computed_ys.shape[1:] == traj_diffrax.ys.shape[1:]
    assert padding_ys.shape[1:] == traj_diffrax.ys.shape[1:]

    traj = Trajectory.from_diffrax(traj_diffrax)
    assert jnp.equal(traj.ys, computed_ys).all()


# --- Fixtures for 2D Systems (no inputs) ---


@pytest.fixture(
    params=[
        # pytest.param(DoubleIntegrator(), id="HarmOsc"),
        pytest.param(HarmOsc(), id="HarmOsc"),
        pytest.param(VanDerPolOsc(), id="VanDerPolOsc"),
    ]
)
def system_2d(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(jnp.array([1.0, 0.0]), id="unit_x"),
        pytest.param(jnp.array([0.0, 1.0]), id="unit_y"),
        pytest.param(jnp.zeros((2,)), id="zeros"),
        pytest.param(jnp.ones((2,)), id="ones"),
        pytest.param(jnp.array([0.5, -0.5]), id="pos_neg"),
    ]
)
def x0_2d(request):
    return request.param


# --- Fixtures for 4D Systems (with inputs) ---


@pytest.fixture
def system_4d():
    return Vehicle()


@pytest.fixture
def x0_4d():
    return jnp.array([0.0, 0.0, 0.0, 1.0])


@pytest.fixture(
    params=[
        pytest.param(
            (lambda t, x: jnp.array([0.1, 0.1]), lambda x, t: jnp.zeros(1)),
            id="const_input_1",
        ),
        pytest.param(
            (lambda t, x: jnp.array([-0.1, 0.2]), lambda x, t: jnp.zeros(1)),
            id="const_input_2",
        ),
    ]
)
def vehicle_inputs(request):
    """Fixture for Vehicle system inputs."""
    return request.param


@pytest.fixture
def system_linear():
    """Fixture for the LinearSys system."""
    return LinearSys()


# --- Test Functions ---


def test_compute_trajectory_2d(system_2d, x0_2d):
    """Tests trajectory computation for 2D systems without inputs."""
    traj_diffrax = system_2d.compute_trajectory(t0=0, tf=1, x0=x0_2d)
    validate_trajectory(traj_diffrax)


def test_compute_trajectory_4d(system_4d, x0_4d, vehicle_inputs):
    """Tests trajectory computation for the 4D Vehicle system with inputs."""
    traj_diffrax = system_4d.compute_trajectory(
        t0=0, tf=1, x0=x0_4d, inputs=vehicle_inputs
    )
    validate_trajectory(traj_diffrax)


def test_linear_sys_stabilization(system_linear, x0_2d):
    """Tests stabilization of a linear system with feedback control."""
    n = A.shape[0]

    # 1. Compute controllability matrix
    C = jnp.hstack([B] + [jnp.linalg.matrix_power(A, i) @ B for i in range(1, n)])

    # 2. Desired characteristic polynomial (poles at -1, -2)
    # p(s) = s^2 + 3s + 2
    # p(A) = A^2 + 3A + 2I
    pA = jnp.linalg.matrix_power(A, 2) + 3 * A + 2 * jnp.eye(n)

    # 3. Compute gain K using Ackermann's formula
    e_n_T = jnp.zeros((1, n)).at[0, -1].set(1.0)
    K = e_n_T @ jnp.linalg.inv(C) @ pA

    def controller(t, x):
        return -K @ x

    inputs = (controller,)

    traj_diffrax = system_linear.compute_trajectory(
        t0=0, tf=10.0, x0=x0_2d, inputs=inputs
    )

    validate_trajectory(traj_diffrax)

    # Assert that the final state is close to zero
    final_state = traj_diffrax.ys[jnp.where(jnp.isfinite(traj_diffrax.ts))][-1]
    assert jnp.allclose(final_state, jnp.zeros_like(final_state), atol=1e-2)

    # Sanity check: With a zero controller, the system should not stabilize.
    if not jnp.allclose(x0_2d, jnp.zeros_like(x0_2d)):
        zero_controller = (lambda t, x: jnp.zeros((1,)),)
        traj_uncontrolled = system_linear.compute_trajectory(
            t0=0, tf=10.0, x0=x0_2d, inputs=zero_controller
        )
        validate_trajectory(traj_uncontrolled)
        final_state_uncontrolled = traj_uncontrolled.ys[
            jnp.where(jnp.isfinite(traj_uncontrolled.ts))
        ][-1]
        assert not jnp.allclose(
            final_state_uncontrolled,
            jnp.zeros_like(final_state_uncontrolled),
            atol=1e-1,
        )
