import jax
import jax.numpy as jnp
import pytest
from immrax import System, RawTrajectory, RawDiscreteTrajectory, DiscreteTrajectory

# --- Systems ---

# Nilpotent system
A_nilpotent = jnp.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
n_nilpotent = A_nilpotent.shape[0]


class NilpotentSys(System):
    def __init__(self) -> None:
        self.evolution = "discrete"
        self.xlen = n_nilpotent
        self.name = "Nilpotent System"

    def f(self, t, x: jax.Array) -> jax.Array:
        return A_nilpotent @ x


# Linear system
A_linear = jnp.array([[1.1, 0.2], [-0.2, 1.1]])  # Unstable system
B_linear = jnp.array([[0.1], [1.0]])


class LinearSysDiscrete(System):
    def __init__(self) -> None:
        self.evolution = "discrete"
        self.xlen = 2
        self.name = "Discrete Linear System"

    def f(self, t, x: jax.Array, u: jax.Array) -> jax.Array:
        return A_linear @ x + B_linear @ u


# Nonlinear system
class LogisticMap(System):
    def __init__(self, r: float = 3.9) -> None:
        self.evolution = "discrete"
        self.xlen = 1
        self.name = "Logistic Map"
        self.r = r

    def f(self, t, x: jax.Array) -> jax.Array:
        return self.r * x * (1 - x)


# --- Helper Functions ---


def validate_trajectory(traj_raw: RawTrajectory):
    """Validates the properties of a computed trajectory."""
    assert isinstance(traj_raw, RawDiscreteTrajectory)
    assert traj_raw is not None

    t_finite = jnp.isfinite(traj_raw.ts)
    computed_ys = traj_raw.ys[jnp.where(t_finite)]
    padding_ys = traj_raw.ys[jnp.where(~t_finite)]

    assert jnp.isfinite(computed_ys).all()
    # assert jnp.isinf(padding_ys).all()
    assert computed_ys.shape[1:] == traj_raw.ys.shape[1:]
    assert padding_ys.shape[1:] == traj_raw.ys.shape[1:]

    traj = traj_raw.to_convenience()
    assert isinstance(traj, DiscreteTrajectory)
    assert jnp.equal(traj.ys, computed_ys).all()


# --- Fixtures ---


@pytest.fixture(params=[pytest.param(LogisticMap(), id="LogisticMap")])
def system_1d(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(jnp.array([0.1]), id="0.1"),
        pytest.param(jnp.array([0.5]), id="0.5"),
    ]
)
def x0_1d(request):
    return request.param


@pytest.fixture
def system_linear_discrete():
    return LinearSysDiscrete()


@pytest.fixture(
    params=[
        pytest.param(jnp.array([1.0, 0.0]), id="unit_x"),
        pytest.param(jnp.array([0.0, 1.0]), id="unit_y"),
        pytest.param(jnp.ones(2), id="ones"),
    ]
)
def x0_2d(request):
    return request.param


@pytest.fixture
def system_nilpotent():
    return NilpotentSys()


@pytest.fixture
def x0_3d():
    return jnp.array([1.0, 2.0, 3.0])


# --- Test Functions ---


def test_compute_trajectory_1d(system_1d, x0_1d):
    """Tests trajectory computation for 1D systems without inputs."""
    traj = system_1d.compute_trajectory(t0=0, tf=10, x0=x0_1d)
    validate_trajectory(traj)
    traj = traj.to_convenience()
    assert jnp.equal(traj.ys[0], x0_1d).all()
    assert traj.ys.shape == (11, 1)


def test_compute_trajectory_2d(system_linear_discrete, x0_2d):
    """Tests trajectory computation for 2D systems with constant input."""

    def controller(t, x):
        return jnp.array([0.1])

    inputs = (controller,)
    traj = system_linear_discrete.compute_trajectory(
        t0=0, tf=10, x0=x0_2d, inputs=inputs
    )
    validate_trajectory(traj)
    traj = traj.to_convenience()
    assert jnp.equal(traj.ys[0], x0_2d).all()
    assert traj.ys.shape == (11, 2)


def test_nilpotent_convergence(system_nilpotent, x0_3d):
    """Tests that a nilpotent system converges to zero in n steps."""
    n = system_nilpotent.xlen
    traj = system_nilpotent.compute_trajectory(t0=0, tf=n, x0=x0_3d)
    validate_trajectory(traj)
    traj = traj.to_convenience()
    assert jnp.equal(traj.ys[0], x0_3d).all()
    assert traj.ys.shape[0] == n + 1

    # State at time n should be zero
    final_state = traj.ys[-1]
    assert jnp.allclose(final_state, jnp.zeros_like(final_state))

    # For sanity, check that A^n is the zero matrix
    An = jnp.linalg.matrix_power(A_nilpotent, n)
    assert jnp.allclose(An, jnp.zeros_like(An))


def test_linear_sys_stabilization_discrete(system_linear_discrete, x0_2d):
    """Tests stabilization of a discrete linear system with feedback control."""
    n = system_linear_discrete.xlen
    A, B = A_linear, B_linear

    # 1. Compute controllability matrix
    C = jnp.hstack([B] + [jnp.linalg.matrix_power(A, i) @ B for i in range(1, n)])
    assert jnp.linalg.matrix_rank(C) == n, "System is not controllable"

    # 2. Desired characteristic polynomial (poles at 0.5, 0.6)
    # p(z) = (z - 0.5)(z - 0.6) = z^2 - 1.1z + 0.3
    # p(A) = A^2 - 1.1A + 0.3I
    pA = jnp.linalg.matrix_power(A, 2) - 1.1 * A + 0.3 * jnp.eye(n)

    # 3. Compute gain K using Ackermann's formula
    e_n_T = jnp.zeros((1, n)).at[0, -1].set(1.0)
    K = e_n_T @ jnp.linalg.inv(C) @ pA

    def controller(t, x):
        return -K @ x

    inputs = (controller,)

    traj = system_linear_discrete.compute_trajectory(
        t0=0, tf=50, x0=x0_2d, inputs=inputs
    )

    validate_trajectory(traj)
    traj = traj.to_convenience()
    assert jnp.equal(traj.ys[0], x0_2d).all()

    # Assert that the final state is close to zero
    final_state = traj.ys[-1]
    assert jnp.allclose(final_state, jnp.zeros_like(final_state), atol=1e-2)

    # Sanity check: With a zero controller, the system should not stabilize.
    if not jnp.allclose(x0_2d, jnp.zeros_like(x0_2d)):
        zero_controller = (lambda t, x: jnp.zeros((1,)),)
        traj_uncontrolled = system_linear_discrete.compute_trajectory(
            t0=0, tf=50, x0=x0_2d, inputs=zero_controller
        )
        validate_trajectory(traj_uncontrolled)
        traj_uncontrolled = traj_uncontrolled.to_convenience()
        final_state_uncontrolled = traj_uncontrolled.ys[-1]
        assert not jnp.allclose(
            final_state_uncontrolled,
            jnp.zeros_like(final_state_uncontrolled),
            atol=1e-1,
        )


def test_ragged_trajectory_discrete(system_1d, x0_1d):
    """Tests ragged trajectory creation for discrete systems."""
    tfs = jnp.arange(5, 10)

    # vmap compute_trajectory over tf
    # We expect this to create a ragged trajectory, as each `tf` is different.
    compute_traj_vmap = jax.vmap(system_1d.compute_trajectory, in_axes=(None, 0, None))

    raw_traj = compute_traj_vmap(0, tfs, x0_1d)

    traj = raw_traj.to_convenience()

    # 1. is_ragged should return true
    assert traj.is_ragged()

    # 2. The list of ts should have the same length as the range of tfs
    assert len(traj.ts) == len(tfs)

    # 3. Each ts should correspond to a ys of the same length
    for i in range(len(tfs)):
        assert len(traj.ts[i]) == len(traj.ys[i])
        # For discrete systems, trajectory length is tf + 1 (for t0=0)
        assert len(traj.ts[i]) == tfs[i] + 1

    # 4. Each element of ys should be finite
    for i in range(len(tfs)):
        assert jnp.all(jnp.isfinite(traj.ys[i]))
