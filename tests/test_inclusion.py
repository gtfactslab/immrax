import jax
import jax.numpy as jnp
import pytest

import immrax as irx
from tests.utils import validate_overapproximation_nd

# --- Test Case Flags ---
TEST_NATIF = True
TEST_JACIF = True
TEST_MJACIF = True
TEST_LIN_SYS = True


# --- Helper Functions ---
def square_sin(x):
    """A simple non-linear function for testing."""
    return jnp.sin(x**2)


def exp_add(x):
    """A simple non-linear function for testing."""
    return jnp.exp(x) + x


def lin_sys(x, u):
    A = jnp.array([[1.0, 2.0], [1.0, 1.0]])
    B = jnp.array([[0.0], [1.0]])
    return A @ x[:] + B @ u[:]


# --- Fixtures for test data ---


@pytest.fixture(
    params=[
        pytest.param(square_sin, id="sin(x^2)"),
        pytest.param(exp_add, id="exp(x)+x"),
    ]
)
def jax_fn(request):
    """Parametrized fixture for JAX-traceable functions."""
    return request.param


@pytest.fixture(
    params=[
        # pytest.param(jnp.array(1.0), id="scalar"), # FIXME: jacobian-based IFs change this shape
        pytest.param(jnp.array([1.0]), id="(1,) array"),
        pytest.param(jnp.array([1.0, 2.0]), id="(2,) array"),
    ]
)
def eval_point(request):
    """Parametrized fixture for evaluation points."""
    return request.param


@pytest.fixture
def eval_interval(eval_point):
    """Fixture for evaluation intervals, derived from eval_point."""
    return irx.icentpert(eval_point, 0.1)


# --- Test Functions ---


@pytest.mark.skipif(not TEST_NATIF, reason="natif tests are disabled")
def test_natif(jax_fn, eval_point, eval_interval):
    """Tests the natural interval extension (natif)."""
    inclusion_fn = irx.natif(jax_fn)
    result = inclusion_fn(eval_interval)

    # 1. Check that the output is an Interval
    assert isinstance(result, irx.Interval)

    # 2. Check that the shape is correct
    expected_shape = jax.eval_shape(jax_fn, eval_point).shape
    assert result.shape == expected_shape

    # 3. Check that the interval bounds are valid
    assert jnp.all(result.lower <= result.upper)

    # 4. Validate that the interval overapproximates the true function range
    validate_overapproximation_nd(jax_fn, eval_interval, result)


@pytest.mark.skipif(not TEST_JACIF, reason="jacif tests are disabled")
def test_jacif(jax_fn, eval_point, eval_interval):
    """Tests the Jacobian-based inclusion function (jacif)."""
    inclusion_fn = irx.jacif(jax_fn)
    result = inclusion_fn(eval_interval)

    # 1. Check that the output is an Interval
    assert isinstance(result, irx.Interval)

    # 2. Check that the shape is correct
    expected_shape = jax.eval_shape(jax_fn, eval_point).shape
    assert result.shape == expected_shape

    # 3. Check that the interval bounds are valid
    assert jnp.all(result.lower <= result.upper)

    # 4. Validate that the interval overapproximates the true Jacobian range
    validate_overapproximation_nd(jax_fn, eval_interval, result)


@pytest.mark.skipif(not TEST_MJACIF, reason="mjacif tests are disabled")
def test_mjacif(jax_fn, eval_point, eval_interval):
    """Tests the mixed-order Jacobian-based inclusion function (mjacif)."""
    inclusion_fn = irx.mjacif(jax_fn)
    result = inclusion_fn(eval_interval)

    # 1. Check that the output is an Interval
    assert isinstance(result, irx.Interval)

    # 2. Check that the shape is correct
    expected_shape = jax.eval_shape(jax_fn, eval_point).shape
    assert result.shape == expected_shape

    # 3. Check that the interval bounds are valid
    assert jnp.all(result.lower <= result.upper)

    # 4. Validate that the interval overapproximates the true Jacobian range
    validate_overapproximation_nd(jax_fn, eval_interval, result)


# --- Tests for lin_sys ---


@pytest.fixture
def x_vec():
    """Fixture for the state vector 'x' for lin_sys."""
    return jnp.array([1.0, 2.0])


@pytest.fixture
def u_vec():
    """Fixture for the control vector 'u' for lin_sys."""
    return jnp.array([0.5])


@pytest.fixture
def x_interval(x_vec):
    """Fixture for the state interval 'x' for lin_sys."""
    return irx.icentpert(x_vec, 0.1)


@pytest.fixture
def u_interval(u_vec):
    """Fixture for the control interval 'u' for lin_sys."""
    return irx.icentpert(u_vec, 0.05)


@pytest.mark.skipif(not TEST_LIN_SYS, reason="lin_sys tests are disabled")
def test_lin_sys_natif(x_vec, u_vec, x_interval, u_interval):
    """Tests the natural interval extension (natif) for a multi-argument function."""
    inclusion_fn = irx.natif(lin_sys)
    result = inclusion_fn(x_interval, u_interval)

    # 1. Check that the output is an Interval
    assert isinstance(result, irx.Interval)

    # 2. Check that the shape is correct
    expected_shape = jax.eval_shape(lin_sys, x_vec, u_vec).shape
    assert result.shape == expected_shape

    # 3. Check that the interval bounds are valid
    assert jnp.all(result.lower <= result.upper)

    # 4. Validate that the interval overapproximates the true function range
    def wrapped_lin_sys(xu_vec):
        x = xu_vec[:, :2]
        u = xu_vec[:, 2:]
        # Vectorize lin_sys to handle the batch of 100 samples
        vmapped_lin_sys = jax.vmap(lin_sys, in_axes=(0, 0))
        return vmapped_lin_sys(x, u)

    combined_lower = jnp.concatenate([x_interval.lower, u_interval.lower])
    combined_upper = jnp.concatenate([x_interval.upper, u_interval.upper])
    combined_interval = irx.interval(combined_lower, combined_upper)

    validate_overapproximation_nd(wrapped_lin_sys, combined_interval, result)
