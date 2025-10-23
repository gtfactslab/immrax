import jax
import jax.numpy as jnp
import pytest

import immrax as irx
from immrax.inclusion.polynomial import polynomial

# --- Test Case Flags ---
# Set these to True to run the corresponding tests.
# You can then run tests using `pytest` in your terminal.
TEST_VECTOR_INPUTS = True
TEST_INCLUSION_FUNCTIONS = True
TEST_JACFWD = True
TEST_JACREV = True
TEST_JIT_COMPILATION = True

# --- Fixtures for test data ---

coeff_params = [
    pytest.param(jnp.array([1.0, 4]), id="2nd-order"),
    pytest.param(jnp.array([1.0, 4, -5]), id="3rd-order"),
    pytest.param(jnp.array([1.0, 4, -5, -3]), id="4th-order"),
]

# --- Helper Functions and Dynamic Parameters ---


# Build the list of parameters for the evaluation points
eval_point_params = [
    pytest.param(1.0, id="scalar_float"),
    pytest.param(jnp.array([1.0]), id="scalar_array"),
]
if TEST_VECTOR_INPUTS:
    eval_point_params.append(pytest.param(jnp.array([1.0, 2.0, 3.0]), id="vector"))

# Build the list of parameters for interval evaluation

eval_interval_params = []

if TEST_INCLUSION_FUNCTIONS:
    eval_interval_params.extend(
        [
            pytest.param(irx.icentpert(1.0, 0.1), id="scalar_interval"),
            pytest.param(
                irx.icentpert(jnp.array([1.0]), 0.1), id="scalar_array_interval"
            ),
        ]
    )

    if TEST_VECTOR_INPUTS:
        eval_interval_params.append(
            pytest.param(
                irx.icentpert(jnp.array([1.0, 2.0, 3.0]), 0.1), id="vector_interval"
            )
        )


# --- Parametrized Fixtures ---


@pytest.fixture(params=coeff_params)
def poly_coeff(request):
    """Parametrized fixture for polynomial coefficients."""

    return request.param


@pytest.fixture(params=eval_point_params)
def eval_point(request):
    """Parametrized fixture for evaluation points (scalar/vector)."""

    return request.param


@pytest.fixture(params=eval_interval_params)
def eval_interval(request):
    """Parametrized fixture for evaluation intervals."""

    return request.param


# --- Generic Test Functions ---


def test_polynomial_evaluation(poly_coeff, eval_point):
    """Tests polynomial evaluation for various dynamically-provided input types."""

    result = polynomial(poly_coeff, eval_point)

    expected = jnp.polyval(poly_coeff, eval_point)

    assert jnp.all(result == expected)


@pytest.mark.skipif(
    not TEST_INCLUSION_FUNCTIONS, reason="Inclusion function tests are disabled"
)
def test_inclusion_function(poly_coeff, eval_interval):
    """Tests the natif of the inclusion function for various interval types."""
    poly_natif = irx.natif(polynomial)
    result = poly_natif(poly_coeff, eval_interval)

    # Check the type first
    assert isinstance(result, irx.Interval)

    # Validate the overapproximation by sampling
    validate_overapproximation(polynomial, poly_coeff, eval_interval, result)


@pytest.mark.skipif(not TEST_JACFWD, reason="JACFWD tests are disabled")
def test_jacfwd(poly_coeff, eval_point):
    """Tests forward-mode AD for various dynamically-provided input types."""
    pd_fwd = jax.jacfwd(polynomial, argnums=1)
    der_ad = pd_fwd(poly_coeff, eval_point)

    der_sym = jnp.polyval(jnp.polyder(poly_coeff), eval_point)

    # print()
    # print(f"{eval_point=}")
    # print(f"{der_ad=}")
    # print(f"{der_sym=}")

    if jnp.ndim(eval_point) == 0:
        expected_jacobian = der_sym
    else:
        expected_jacobian = jnp.diag(der_sym)

    assert jnp.allclose(der_ad, expected_jacobian)


@pytest.mark.skipif(not TEST_JACREV, reason="JACREV tests are disabled")
def test_jacrev(poly_coeff, eval_point):
    """Tests reverse-mode AD for various dynamically-provided input types."""
    pd_rev = jax.jacrev(polynomial, argnums=1)
    jacobian = pd_rev(poly_coeff, eval_point)

    deriv_vals = jnp.polyval(jnp.polyder(poly_coeff), eval_point)

    if jnp.ndim(eval_point) == 0:
        expected_jacobian = deriv_vals
    else:
        expected_jacobian = jnp.diag(deriv_vals)

    assert jnp.allclose(jacobian, expected_jacobian)


def validate_overapproximation(func, poly_coeff, input_interval, output_interval):
    """
    Validates that the output_interval overapproximates the true values of the
    function over the input_interval by sampling.
    """
    # Generate 100 sample points along the line from lower to upper bound
    sample_points = jnp.linspace(input_interval.lower, input_interval.upper, 100)

    # Apply the original polynomial function to these sample points
    true_values = func(poly_coeff, sample_points)

    # Check that all resulting values are within the computed interval bounds
    # Note: This check is along a line, not the full interval hyper-rectangle for vector inputs
    assert jnp.all(true_values >= output_interval.lower)
    assert jnp.all(true_values <= output_interval.upper)


@pytest.mark.skipif(
    not (TEST_JIT_COMPILATION and TEST_INCLUSION_FUNCTIONS),
    reason="JIT inclusion tests are disabled",
)
def test_jit_inclusion(poly_coeff, eval_interval):
    """Tests the JIT-compiled inclusion function for various interval types."""
    poly_natif = irx.natif(polynomial)
    poly_natif_jit = jax.jit(poly_natif)
    result = poly_natif_jit(poly_coeff, eval_interval)

    # Check the type first
    assert isinstance(result, irx.Interval)

    # Validate the overapproximation by sampling
    validate_overapproximation(polynomial, poly_coeff, eval_interval, result)
