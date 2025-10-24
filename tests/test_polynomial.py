import jax
import jax.numpy as jnp
import pytest

import immrax as irx
from immrax.inclusion.polynomial import polynomial
from tests.utils import validate_overapproximation_nd

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
    validate_overapproximation_nd(
        lambda x: polynomial(poly_coeff, x), eval_interval, result
    )


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
    validate_overapproximation_nd(
        lambda x: polynomial(poly_coeff, x), eval_interval, result
    )


if __name__ == "__main__":
    a = jnp.array([1, 2, 3])
    # The major axis determines the degree of the polynomial (i.e. num rows = degree + 1)
    # The minor axis determines the number of polynomials (i.e. num cols = num polynomials)
    a_multiple = jnp.array(
        [
            [1.0, 4, 2],
            [-3, 2, 2],
            [1, 2, 2],
        ]
    )
    x = 2
    x_multiple = 1 + jnp.arange(
        a_multiple.shape[1]
    )  # WARN: this only works if x_multiple.shape() = a_multiple.shape()[1]

    # Confirmed same behavior as jnp.polyval
    print(polynomial(a, x))
    print(polynomial(a_multiple, x))

    print(polynomial(a, x_multiple))
    print(polynomial(a_multiple, x_multiple))
    # This batches over both arguments simultaneously (NOT product wise)
    # That is, the first polynomial is ONLY evaluated at the first point, the second at the second, etc.

    # Checking polyder
    print(jnp.polyder(a))
    # print(jnp.polyder(a_multiple)) # This is not supported by jnp or numpy

    # print(jnp.polyder(jnp.array([[1]])))
    # print(jnp.polyder(jnp.array([[1], [2]])))
    # print(jnp.polyder(jnp.array([[1], [2], [3]])))
    # print(jnp.polyder(jnp.array([[1], [2], [3], [4]]).squeeze()))
