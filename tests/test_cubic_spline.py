import jax
import jax.numpy as jnp
import numpy as np
import pytest
import matplotlib.pyplot as plt

import immrax as irx
from immrax.inclusion.cubic_spline import (
    create_cubic_spline_coeffs,
    make_spline_eval_fn,
)
from immrax.inclusion.polynomial import polynomial
from tests.utils import validate_overapproximation_1d_list

# --- Test Case Flags ---
TEST_INCLUSION_FUNCTIONS = True
TEST_JIT_COMPILATION = True
TEST_JACFWD = True
TEST_JACREV = True

# --- Fixtures for test data ---


@pytest.fixture(scope="module")
def spline_raw_data():
    """Pytest fixture to generate raw data for spline tests."""
    num_points = 20
    x_range = (0.0, 50.0)
    x_coords = np.linspace(x_range[0], x_range[1], num_points)
    y_range = (-10.0, 10.0)
    y_coords = np.random.uniform(y_range[0], y_range[1], size=num_points)
    points = jnp.array(np.vstack((x_coords, y_coords)).T)
    return points


@pytest.fixture(scope="module")
def spline_coeffs(spline_raw_data):
    """Pytest fixture to compute spline coefficients."""
    return create_cubic_spline_coeffs(spline_raw_data)


@pytest.fixture(scope="module")
def spline_eval_fn(spline_coeffs):
    """Pytest fixture to create a spline evaluation function."""
    x_knots, coeffs = spline_coeffs
    return make_spline_eval_fn(x_knots, coeffs)


@pytest.fixture(
    params=[
        pytest.param(10.5, id="scalar_float"),
        pytest.param(jnp.array([10.5]), id="scalar_array"),
        pytest.param(jnp.array([10.5, 20.5, 30.5]), id="vector"),
    ]
)
def eval_point(request):
    """Parametrized fixture for evaluation points."""
    return request.param


# --- Helper Functions ---


def make_spline_derivative_eval_fn(x_knots, coeffs):
    """Creates a function that evaluates the derivative of the cubic spline."""
    a, b, c, d = coeffs

    def derivative_eval_fn(x_eval):
        """Evaluates the spline derivative at given x-values."""
        i = jnp.sum(
            jax.vmap(lambda x: x_knots[1:] < x)(jnp.atleast_1d(x_eval)),
            axis=1,
        )
        dx = x_eval - x_knots[i]

        # Derivative of the polynomial: 3*d*dx^2 + 2*c*dx + b
        der_coeffs = jnp.array([3 * d[i], 2 * c[i], b[i]]).squeeze()
        return polynomial(der_coeffs, dx)

    return derivative_eval_fn


# --- Test Functions ---


def test_spline_evaluation_at_knots(spline_raw_data, spline_eval_fn):
    """Tests if the spline evaluates to the original y-values at the knots."""
    original_x = spline_raw_data[:, 0]
    original_y = spline_raw_data[:, 1]

    evaluated_y = spline_eval_fn(original_x)

    assert jnp.allclose(original_y, evaluated_y, atol=1e-6)


@pytest.mark.skipif(
    not TEST_INCLUSION_FUNCTIONS, reason="Inclusion function tests are disabled"
)
def test_inclusion_function(spline_coeffs, spline_eval_fn):
    """Tests the natif of the inclusion function for the spline."""
    x_knots, _ = spline_coeffs

    eval_interval = irx.interval(x_knots[:-1] + 0.1, x_knots[1:] - 0.1)

    spline_natif = jax.vmap(irx.natif(spline_eval_fn))
    result = spline_natif(eval_interval)

    assert isinstance(result, irx.Interval)

    validate_overapproximation_1d_list(spline_eval_fn, eval_interval, result)


@pytest.mark.skipif(
    not (TEST_JIT_COMPILATION and TEST_INCLUSION_FUNCTIONS),
    reason="JIT inclusion tests are disabled",
)
def test_jit_inclusion(spline_coeffs, spline_eval_fn):
    """Tests the JIT-compiled inclusion function for the spline."""
    x_knots, _ = spline_coeffs

    eval_interval = irx.interval(x_knots[:-1] + 0.1, x_knots[1:] - 0.1)

    spline_natif_jit = jax.jit(jax.vmap(irx.natif(spline_eval_fn)))
    result = spline_natif_jit(eval_interval)

    assert isinstance(result, irx.Interval)

    validate_overapproximation_1d_list(spline_eval_fn, eval_interval, result)


@pytest.mark.skipif(not TEST_JACFWD, reason="JACFWD tests are disabled")
def test_jacfwd(spline_coeffs, spline_eval_fn, eval_point):
    """Tests forward-mode AD for the spline evaluation function."""
    x_knots, coeffs = spline_coeffs

    spline_jacfwd = jax.jacfwd(spline_eval_fn)
    der_ad = spline_jacfwd(eval_point)

    spline_der_fn = make_spline_derivative_eval_fn(x_knots, coeffs)
    der_sym = spline_der_fn(eval_point)

    if jnp.ndim(eval_point) == 0:
        expected_jacobian = der_sym
    else:
        expected_jacobian = jnp.diag(der_sym)

    assert jnp.allclose(der_ad, expected_jacobian)


@pytest.mark.skipif(not TEST_JACREV, reason="JACREV tests are disabled")
def test_jacrev(spline_coeffs, spline_eval_fn, eval_point):
    """Tests reverse-mode AD for the spline evaluation function."""
    x_knots, coeffs = spline_coeffs

    spline_jacrev = jax.jacrev(spline_eval_fn)
    der_ad = spline_jacrev(eval_point)

    spline_der_fn = make_spline_derivative_eval_fn(x_knots, coeffs)
    der_sym = spline_der_fn(eval_point)

    if jnp.ndim(eval_point) == 0:
        expected_jacobian = der_sym
    else:
        expected_jacobian = jnp.diag(der_sym)

    assert jnp.allclose(der_ad, expected_jacobian)


def plot_interval_bounds(ax, interval_bounds, output_bounds, color, label):
    """
    Adds shaded rectangles to a plot to visualize interval bounds.
    """
    for i in range(len(interval_bounds.lower)):
        rect = plt.Rectangle(
            (interval_bounds.lower[i], output_bounds.lower[i]),
            interval_bounds.upper[i] - interval_bounds.lower[i],
            output_bounds.upper[i] - output_bounds.lower[i],
            facecolor=color,
            alpha=0.4,
            label=label if i == 0 else "",
        )
        ax.add_patch(rect)


if __name__ == "__main__":
    # Example of calling the function and comparing to input values.
    # This example uses JAX for computation and matplotlib for plotting.
    # You can install them with: pip install jax jaxlib numpy matplotlib

    # 1. Define the input data points (x, y)
    # Procedurally generate points to test scaling
    num_points = 20  # Change this value to test with different numbers of points

    # Generate sorted x-coordinates over a fixed range
    x_range = (0.0, 50.0)
    x_coords = np.linspace(x_range[0], x_range[1], num_points)

    # Generate y-coordinates randomly from a uniform distribution
    y_range = (-10.0, 10.0)
    y_coords = np.random.uniform(y_range[0], y_range[1], size=num_points)

    # Combine into a single array of points
    input_points_np = np.vstack((x_coords, y_coords)).T

    # Convert to JAX array for spline computation
    input_points_jax = jnp.array(input_points_np)

    # Sort for comparison and plotting
    sorted_indices = np.argsort(input_points_np[:, 0])
    input_points_np_sorted = input_points_np[sorted_indices]

    # 2. Create the cubic spline coefficients
    x_knots, coeffs = create_cubic_spline_coeffs(input_points_jax)

    # 3. Create the evaluation function
    spline_eval_fn = make_spline_eval_fn(x_knots, coeffs)

    # 4. Compare to input values
    original_x = input_points_np_sorted[:, 0]
    original_y = input_points_np_sorted[:, 1]

    # 5. Demonstrate JAX traceability and interval bounds
    # Generate points for plotting
    x_smooth = np.linspace(x_knots[0], x_knots[-1], 200)
    y_smooth = spline_eval_fn(x_smooth)

    # Define interval bounds for testing
    # Case 3: N evenly spaced overlapping intervals
    N = 25
    w = 1.2 * (x_knots[-1] - x_knots[0]) / (N - 1)
    centers = np.linspace(x_knots[0], x_knots[-1], N)
    interval_bounds_overlapping = irx.interval(
        jnp.array(centers - w / 2), jnp.array(centers + w / 2)
    )

    # Test interval bounds using immrax: natif
    spline_inclusion_fn_natif = jax.vmap(irx.natif(spline_eval_fn))
    output_bounds_natif_overlapping = spline_inclusion_fn_natif(
        interval_bounds_overlapping
    )

    # 6. Plot the results for visual comparison
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Plot the main spline curve
    plt.plot(
        x_smooth, y_smooth, label="Cubic Spline", color="blue", linewidth=2.5, zorder=5
    )

    # Plot the original data points
    plt.plot(
        original_x,
        original_y,
        "o",
        label="Original Data Points",
        color="red",
        markersize=8,
        zorder=10,
    )

    plot_interval_bounds(
        ax,
        interval_bounds_overlapping,
        output_bounds_natif_overlapping,
        "purple",
        "Overlapping",
    )

    plt.title("Cubic Spline Interpolation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
