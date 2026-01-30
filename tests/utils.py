import jax
import jax.numpy as jnp


def validate_overapproximation_nd(
    func, input_interval, output_interval, rtol=1e-5, atol=1e-7
):
    """
    Validates that the output_interval overapproximates the true values of the
    function over the input_interval by sampling along the diagonal for N-D intervals.

    This function is for functions that naturally handle batched/multi-dimensional
    inputs, either through broadcasting or internal vectorization.

    Uses fuzzy comparison to accommodate floating point precision errors.
    The check passes if true_values are within the interval bounds up to
    a small tolerance: lower - tol <= true_values <= upper + tol

    Parameters
    ----------
    func : callable
        The function to evaluate. Should accept batched inputs directly.
    input_interval : Interval
        The input interval to sample from
    output_interval : Interval
        The output interval that should overapproximate func(input_interval)
    rtol : float
        Relative tolerance for fuzzy comparison (default 1e-5)
    atol : float
        Absolute tolerance for fuzzy comparison (default 1e-7)
    """
    # Generate 100 sample points along the line from lower to upper bound
    sample_points = jnp.linspace(input_interval.lower, input_interval.upper, 100)

    # Apply the original function to these sample points
    # The function handles batching internally (either via broadcasting or vmap)
    true_values = func(sample_points)

    # Compute tolerances based on the magnitude of the bounds
    lower_tol = atol + rtol * jnp.abs(output_interval.lower)
    upper_tol = atol + rtol * jnp.abs(output_interval.upper)

    # Check that all resulting values are within the computed interval bounds
    # with tolerance for floating point errors
    # Note: This check is along a line, not the full interval hyper-rectangle for vector inputs
    assert jnp.all(true_values >= output_interval.lower - lower_tol), (
        f"Lower bound violation: min(true_values)={jnp.min(true_values)}, "
        f"lower_bound={output_interval.lower}, tol={lower_tol}"
    )
    assert jnp.all(true_values <= output_interval.upper + upper_tol), (
        f"Upper bound violation: max(true_values)={jnp.max(true_values)}, "
        f"upper_bound={output_interval.upper}, tol={upper_tol}"
    )


def validate_elementwise_overapproximation(
    func, input_interval, output_interval, rtol=1e-5, atol=1e-7
):
    """
    Validates that the output_interval overapproximates the true values of a
    scalar-to-scalar function over the input_interval.

    This function uses vmap to evaluate the function element-wise on sample points.
    This is necessary for functions like gradient functions (jacrev/jacfwd) which
    return Jacobian matrices when given batched inputs instead of element-wise results.

    Uses fuzzy comparison to accommodate floating point precision errors.

    Parameters
    ----------
    func : callable
        A scalar-to-scalar function to evaluate element-wise.
    input_interval : Interval
        The input interval to sample from
    output_interval : Interval
        The output interval that should overapproximate func(input_interval)
    rtol : float
        Relative tolerance for fuzzy comparison (default 1e-5)
    atol : float
        Absolute tolerance for fuzzy comparison (default 1e-7)
    """
    # Generate 100 sample points along the line from lower to upper bound
    sample_points = jnp.linspace(input_interval.lower, input_interval.upper, 100)

    # Apply the function element-wise using vmap
    # This is necessary for gradient functions which would otherwise return
    # a Jacobian matrix instead of a vector of element-wise derivatives
    true_values = jax.vmap(func)(sample_points)

    # Compute tolerances based on the magnitude of the bounds
    lower_tol = atol + rtol * jnp.abs(output_interval.lower)
    upper_tol = atol + rtol * jnp.abs(output_interval.upper)

    # Check that all resulting values are within the computed interval bounds
    # with tolerance for floating point errors
    assert jnp.all(true_values >= output_interval.lower - lower_tol), (
        f"Lower bound violation: min(true_values)={jnp.min(true_values)}, "
        f"lower_bound={output_interval.lower}, tol={lower_tol}"
    )
    assert jnp.all(true_values <= output_interval.upper + upper_tol), (
        f"Upper bound violation: max(true_values)={jnp.max(true_values)}, "
        f"upper_bound={output_interval.upper}, tol={upper_tol}"
    )


def validate_overapproximation_1d_list(
    func, input_interval, output_interval, rtol=1e-5, atol=1e-7
):
    """
    Validates that the output_interval overapproximates the true values of the
    function over a list of 1D input_intervals by sampling.

    Uses fuzzy comparison to accommodate floating point precision errors.

    Parameters
    ----------
    func : callable
        The function to evaluate
    input_interval : Interval
        The input interval to sample from
    output_interval : Interval
        The output interval that should overapproximate func(input_interval)
    rtol : float
        Relative tolerance for fuzzy comparison (default 1e-5)
    atol : float
        Absolute tolerance for fuzzy comparison (default 1e-7)
    """
    for i in range(len(input_interval.lower)):
        sample_points = jnp.linspace(
            input_interval.lower[i], input_interval.upper[i], 100
        )
        true_values = jax.vmap(func)(sample_points)

        lower_bound = output_interval.lower[i].squeeze()
        upper_bound = output_interval.upper[i].squeeze()

        # Compute tolerances based on the magnitude of the bounds
        lower_tol = atol + rtol * jnp.abs(lower_bound)
        upper_tol = atol + rtol * jnp.abs(upper_bound)

        assert jnp.all(true_values >= lower_bound - lower_tol), (
            f"Lower bound violation at index {i}: min(true_values)={jnp.min(true_values)}, "
            f"lower_bound={lower_bound}, tol={lower_tol}"
        )
        assert jnp.all(true_values <= upper_bound + upper_tol), (
            f"Upper bound violation at index {i}: max(true_values)={jnp.max(true_values)}, "
            f"upper_bound={upper_bound}, tol={upper_tol}"
        )
