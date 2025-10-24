import jax
import jax.numpy as jnp


def validate_overapproximation_nd(func, input_interval, output_interval):
    """
    Validates that the output_interval overapproximates the true values of the
    function over the input_interval by sampling along the diagonal for N-D intervals.
    """
    # Generate 100 sample points along the line from lower to upper bound
    sample_points = jnp.linspace(input_interval.lower, input_interval.upper, 100)

    # Apply the original function to these sample points
    true_values = func(sample_points)

    # Check that all resulting values are within the computed interval bounds
    # Note: This check is along a line, not the full interval hyper-rectangle for vector inputs
    assert jnp.all(true_values >= output_interval.lower)
    assert jnp.all(true_values <= output_interval.upper)


def validate_overapproximation_1d_list(func, input_interval, output_interval):
    """
    Validates that the output_interval overapproximates the true values of the
    function over a list of 1D input_intervals by sampling.
    """
    for i in range(len(input_interval.lower)):
        sample_points = jnp.linspace(
            input_interval.lower[i], input_interval.upper[i], 100
        )
        true_values = jax.vmap(func)(sample_points)
        assert jnp.all(true_values >= output_interval.lower[i].squeeze())
        assert jnp.all(true_values <= output_interval.upper[i].squeeze())
