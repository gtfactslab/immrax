import jax
from jax import core
import jax.numpy as jnp

import immrax as irx
from immrax.inclusion.polynomial import polynomial
from immrax.inclusion import register_inclusion_primitive


def create_cubic_spline_coeffs(points):
    """
    Computes the coefficients for a natural cubic spline from a list of points.
    This implementation is designed to be JAX-traceable.

    Args:
      points: A JAX numpy array of (x, y) coordinates.

    Returns:
      A tuple containing:
        - x_knots: The x-coordinates of the knots.
        - coeffs: A tuple of (a, b, c, d) coefficients for the n-1 splines.
    """
    # Sort points by x-values, as JAX requires static shapes for many operations
    # and sorting inside a jitted function can be tricky if not handled carefully.
    # Here we assume points are sorted or we sort them before passing to a jitted function.
    sorted_indices = jnp.argsort(points[:, 0])
    points = points[sorted_indices]

    x = points[:, 0]
    y = points[:, 1]
    n = len(x)

    if n < 3:
        raise ValueError("At least 3 points are required to build a cubic spline.")

    h = jnp.diff(x)

    # Setup the tridiagonal system for the second derivatives (related to 'c' coeffs)
    # For a natural spline, the second derivatives at the endpoints are 0.
    # This leaves us with n-2 unknowns to solve for.

    # Main diagonal of the tridiagonal matrix
    diag = 2 * (h[:-1] + h[1:])

    # Off-diagonals
    off_diag = h[1:-1]

    # Construct the (n-2)x(n-2) tridiagonal matrix A
    A = jnp.diag(diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)

    # Right-hand side vector B
    B = 3 * (jnp.diff(y[1:]) / h[1:] - jnp.diff(y[:-1]) / h[:-1])

    # Solve Ac = B for the inner c coefficients (c_1 to c_{n-2})
    # Using jnp.linalg.solve which is traceable
    c_inner = jnp.linalg.solve(A, B)

    # Combine with boundary conditions (c_0 = c_{n-1} = 0 for natural spline)
    c = jnp.concatenate([jnp.array([0.0]), c_inner, jnp.array([0.0])])

    # Calculate remaining coefficients a, b, d from c
    # a_i = y_i
    a = y[:-1]

    # b_i = (y_{i+1} - y_i)/h_i - h_i/3 * (2*c_i + c_{i+1})
    b = (jnp.diff(y) / h) - (h / 3) * (2 * c[:-1] + c[1:])

    # d_i = (c_{i+1} - c_i) / (3*h_i)
    d = jnp.diff(c) / (3 * h)

    # The coefficients are for n-1 polynomials.
    # c is of length n, so we take c[:-1] for the c_i coefficient of the i-th polynomial.
    coeffs = (a, b, c[:-1], d)
    x_knots = x

    return x_knots, coeffs


def make_spline_eval_fn(x_knots, coeffs):
    """
    Creates a JAX-traceable function that evaluates the cubic spline.

    Args:
      x_knots: The x-coordinates of the knots.
      coeffs: A tuple of (a, b, c, d) spline coefficients.

    Returns:
      A function that takes a single x-value or an array of x-values and
      returns the corresponding interpolated y-value(s).
    """
    a, b, c, d = coeffs

    def eval_fn(x_eval):
        """
        Evaluates the spline at given x-values.
        """
        # Vectorized search for the correct interval for each x_eval
        i = jnp.clip(jnp.searchsorted(x_knots, x_eval, side="right") - 1, 0, len(a) - 1)

        # Evaluate the polynomial for each x_eval using its interval index i
        dx = x_eval - x_knots[i]
        my_eval = polynomial(jnp.array([d[i], c[i], b[i], a[i]]).squeeze(), dx)

        return my_eval

    def incl_fn(int_eval):
        # For extrapolation, we extend the first and last domains to infinity.
        lower_bins = jnp.concatenate((-jnp.array([jnp.inf]), x_knots[1:-1]))
        upper_bins = jnp.concatenate((x_knots[1:-1], jnp.array([jnp.inf])))
        bins = irx.interval(lower_bins, upper_bins)

        # int_eval_partitioned = jax.vmap(lambda x, y: x and y, in_axes=(0, None))(
        int_eval_partitioned = jax.vmap(
            lambda x, y: x & y,
            in_axes=(0, None),
        )(bins, int_eval)

        # The polynomial is evaluated on dx = x - x_knots[i].
        # We need to compute the interval for dx for each bin.
        dx_intervals = int_eval_partitioned - x_knots[:-1]

        out_partitioned = jax.vmap(irx.natif(polynomial), out_axes=(-1))(
            jnp.array([d, c, b, a]).T, dx_intervals
        )

        # Identify empty intervals in the input
        is_empty = int_eval_partitioned.lower >= int_eval_partitioned.upper

        # Use jnp.where to replace empty intervals with neutral elements for the reduction.
        valid_lowers = jnp.where(is_empty, jnp.inf, out_partitioned.lower)
        valid_uppers = jnp.where(is_empty, -jnp.inf, out_partitioned.upper)

        # jax.debug.print(
        #     "int_eval_partitioned: {}\n dx_intervals: {}\nis_empty: {},\nout_partitioned: {},\nvalid_lowers: {},\nvalid_uppers: {}",
        #     int_eval_partitioned,
        #     dx_intervals,
        #     is_empty,
        #     out_partitioned,
        #     valid_lowers,
        #     valid_uppers,
        # )

        return irx.interval(jnp.min(valid_lowers), jnp.max(valid_uppers))

    eval = register_inclusion_primitive(incl_fn)(eval_fn)

    return eval
