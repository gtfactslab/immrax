import jax
from jax import core
from jax._src import api
import jax.numpy as jnp
from jax.interpreters import mlir, ad, batching
import numpy as onp
from immrax.inclusion.interval import interval, Interval
from immrax.inclusion import nif
from immrax.inclusion import jacobian as jif

polynomial_p = jax.extend.core.Primitive("polynomial")

impl = jnp.polyval


def polynomial(a, x):
    return polynomial_p.bind(a, x)


def polynomial_impl(a, x):
    return impl(a, x)
    # return jnp.sum([x**n for n in reversed(range(a.shape[0]))])


def polynomial_batch(vector_arg_values, batch_axes):
    res = jax.vmap(polynomial_impl, batch_axes)(*vector_arg_values)
    return res, 0


def polynomial_abstract_eval(a, x):
    try:
        # FIXME: this is flat out wrong. a cannot be more than 1D consistently
        # jnp.polyval broadcasts over the leading dimensions of `a` and all of `x`'s dimensions.
        # The output shape is the result of broadcasting `a.shape[:-1]` with `x.shape`.
        out_shape = jnp.broadcast_shapes(a.shape[:-1], x.shape)
    except ValueError:
        raise ValueError(
            f"Incompatible shapes for polynomial evaluation: cannot broadcast polynomial shape {a.shape[:-1]} with input shape {x.shape}."
        )
    return core.ShapedArray(out_shape, x.dtype)


polynomial_lowering = mlir.lower_fun(polynomial_impl, False)


def polynomial_inclusion(a, x):
    # if not isinstance(a, Interval) or jnp.allclose(a.lower, a.upper) :
    if True:
        # TODO: Can we make this static wrt a somehow, to avoid recomputing the critical points?
        """Minimal inclusion function for constant coefficient a.

        ulf = min_{x \\in [ulx, olx]} f(a, x)
        olf = max_{x \\in [ulx, olx]} f(a, x)

        Since f is a polynomial, check critical points and endpoints.
        """
        if isinstance(a, Interval):
            a = a.lower

        ad = jnp.polyder(a)
        ad_roots = jnp.roots(ad, strip_zeros=False)
        # Critical points and values
        # crit = jnp.real(ad_roots[jnp.isreal(ad_roots)])
        # crit = ad_roots[jnp.where(jnp.isreal(ad_roots))]
        # crit = ad_roots

        crit_vals = jnp.real(jax.vmap(jnp.polyval, in_axes=(None, 0))(a, ad_roots))
        crit_in_x = jax.vmap(
            lambda crit: jnp.logical_and(
                jnp.logical_and(
                    crit > jnp.atleast_1d(x.lower), crit < jnp.atleast_1d(x.upper)
                ),
                jnp.isreal(crit),
            )
        )(ad_roots)

        end_vals = jnp.array(
            [
                jnp.polyval(a, jnp.atleast_1d(x.lower)),
                jnp.polyval(a, jnp.atleast_1d(x.upper)),
            ]
        )

        # print(f"{ad_roots.shape=}")
        # print(f"{end_vals.shape=}")
        # print(f"{crit_vals[:, None].shape=},\n{crit_in_x.shape=}")
        l_vals = jnp.concatenate(
            (end_vals, jnp.where(crit_in_x, crit_vals[:, None], jnp.inf))
        )
        u_vals = jnp.concatenate(
            (end_vals, jnp.where(crit_in_x, crit_vals[:, None], -jnp.inf))
        )

        return interval(jnp.min(l_vals, axis=0), jnp.max(u_vals, axis=0))

    else:
        """Otherwise, simply use natural inclusion function."""
        print("Using natural inclusion function for polynomial primitive.")
        return nif.natif(polynomial_impl)(a, x)


polynomial_p.def_impl(polynomial_impl)  # Primal evaluation (concrete implementation)
polynomial_p.def_abstract_eval(
    polynomial_abstract_eval
)  # Abstract evaluation (shape and dtype propagation)
mlir.register_lowering(polynomial_p, polynomial_lowering)  # JIT lowering to XLA


def polynomial_p_jvp_a(tangent_a, a, x):
    tangent_x_zero = jax.tree_util.tree_map(jnp.zeros_like, x)
    _, tangent_out = jax.jvp(impl, (a, x), (tangent_a, tangent_x_zero))
    return tangent_out


def polynomial_p_jvp_x(tangent_x, a, x):
    tangent_a_zero = jax.tree_util.tree_map(jnp.zeros_like, a)
    _, tangent_out = jax.jvp(impl, (a, x), (tangent_a_zero, tangent_x))
    return tangent_out


batching.primitive_batchers[polynomial_p] = polynomial_batch  # vmap batching rule
ad.defjvp(polynomial_p, polynomial_p_jvp_a, polynomial_p_jvp_x)  # autodiff jvp rule
nif.inclusion_registry[polynomial_p] = (
    polynomial_inclusion  # Inclusion function (natif support)
)


if __name__ == "__main__":
    # Test the polynomial primitive

    a = jnp.array([1.0, 3.0, 5.0])
    x = jnp.array([2.0])

    with jax.disable_jit():
        assert jnp.allclose(jnp.polyval(a, x), polynomial(a, x))
        assert jnp.allclose(jax.jit(polynomial)(a, x), polynomial(a, x))
        assert jnp.allclose(
            jax.jacfwd(jnp.polyval, 0)(a, x), jax.jacfwd(polynomial, 0)(a, x)
        )
        assert jnp.allclose(
            jax.jacfwd(jnp.polyval, 1)(a, x), jax.jacfwd(polynomial, 1)(a, x)
        )
        assert jnp.allclose(
            jax.jacfwd(polynomial, 0)(a, x), jax.jit(jax.jacfwd(polynomial, 0))(a, x)
        )
        assert jnp.allclose(
            jax.jacfwd(polynomial, 1)(a, x), jax.jit(jax.jacfwd(polynomial, 1))(a, x)
        )
        # assert jnp.allclose(
        #     jax.jacrev(jnp.polyval, 0)(a, x), jax.jacrev(polynomial, 0)(a, x)
        # )
        # assert jnp.allclose(
        #     jax.jacrev(jnp.polyval, 1)(a, x), jax.jacrev(polynomial, 1)(a, x)
        # )
        # assert jnp.allclose(
        #     jax.jacrev(polynomial, 0)(a, x), jax.jit(jax.jacrev(polynomial, 0))(a, x)
        # )
        # assert jnp.allclose(
        #     jax.jacrev(polynomial, 1)(a, x), jax.jit(jax.jacrev(polynomial, 1))(a, x)
        # )
    print("Passed all tests of primitive functionality.")

    """
    We need to disable JIT for this inclusion function to work as I expected it to.
    This is actually an important point to realize.

    While polynomial_p is a primitive on its own, when JITed, it is compiled into its binded implementation.
    In this case, this uses scan_p, which is not yet defined in the inclusion module.
    The bigger problem, however, is that we expect derivatives to use polynomial_p so that the good inclusion function is used.

    This is indicative of a possible problem with the current design---we might benefit from avoiding a
        JIT compilation of jacfwd/jacrev when we call these primitives in jacM.
        jacfwd/jacrev internally use vmap, which JITs the function.
    """

    with jax.disable_jit():
        # Test the inclusion function
        ix = interval(jnp.array([-10.0]), jnp.array([10.0]))
        print(nif.natif(polynomial)(a, ix))
        print(nif.natif(jax.jacfwd(polynomial, 1))(a, ix))
        # print(jif.jacM(polynomial)(a, ix))
