import jax
from jax import core
import jax.numpy as jnp
from jax.interpreters import mlir, ad, batching
from immrax.inclusion.interval import interval, Interval
from immrax.inclusion import nif

polynomial_p = core.Primitive("polynomial")


# @jax.jit
def polynomial(a, x):
    return polynomial_p.bind(a, x)


def polynomial_impl(a, x):
    return jnp.polyval(a, x)
    # return jnp.sum([x**n for n in reversed(range(a.shape[0]))])


def polynomial_batch(vector_arg_values, batch_axes):
    res = jax.vmap(polynomial_impl, batch_axes)(*vector_arg_values)
    return res, 0


def polynomial_abstract_eval(a, x):
    return core.ShapedArray(x.shape, x.dtype)


polynomial_lowering = mlir.lower_fun(polynomial_impl, False)


def polynomial_jvp(primals, tangents):
    """JVP for the polynomial primitive.

    f(a, x) = a[0]*x^(n-1) + a[1]*x^(n-2) + ... + a[n-1]
    (df/da) = [x^(n-1), x^(n-2), ..., 1]
    (df/dx) = (n-1)*a[0]*x^(n-2) + (n-2)*a[1]*x^(n-3) + ... + a[n-2]
            = polynomial(polyder(a), x)
    (df/da)at = polynomial(at, x)
    (df/dx)xt = polynomial(xt*jnp.polyder(a), x)
    """
    a, x = primals
    at, xt = tangents
    primal_out = polynomial(a, x)

    if isinstance(xt, ad.Zero):
        return primal_out, polynomial(at, x)
    elif isinstance(at, ad.Zero):
        return primal_out, polynomial(xt * jnp.polyder(a), x)
    else:
        return primal_out, polynomial(jnp.polyadd(xt * jnp.polyder(a), at), x)


def polynomial_transpose(ct, a, x):
    """Transpose rule for the polynomial primitive.

    Needs to return f^T at the cotangent input. f is linear in a wrt fixed x.
    """
    assert ad.is_undefined_primal(a) and not ad.is_undefined_primal(x)
    if type(ct) is ad.Zero:
        return ad.Zero, None
    else:
        ret = ct * jnp.array([x**n for n in reversed(range(a.aval.shape[0]))]).reshape(
            a.aval.shape
        )
        return ret, None


def polynomial_inclusion(a, x):
    if not isinstance(a, Interval) or jnp.allclose(a.lower, a.upper):
        # TODO: Can we make this static wrt a somehow, to avoid recomputing the critical points?
        """Minimal inclusion function for constant coefficient a.

        ulf = min_{x \in [ulx, olx]} f(a, x)
        olf = max_{x \in [ulx, olx]} f(a, x)

        Since f is a polynomial, check critical points and endpoints.
        """
        if isinstance(a, Interval):
            a = a.lower

        ad = jnp.polyder(a)
        ad_roots = jnp.roots(ad)
        # Critical points and values
        crit = jnp.real(ad_roots[jnp.isreal(ad_roots)])
        crit_vals = jnp.array([jnp.polyval(a, c) for c in crit])

        crit_in_x = jnp.logical_and(crit > x.lower, crit < x.upper)

        end_vals = jnp.array([jnp.polyval(a, x.lower), jnp.polyval(a, x.upper)])

        l_vals = jnp.concatenate((end_vals, jnp.where(crit_in_x, crit_vals, jnp.inf)))
        u_vals = jnp.concatenate((end_vals, jnp.where(crit_in_x, crit_vals, -jnp.inf)))

        return interval(jnp.min(l_vals), jnp.max(u_vals))

    else:
        """Otherwise, simply use natural inclusion function."""
        print("Using natural inclusion function for polynomial primitive.")
        return nif.natif(polynomial_impl)(a, x)


# Primal evaluation (concrete implementation)
polynomial_p.def_impl(polynomial_impl)
# Abstract evaluation (shape and dtype propagation)
polynomial_p.def_abstract_eval(polynomial_abstract_eval)
# JIT lowering to XLA
mlir.register_lowering(polynomial_p, polynomial_lowering)
# Linearizing rule (JVP support)
ad.primitive_jvps[polynomial_p] = polynomial_jvp
# vmap batching rule
batching.primitive_batchers[polynomial_p] = polynomial_batch
# Transpose rule (VJP support)
ad.primitive_transposes[polynomial_p] = polynomial_transpose
# Inclusion function (natif support)
nif.inclusion_registry[polynomial_p] = polynomial_inclusion


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
        assert jnp.allclose(
            jax.jacrev(jnp.polyval, 0)(a, x), jax.jacrev(polynomial, 0)(a, x)
        )
        assert jnp.allclose(
            jax.jacrev(jnp.polyval, 1)(a, x), jax.jacrev(polynomial, 1)(a, x)
        )
        assert jnp.allclose(
            jax.jacrev(polynomial, 0)(a, x), jax.jit(jax.jacrev(polynomial, 0))(a, x)
        )
        assert jnp.allclose(
            jax.jacrev(polynomial, 1)(a, x), jax.jit(jax.jacrev(polynomial, 1))(a, x)
        )
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
        ix = interval(-10.0, 10.0)
        print(nif.natif(polynomial)(a, ix))
        print(nif.natif(jax.jacfwd(polynomial, 1))(a, ix))
        # print(jif.jacM(polynomial)(a, ix))
