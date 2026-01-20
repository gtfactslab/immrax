import jax
import jax.numpy as jnp
from immrax.inclusion.interval import interval, Interval
from immrax.inclusion import nif, custom_if


@custom_if
def polynomial(a, x):
    return jnp.polyval(a, x)


@polynomial.defif
def polynomial_inclusion(a, x):
    # if not isinstance(a, Interval) or jnp.allclose(a.lower, a.upper) :
    # TODO: This only works for constant coefficients. Try to take inclusion for x, natif into inclusion for both
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
