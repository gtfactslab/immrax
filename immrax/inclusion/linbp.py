from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax._src import ad_util, source_info_util
from jax._src.core import (
    Atom,
    Jaxpr,
    Literal,
    Var,
    clean_up_dead_vars,
    last_used,
    typecheck,
)
from jax._src import config
from jax._src.util import safe_map
from jax.tree_util import register_pytree_node_class

from immrax.inclusion.interval import Interval, interval

"""
Forward linear bound propagation through a JAX function via Jaxpr interpretation.

For each intermediate variable y of shape S, we maintain:
    lA @ x_in + lb  <=  y  <=  uA @ x_in + ub
where lA, uA have shape (*S, n_in) and lb, ub have shape S.

We also maintain concrete bounds l, u (valid IBP bounds) for ReLU relaxation.
"""


@register_pytree_node_class
class LinearBound:
    """LinearBound: affine bounds as a function of network input x_in.

    For a variable y of shape S, represents:
        lA @ x_in + lb  <=  y  <=  uA @ x_in + ub

    where lA, uA have shape (*S, n_in) and lb, ub have shape S.
    Concrete bounds l, u (valid for any x_in in [x_lb, x_ub]) are
    also tracked for use in activation relaxations.
    """

    lA: jax.Array  # (*S, n_in)
    lb: jax.Array  # S
    uA: jax.Array  # (*S, n_in)
    ub: jax.Array  # S
    l: jax.Array  # S  concrete lower
    u: jax.Array  # S  concrete upper

    def __init__(
        self,
        lA: jax.Array,
        lb: jax.Array,
        uA: jax.Array,
        ub: jax.Array,
        l: jax.Array,
        u: jax.Array,
    ) -> None:
        self.lA = lA
        self.lb = lb
        self.uA = uA
        self.ub = ub
        self.l = l
        self.u = u

    def tree_flatten(self):
        return ((self.lA, self.lb, self.uA, self.ub, self.l, self.u), "LinearBound")

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    @property
    def shape(self):
        return self.l.shape

    @property
    def n_in(self):
        return self.lA.shape[-1]


# ---------------------------------------------------------------------------
# Registry and interpreter
# ---------------------------------------------------------------------------

linbp_registry = {}

# Primitives that recurse into inner jaxprs — these need tighten_bounds/x_lb/x_ub
# forwarded so they can pass them to nested _linbp_jaxpr calls.
_linbp_recursive_prims: set = set()

# Primitives where calling _concretize after the handler gives strictly tighter
# concrete bounds than the IBP l/u already computed by the handler.
#
# Only dot_general_p qualifies: IBP computes W_pos @ l + W_neg @ u treating each
# output neuron independently, but the accumulated affine bound (lA @ x + lb)
# respects that all output neurons share the same input x — so evaluating it at
# the input box recovers cross-neuron correlations that IBP discards.
#
# Activation primitives (max_p, min_p, logistic_p) do NOT benefit: their
# post-activation affine upper bound always evaluates to exactly max/min(u, c)
# at the input box, which matches the IBP concrete bound already stored in u.
# Structural primitives (reshape, broadcast, transpose, etc.) are pure shape
# operations whose concrete bounds are invariant to tightening.
_linbp_tighten_prims: set = {lax.dot_general_p}


def _linbp_jaxpr(
    jaxpr: Jaxpr,
    consts,
    *args,
    relu_mode: str = "adaptive",
    propagate_source_info=True,
    tighten_bounds: bool = True,
    x_lb=None,
    x_ub=None,
) -> list[Any]:
    def read(v: Atom) -> Any:
        return v.val if isinstance(v, Literal) else env[v]

    def write(v: Var, val: Any) -> None:
        if config.enable_checks.value and not config.dynamic_shapes.value:
            assert typecheck(v.aval, val), (v.aval, val)
        env[v] = val

    def _tighten(a):
        """Intersect IBP concrete bounds with affine-evaluated bounds."""
        if not isinstance(a, LinearBound) or x_lb is None:
            return a
        concrete = _concretize(a, x_lb, x_ub)
        return LinearBound(
            a.lA, a.lb, a.uA, a.ub,
            jnp.maximum(a.l, concrete.lower),
            jnp.minimum(a.u, concrete.upper),
        )

    env: dict[Var, Any] = {}
    safe_map(write, jaxpr.constvars, consts)
    safe_map(write, jaxpr.invars, args)
    lu = last_used(jaxpr)

    for eqn in jaxpr.eqns:
        name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
        traceback = eqn.source_info.traceback if propagate_source_info else None
        with source_info_util.user_context(traceback, name_stack=name_stack):
            invars = safe_map(read, eqn.invars)
            if any(isinstance(v, LinearBound) for v in invars):
                if eqn.primitive not in linbp_registry:
                    raise NotImplementedError(f"{eqn.primitive} not in linbp_registry")
                # Pass eqn.params directly (avoids get_bind_params moving things like
                # call_jaxpr/num_consts out of kwargs and into subfuns).
                # Only recursive primitives (jit_p, custom_jvp_call_p) need
                # tighten_bounds/x_lb/x_ub — they forward them to nested jaxpr calls.
                if eqn.primitive in _linbp_recursive_prims:
                    extra = dict(tighten_bounds=tighten_bounds, x_lb=x_lb, x_ub=x_ub)
                else:
                    extra = {}
                ans = linbp_registry[eqn.primitive](
                    *invars, relu_mode=relu_mode, **extra, **eqn.params,
                )
                if tighten_bounds and eqn.primitive in _linbp_tighten_prims:
                    if eqn.primitive.multiple_results:
                        ans = [_tighten(a) for a in ans]
                    else:
                        ans = _tighten(ans)
            else:
                subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
                ans = eqn.primitive.bind(*subfuns, *invars, **bind_params)
        if eqn.primitive.multiple_results:
            safe_map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)
        clean_up_dead_vars(eqn, env, lu)

    return safe_map(read, jaxpr.outvars)


# ---------------------------------------------------------------------------
# Helper: concretize LinearBound to Interval using x_lb, x_ub
# ---------------------------------------------------------------------------


def _concretize(lb: LinearBound, x_lb: jax.Array, x_ub: jax.Array) -> Interval:
    """Evaluate the affine bounds at (x_lb, x_ub) to get concrete interval."""
    lAp = jnp.clip(lb.lA, 0, None)
    lAn = jnp.clip(lb.lA, None, 0)
    uAp = jnp.clip(lb.uA, 0, None)
    uAn = jnp.clip(lb.uA, None, 0)
    # lower = lA @ x gives the minimum over x in [x_lb, x_ub]
    lower = (lAp * x_lb + lAn * x_ub).sum(-1) + lb.lb
    upper = (uAp * x_ub + uAn * x_lb).sum(-1) + lb.ub
    return interval(lower, upper)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _linbp_add_p(x, y, *, relu_mode, **kwargs):
    if isinstance(x, LinearBound) and isinstance(y, LinearBound):
        return LinearBound(
            lA=x.lA + y.lA,
            lb=x.lb + y.lb,
            uA=x.uA + y.uA,
            ub=x.ub + y.ub,
            l=x.l + y.l,
            u=x.u + y.u,
        )
    elif isinstance(x, LinearBound):
        # y is a plain array (bias, etc.)
        return LinearBound(
            lA=x.lA,
            lb=x.lb + y,
            uA=x.uA,
            ub=x.ub + y,
            l=x.l + y,
            u=x.u + y,
        )
    else:
        # x is a plain array, y is LinearBound
        return LinearBound(
            lA=y.lA,
            lb=x + y.lb,
            uA=y.uA,
            ub=x + y.ub,
            l=x + y.l,
            u=x + y.u,
        )


linbp_registry[lax.add_p] = _linbp_add_p
linbp_registry[ad_util.add_any_p] = _linbp_add_p


def _linbp_neg_p(x, *, relu_mode, **kwargs):
    return LinearBound(
        lA=-x.uA,
        lb=-x.ub,
        uA=-x.lA,
        ub=-x.lb,
        l=-x.u,
        u=-x.l,
    )


linbp_registry[lax.neg_p] = _linbp_neg_p


def _linbp_sub_p(x, y, *, relu_mode, **kwargs):
    if isinstance(y, LinearBound):
        neg_y = _linbp_neg_p(y, relu_mode=relu_mode)
        return _linbp_add_p(x, neg_y, relu_mode=relu_mode)
    else:
        # x is LinearBound, y is array (may be a TypedNdArray from equinox)
        return _linbp_add_p(x, jnp.negative(jnp.asarray(y)), relu_mode=relu_mode)


linbp_registry[lax.sub_p] = _linbp_sub_p


def _linbp_mul_p(x, y, *, relu_mode, **kwargs):
    # One must be a non-LinearBound (constant scaling)
    if isinstance(x, LinearBound) and not isinstance(y, LinearBound):
        c = y
        cp = jnp.clip(c, 0, None)
        cn = jnp.clip(c, None, 0)
        # c has shape S; lA has shape (*S, n_in) — need [..., None] for broadcast
        uA = cp[..., None] * x.uA + cn[..., None] * x.lA
        lA = cp[..., None] * x.lA + cn[..., None] * x.uA
        ub = cp * x.ub + cn * x.lb
        lb = cp * x.lb + cn * x.ub
        l = jnp.minimum(c * x.l, c * x.u)
        u = jnp.maximum(c * x.l, c * x.u)
        return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=l, u=u)
    elif not isinstance(x, LinearBound) and isinstance(y, LinearBound):
        return _linbp_mul_p(y, x, relu_mode=relu_mode)
    else:
        # Both LinearBound: fall back to IBP using concrete bounds
        l = jnp.minimum(
            jnp.minimum(x.l * y.l, x.l * y.u),
            jnp.minimum(x.u * y.l, x.u * y.u),
        )
        u = jnp.maximum(
            jnp.maximum(x.l * y.l, x.l * y.u),
            jnp.maximum(x.u * y.l, x.u * y.u),
        )
        n_in = x.n_in
        S = l.shape
        return LinearBound(
            lA=jnp.zeros((*S, n_in)),
            lb=l,
            uA=jnp.zeros((*S, n_in)),
            ub=u,
            l=l,
            u=u,
        )


linbp_registry[lax.mul_p] = _linbp_mul_p


def _linbp_dot_general_p(x, y, *, relu_mode, dimension_numbers, **kwargs):
    (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
    dim_nums = dimension_numbers

    if isinstance(x, LinearBound) and not isinstance(y, LinearBound):
        # x is LinearBound, y (W) is constant: e.g. x @ W  (equinox linear)
        W = y
        Wp = jnp.clip(W, 0, None)
        Wn = jnp.clip(W, None, 0)
        # lA_x has shape (*S_x, n_in); dot_general contracts over S_x axes
        # The n_in axis is free (not in dimension_numbers), so this is correct
        uA = lax.dot_general(x.uA, Wp, dim_nums) + lax.dot_general(x.lA, Wn, dim_nums)
        lA = lax.dot_general(x.lA, Wp, dim_nums) + lax.dot_general(x.uA, Wn, dim_nums)
        ub = lax.dot_general(x.ub, Wp, dim_nums) + lax.dot_general(x.lb, Wn, dim_nums)
        lb = lax.dot_general(x.lb, Wp, dim_nums) + lax.dot_general(x.ub, Wn, dim_nums)
        # Concrete bounds
        ul = lax.dot_general(x.u, Wp, dim_nums) + lax.dot_general(x.l, Wn, dim_nums)
        ll = lax.dot_general(x.l, Wp, dim_nums) + lax.dot_general(x.u, Wn, dim_nums)
        return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=ll, u=ul)

    elif not isinstance(x, LinearBound) and isinstance(y, LinearBound):
        # x (W) is constant, y is LinearBound: e.g. W @ y
        W = x
        Wp = jnp.clip(W, 0, None)
        Wn = jnp.clip(W, None, 0)
        uA = lax.dot_general(Wp, y.uA, dim_nums) + lax.dot_general(Wn, y.lA, dim_nums)
        lA = lax.dot_general(Wp, y.lA, dim_nums) + lax.dot_general(Wn, y.uA, dim_nums)
        ub = lax.dot_general(Wp, y.ub, dim_nums) + lax.dot_general(Wn, y.lb, dim_nums)
        lb = lax.dot_general(Wp, y.lb, dim_nums) + lax.dot_general(Wn, y.ub, dim_nums)
        ul = lax.dot_general(Wp, y.u, dim_nums) + lax.dot_general(Wn, y.l, dim_nums)
        ll = lax.dot_general(Wp, y.l, dim_nums) + lax.dot_general(Wn, y.u, dim_nums)
        return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=ll, u=ul)

    else:
        # Both LinearBound: IBP fallback
        n_in = x.n_in
        ll = (
            lax.dot_general(jnp.clip(x.l, 0, None), y.l, dim_nums)
            + lax.dot_general(jnp.clip(x.l, None, 0), y.u, dim_nums)
            + lax.dot_general(jnp.clip(x.u, 0, None), y.l, dim_nums)
            + lax.dot_general(jnp.clip(x.u, None, 0), y.u, dim_nums)
        )
        # Simple IBP
        corners = [
            lax.dot_general(x.l, y.l, dim_nums),
            lax.dot_general(x.l, y.u, dim_nums),
            lax.dot_general(x.u, y.l, dim_nums),
            lax.dot_general(x.u, y.u, dim_nums),
        ]
        ll = jnp.minimum(
            jnp.minimum(corners[0], corners[1]), jnp.minimum(corners[2], corners[3])
        )
        ul = jnp.maximum(
            jnp.maximum(corners[0], corners[1]), jnp.maximum(corners[2], corners[3])
        )
        S = ll.shape
        return LinearBound(
            lA=jnp.zeros((*S, n_in)),
            lb=ll,
            uA=jnp.zeros((*S, n_in)),
            ub=ul,
            l=ll,
            u=ul,
        )


linbp_registry[lax.dot_general_p] = _linbp_dot_general_p


def _linbp_broadcast_in_dim_p(x, *, relu_mode, shape, broadcast_dimensions, **kwargs):
    n_in = x.n_in
    new_shape = (*shape, n_in)
    new_broadcast_dims = tuple(list(broadcast_dimensions) + [len(shape)])
    lA = lax.broadcast_in_dim(x.lA, new_shape, new_broadcast_dims)
    uA = lax.broadcast_in_dim(x.uA, new_shape, new_broadcast_dims)
    lb = lax.broadcast_in_dim(x.lb, shape, broadcast_dimensions)
    ub = lax.broadcast_in_dim(x.ub, shape, broadcast_dimensions)
    l = lax.broadcast_in_dim(x.l, shape, broadcast_dimensions)
    u = lax.broadcast_in_dim(x.u, shape, broadcast_dimensions)
    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=l, u=u)


linbp_registry[lax.broadcast_in_dim_p] = _linbp_broadcast_in_dim_p


def _linbp_reshape_p(x, *, relu_mode, new_sizes, dimensions=None, **kwargs):
    n_in = x.n_in
    lA = x.lA.reshape(*new_sizes, n_in)
    uA = x.uA.reshape(*new_sizes, n_in)
    lb = x.lb.reshape(new_sizes)
    ub = x.ub.reshape(new_sizes)
    l = x.l.reshape(new_sizes)
    u = x.u.reshape(new_sizes)
    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=l, u=u)


linbp_registry[lax.reshape_p] = _linbp_reshape_p


def _linbp_transpose_p(x, *, relu_mode, permutation, **kwargs):
    perm = list(permutation)
    n_in_axis = len(perm)
    lA = jnp.transpose(x.lA, perm + [n_in_axis])
    uA = jnp.transpose(x.uA, perm + [n_in_axis])
    lb = jnp.transpose(x.lb, perm)
    ub = jnp.transpose(x.ub, perm)
    l = jnp.transpose(x.l, perm)
    u = jnp.transpose(x.u, perm)
    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=l, u=u)


linbp_registry[lax.transpose_p] = _linbp_transpose_p


def _linbp_squeeze_p(x, *, relu_mode, dimensions, **kwargs):
    n_in = x.n_in
    # Axes to squeeze in the data array; don't squeeze the n_in axis
    data_ndim = x.lA.ndim - 1  # ndim of shape S
    lA = jnp.squeeze(x.lA, axis=tuple(dimensions))
    uA = jnp.squeeze(x.uA, axis=tuple(dimensions))
    lb = jnp.squeeze(x.lb, axis=tuple(dimensions))
    ub = jnp.squeeze(x.ub, axis=tuple(dimensions))
    l = jnp.squeeze(x.l, axis=tuple(dimensions))
    u = jnp.squeeze(x.u, axis=tuple(dimensions))
    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=l, u=u)


linbp_registry[lax.squeeze_p] = _linbp_squeeze_p


def _linbp_convert_element_type_p(x, *, relu_mode, new_dtype, **kwargs):
    lA = x.lA.astype(new_dtype)
    uA = x.uA.astype(new_dtype)
    lb = x.lb.astype(new_dtype)
    ub = x.ub.astype(new_dtype)
    l = x.l.astype(new_dtype)
    u = x.u.astype(new_dtype)
    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=l, u=u)


linbp_registry[lax.convert_element_type_p] = _linbp_convert_element_type_p


def _linbp_max_p(x, y, *, relu_mode, **kwargs):
    """General max(x, c) handler for any constant c.

    Upper bound: chord from (l, c) to (u, u) for active neurons.
    Lower bound for active neurons is selected by relu_mode:
      'same-slope' — parallel to upper, tight at kink x = c
      'adaptive'   — slope 0 or 1 based on area heuristic (l+u >= 2c → slope 1)
      'zero'       — slope 0 (constant c)
      'one'        — slope 1 (identity)
    Both alpha values are always >= 0, so upper/lower A-matrices are used
    in the tightest direction.
    """
    if isinstance(x, LinearBound) and not isinstance(y, LinearBound):
        lb_in = x
        c = jnp.asarray(y)
    elif isinstance(y, LinearBound) and not isinstance(x, LinearBound):
        lb_in = y
        c = jnp.asarray(x)
    else:
        # Both LinearBound: IBP fallback
        n_in = x.n_in
        l = jnp.maximum(x.l, y.l)
        u = jnp.maximum(x.u, y.u)
        S = l.shape
        return LinearBound(
            lA=jnp.zeros((*S, n_in)),
            lb=l,
            uA=jnp.zeros((*S, n_in)),
            ub=u,
            l=l,
            u=u,
        )

    l, u = lb_in.l, lb_in.u

    on = l >= c        # always above threshold: max(x, c) = x
    off = u <= c       # always below threshold: max(x, c) = c
    active = ~on & ~off

    safe_denom = jnp.where(active, u - l, 1.0)

    # Upper bound: chord from (l, c) to (u, u)
    alpha_u_act = (u - c) / safe_denom
    alpha_u = jnp.where(on, 1.0, jnp.where(off, 0.0, alpha_u_act))
    beta_u = jnp.where(on, 0.0, jnp.where(off, c, c - alpha_u * l))

    # Lower bound for active neurons (on/off are always identity/constant)
    if relu_mode == "same-slope":
        alpha_l_act = alpha_u_act
        beta_l_act = c * (1.0 - alpha_u_act)
    elif relu_mode == "adaptive":
        use_id = l + u >= 2.0 * c
        alpha_l_act = jnp.where(use_id, 1.0, 0.0)
        beta_l_act = jnp.where(use_id, 0.0, c)
    elif relu_mode == "zero":
        alpha_l_act = 0.0
        beta_l_act = c
    elif relu_mode == "one":
        alpha_l_act = 1.0
        beta_l_act = 0.0
    else:
        raise ValueError(f"Unknown relu_mode: {relu_mode!r}")

    alpha_l = jnp.where(on, 1.0, jnp.where(off, 0.0, alpha_l_act))
    beta_l = jnp.where(on, 0.0, jnp.where(off, c, beta_l_act))

    uA = alpha_u[..., None] * lb_in.uA
    ub = alpha_u * lb_in.ub + beta_u
    lA = alpha_l[..., None] * lb_in.lA
    lb = alpha_l * lb_in.lb + beta_l

    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=jnp.maximum(l, c), u=jnp.maximum(u, c))


linbp_registry[lax.max_p] = _linbp_max_p


def _linbp_min_p(x, y, *, relu_mode, **kwargs):
    """General min(x, c) handler for any constant c.

    Lower bound: chord from (l, l) to (u, c) for active neurons.
    Upper bound for active neurons is selected by relu_mode:
      'same-slope' — parallel to lower, tight at kink x = c
      'adaptive'   — slope 0 or 1 based on area heuristic (l+u <= 2c → slope 1)
      'zero'       — slope 0 (constant c)
      'one'        — slope 1 (identity)
    Both alpha values are always >= 0.
    """
    if isinstance(x, LinearBound) and not isinstance(y, LinearBound):
        lb_in = x
        c = jnp.asarray(y)
    elif isinstance(y, LinearBound) and not isinstance(x, LinearBound):
        lb_in = y
        c = jnp.asarray(x)
    else:
        # Both LinearBound: IBP fallback
        n_in = x.n_in
        l = jnp.minimum(x.l, y.l)
        u = jnp.minimum(x.u, y.u)
        S = l.shape
        return LinearBound(
            lA=jnp.zeros((*S, n_in)),
            lb=l,
            uA=jnp.zeros((*S, n_in)),
            ub=u,
            l=l,
            u=u,
        )

    l, u = lb_in.l, lb_in.u

    on = u <= c        # always below threshold: min(x, c) = x
    off = l >= c       # always above threshold: min(x, c) = c
    active = ~on & ~off

    safe_denom = jnp.where(active, u - l, 1.0)

    # Lower bound: chord from (l, l) to (u, c)
    alpha_l_act = (c - l) / safe_denom
    alpha_l = jnp.where(on, 1.0, jnp.where(off, 0.0, alpha_l_act))
    beta_l = jnp.where(on, 0.0, jnp.where(off, c, l * (1.0 - alpha_l_act)))

    # Upper bound for active neurons (on/off are always identity/constant)
    if relu_mode == "same-slope":
        alpha_u_act = alpha_l_act
        beta_u_act = c * (1.0 - alpha_l_act)
    elif relu_mode == "adaptive":
        use_id = l + u <= 2.0 * c
        alpha_u_act = jnp.where(use_id, 1.0, 0.0)
        beta_u_act = jnp.where(use_id, 0.0, c)
    elif relu_mode == "zero":
        alpha_u_act = 0.0
        beta_u_act = c
    elif relu_mode == "one":
        alpha_u_act = 1.0
        beta_u_act = 0.0
    else:
        raise ValueError(f"Unknown relu_mode: {relu_mode!r}")

    alpha_u = jnp.where(on, 1.0, jnp.where(off, 0.0, alpha_u_act))
    beta_u = jnp.where(on, 0.0, jnp.where(off, c, beta_u_act))

    uA = alpha_u[..., None] * lb_in.uA
    ub = alpha_u * lb_in.ub + beta_u
    lA = alpha_l[..., None] * lb_in.lA
    lb = alpha_l * lb_in.lb + beta_l

    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=jnp.minimum(l, c), u=jnp.minimum(u, c))


linbp_registry[lax.min_p] = _linbp_min_p


def _linbp_logistic_p(x, *, relu_mode, **kwargs):
    """Sigmoid handler using a chord-based linear relaxation.

    For sigmoid σ on [l, u] we use the chord slope α = (σ(u)−σ(l))/(u−l)
    as the common linear coefficient (preserving linear information from
    earlier layers).  The tightest valid upper/lower intercepts for slope α
    are achieved at the critical points where σ'(x*) = α, i.e.
        σ(x*) = (1 ± √(1−4α)) / 2
    giving x* = log(σ(x*)/(1−σ(x*))).

    Upper intercept β_u = max_{x∈[l,u]} (σ(x) − α·x)
      → achieved at x*_upper (≥ 0, concave region) if it lies in [l, u],
        otherwise at the chord intercept (= σ(l)−α·l).

    Lower intercept β_l = min_{x∈[l,u]} (σ(x) − α·x)
      → achieved at x*_lower (≤ 0, convex region) if it lies in [l, u],
        otherwise at the chord intercept.

    Since α ≥ 0, the lA/uA slots keep their linear meaning (lA is used for
    the lower bound, uA for the upper bound), scaled by α.
    """
    l, u = x.l, x.u
    sig_l, sig_u = jax.nn.sigmoid(l), jax.nn.sigmoid(u)

    # Chord slope α ∈ (0, 0.25]; falls back to σ'(l) for degenerate intervals.
    degenerate = jnp.abs(u - l) < 1e-8
    safe_denom = jnp.where(degenerate, 1.0, u - l)
    alpha = jnp.where(degenerate, sig_l * (1.0 - sig_l), (sig_u - sig_l) / safe_denom)

    # Chord intercept (= σ(l)−α·l = σ(u)−α·u by construction)
    beta_chord = sig_l - alpha * l

    # Critical sigmoid values where σ'(x*) = α → σ(x*)(1−σ(x*)) = α
    # disc = √(1−4α) ≥ 0 because α ≤ max(σ') = 0.25
    disc = jnp.sqrt(jnp.clip(1.0 - 4.0 * alpha, 0.0))

    # Upper critical point in concave region (x*_upper ≥ 0)
    sig_xu = (1.0 + disc) / 2.0
    x_upper = jnp.log(sig_xu) - jnp.log(1.0 - sig_xu)   # logit
    beta_u_crit = sig_xu - alpha * x_upper

    # Lower critical point in convex region (x*_lower ≤ 0)
    sig_xl = (1.0 - disc) / 2.0
    x_lower = jnp.log(sig_xl) - jnp.log(1.0 - sig_xl)   # logit
    beta_l_crit = sig_xl - alpha * x_lower

    # Use the critical-point intercept only when the critical point is in [l, u].
    # x*_upper ≥ 0 ≥ l always, so the only check needed is x*_upper ≤ u.
    # x*_lower ≤ 0 ≤ u always, so the only check needed is x*_lower ≥ l.
    beta_u = jnp.where(x_upper <= u, beta_u_crit, beta_chord)
    beta_l = jnp.where(x_lower >= l, beta_l_crit, beta_chord)

    # α ≥ 0, so upper bound uses uA_x (upper side) and lower bound uses lA_x.
    uA = alpha[..., None] * x.uA
    ub = alpha * x.ub + beta_u
    lA = alpha[..., None] * x.lA
    lb = alpha * x.lb + beta_l

    return LinearBound(lA=lA, lb=lb, uA=uA, ub=ub, l=sig_l, u=sig_u)


linbp_registry[lax.logistic_p] = _linbp_logistic_p


def _linbp_jit_p(*args, relu_mode, tighten_bounds=True, x_lb=None, x_ub=None, **bind_params):
    """Handle jit_p (jax >= 0.9) / pjit_p (jax < 0.9) by recursing."""
    bind_jaxpr = bind_params.pop("jaxpr")
    if isinstance(bind_jaxpr, jax.extend.core.ClosedJaxpr):
        bind_jaxpr = bind_jaxpr.jaxpr
    return _linbp_jaxpr(bind_jaxpr, [], *args, relu_mode=relu_mode,
                        tighten_bounds=tighten_bounds, x_lb=x_lb, x_ub=x_ub)


_jit_primitive = getattr(jax._src.pjit, "jit_p", getattr(jax._src.pjit, "pjit_p", None))
if _jit_primitive is not None:
    linbp_registry[_jit_primitive] = _linbp_jit_p
    _linbp_recursive_prims.add(_jit_primitive)


def _linbp_custom_jvp_call_p(*args, relu_mode, call_jaxpr, num_consts,
                             tighten_bounds=True, x_lb=None, x_ub=None, **bind_params):
    """Handle custom_jvp_call by evaluating the primal call_jaxpr only."""
    if isinstance(call_jaxpr, jax.extend.core.ClosedJaxpr):
        consts = call_jaxpr.consts
        inner_jaxpr = call_jaxpr.jaxpr
    else:
        consts = []
        inner_jaxpr = call_jaxpr
    # Leading num_consts args come from the enclosing closure; remainder are invars
    extra_consts = list(args[:num_consts])
    actual_args = args[num_consts:]
    return _linbp_jaxpr(
        inner_jaxpr, consts + extra_consts, *actual_args, relu_mode=relu_mode,
        tighten_bounds=tighten_bounds, x_lb=x_lb, x_ub=x_ub,
    )


_custom_jvp_call_p = getattr(jax._src.custom_derivatives, "custom_jvp_call_p", None)
if _custom_jvp_call_p is not None:
    linbp_registry[_custom_jvp_call_p] = _linbp_custom_jvp_call_p
    _linbp_recursive_prims.add(_custom_jvp_call_p)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _resolve_linbp_input(inp):
    """Normalize a linbp input to (lb_init, trace_point, x_lb, x_ub).

    Parameters
    ----------
    inp : Interval or LinearBound
        - Interval: converted to an identity LinearBound that represents
          y = x_in exactly.  x_lb/x_ub are set so tighten_bounds works.
        - LinearBound: passed through as-is.  x_lb/x_ub are None, which
          disables tighten_bounds (the original input interval is unknown).

    Returns
    -------
    lb_init : LinearBound
    trace_point : jax.Array
        A concrete array with the shape of f's input, used to trace the Jaxpr.
    x_lb, x_ub : jax.Array or None
        Original input box corners for bound tightening.
    """
    if isinstance(inp, Interval):
        n_in = inp.lower.size
        lb_init = LinearBound(
            lA=jnp.eye(n_in),
            lb=jnp.zeros(n_in),
            uA=jnp.eye(n_in),
            ub=jnp.zeros(n_in),
            l=inp.lower,
            u=inp.upper,
        )
        return lb_init, inp.lower, inp.lower, inp.upper
    elif isinstance(inp, LinearBound):
        # inp.l has the shape of the function's input variable; use it to
        # trace the Jaxpr.  x_lb/x_ub are unknown so tightening is skipped.
        return inp, inp.l, None, None
    else:
        raise TypeError(f"linbp wrapped function expects Interval or LinearBound, got {type(inp)}")


def linbp(f, relu_mode: str = "adaptive", tighten_bounds: bool = True):
    """Function transformation: forward linear bound propagation through f.

    Returns a function that maps an Interval or LinearBound to a LinearBound.

    Parameters
    ----------
    f : callable
        Function to propagate through (e.g. a NeuralNetwork).
    relu_mode : str
        How to relax active (ambiguous) neurons. One of:
        'same-slope' — lower slope parallel to upper, tight at kink
        'adaptive'   — slope 0 or 1 chosen by area heuristic
        'zero'       — lower slope always 0 (constant bound)
        'one'        — lower slope always 1 (identity bound)
    tighten_bounds : bool
        If True (default), after each linear layer the concrete bounds l, u are
        intersected with the affine-evaluated bounds: l ← max(l, lA·x̲ + lb),
        u ← min(u, uA·x̄ + ub).  This gives tighter neuron-status classification
        at subsequent activations.  Only applied when the input is an Interval
        (when the input is a LinearBound the original box is unknown).

    Returns
    -------
    Callable[[Interval | LinearBound], LinearBound]
        Maps an input Interval or LinearBound to a LinearBound representing
        affine bounds: lA @ x + lb <= f(x) <= uA @ x + ub for all x in ix.
    """

    def wrapped(inp) -> LinearBound:
        lb_init, trace_point, x_lb, x_ub = _resolve_linbp_input(inp)
        closed_jaxpr = eqx.filter_make_jaxpr(f)(trace_point)[0]
        outs = _linbp_jaxpr(
            closed_jaxpr.jaxpr, closed_jaxpr.literals, lb_init,
            relu_mode=relu_mode, tighten_bounds=tighten_bounds,
            x_lb=x_lb, x_ub=x_ub,
        )
        return outs[0] if len(outs) == 1 else outs

    return wrapped
