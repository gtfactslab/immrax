"""Interval comparison operations based on Allen's interval algebra.

This module provides interval-aware comparison primitives that work with both
JAX arrays and immrax Intervals, supporting the 13 relations of Allen's
interval algebra.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Bool, Int, Real

from immrax.inclusion import nif
from immrax.inclusion.interval import Interval, interval

# =============================================================================
# IntervalRelation: JAX-compatible Allen's interval algebra relations
# =============================================================================

# Relation bit values
_PRECEDES = 1 << 0
_MEETS = 1 << 1
_OVERLAPS = 1 << 2
_STARTS = 1 << 3
_DURING = 1 << 4
_FINISHES = 1 << 5
_EQUAL = 1 << 6
_PRECEDED_BY = 1 << 7
_MET_BY = 1 << 8
_OVERLAPPED_BY = 1 << 9
_STARTED_BY = 1 << 10
_CONTAINS = 1 << 11
_FINISHED_BY = 1 << 12

# Name lookup for pretty printing
_RELATION_NAMES = {
    _PRECEDES: "PRECEDES",
    _MEETS: "MEETS",
    _OVERLAPS: "OVERLAPS",
    _STARTS: "STARTS",
    _DURING: "DURING",
    _FINISHES: "FINISHES",
    _EQUAL: "EQUAL",
    _PRECEDED_BY: "PRECEDED_BY",
    _MET_BY: "MET_BY",
    _OVERLAPPED_BY: "OVERLAPPED_BY",
    _STARTED_BY: "STARTED_BY",
    _CONTAINS: "CONTAINS",
    _FINISHED_BY: "FINISHED_BY",
}


@register_pytree_node_class
class IntervalRelation:
    """Allen's interval algebra relations as a JAX-compatible bit flag.

    The 13 interval relations can be combined using bitwise OR (|) to create
    composite relations. For example, to check if interval A is before or
    meets interval B: `IntervalRelation.PRECEDES | IntervalRelation.MEETS`

    This class is registered as a JAX pytree, so it can be passed as arguments
    to JIT-compiled functions.

    The relations are:
    - "Lower" relations (A comes before B in some sense):
      PRECEDES, MEETS, OVERLAPS, STARTS, DURING, FINISHES, EQUAL
    - "Higher" relations (converses of the lower ones):
      PRECEDED_BY, MET_BY, OVERLAPPED_BY, STARTED_BY, CONTAINS, FINISHED_BY
    """

    value: Int[Array, "*dims"]

    def __init__(self, value):
        """Create an IntervalRelation from an integer or array value."""
        if isinstance(value, IntervalRelation):
            self.value = value.value
        else:
            self.value = jnp.asarray(value, dtype=jnp.int32)

    # Pytree registration
    def tree_flatten(self):
        return (self.value,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])

    # Bitwise operations
    def __and__(self, other: "IntervalRelation") -> "IntervalRelation":
        other_val = other.value if isinstance(other, IntervalRelation) else other
        return IntervalRelation(self.value & other_val)

    def __or__(self, other: "IntervalRelation") -> "IntervalRelation":
        other_val = other.value if isinstance(other, IntervalRelation) else other
        return IntervalRelation(self.value | other_val)

    def __xor__(self, other: "IntervalRelation") -> "IntervalRelation":
        other_val = other.value if isinstance(other, IntervalRelation) else other
        return IntervalRelation(self.value ^ other_val)

    def __invert__(self) -> "IntervalRelation":
        return IntervalRelation(~self.value & ((1 << 13) - 1))

    def __rand__(self, other) -> "IntervalRelation":
        return self.__and__(other)

    def __ror__(self, other) -> "IntervalRelation":
        return self.__or__(other)

    def __rxor__(self, other) -> "IntervalRelation":
        return self.__xor__(other)

    # Comparison
    def __eq__(self, other) -> jax.Array:
        other_val = other.value if isinstance(other, IntervalRelation) else other
        return self.value == other_val

    def __ne__(self, other) -> jax.Array:
        other_val = other.value if isinstance(other, IntervalRelation) else other
        return self.value != other_val

    def __contains__(self, other: "IntervalRelation") -> jax.Array:
        """Check if this relation mask contains the given relation."""
        other_val = other.value if isinstance(other, IntervalRelation) else other
        return (self.value & other_val) != 0

    def matches(self, mask: "IntervalRelation") -> jax.Array:
        """Check if this relation matches any relation in the given mask."""
        mask_val = mask.value if isinstance(mask, IntervalRelation) else mask
        return (self.value & mask_val) != 0

    # Pretty printing
    def _get_names(self) -> list[str]:
        """Get list of relation names for scalar value."""
        val = int(self.value)
        if val == 0:
            return ["NONE"]
        names = []
        for bit, name in _RELATION_NAMES.items():
            if val & bit:
                names.append(name)
        return names

    def __repr__(self) -> str:
        if self.value.shape == ():
            names = self._get_names()
            return f"IntervalRelation({' | '.join(names)})"
        else:
            return f"IntervalRelation(shape={self.value.shape})"

    def __str__(self) -> str:
        if self.value.shape == ():
            return " | ".join(self._get_names())
        else:
            return f"IntervalRelation[{self.value.shape}]"

    # Integer conversion
    def __int__(self) -> int:
        return int(self.value)

    def __index__(self) -> int:
        return int(self.value)

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    # Class-level relation constants
    PRECEDES: "IntervalRelation"
    MEETS: "IntervalRelation"
    OVERLAPS: "IntervalRelation"
    STARTS: "IntervalRelation"
    DURING: "IntervalRelation"
    FINISHES: "IntervalRelation"
    EQUAL: "IntervalRelation"
    PRECEDED_BY: "IntervalRelation"
    MET_BY: "IntervalRelation"
    OVERLAPPED_BY: "IntervalRelation"
    STARTED_BY: "IntervalRelation"
    CONTAINS: "IntervalRelation"
    FINISHED_BY: "IntervalRelation"

    @classmethod
    def BEFORE(cls) -> "IntervalRelation":
        """A is strictly before B (PRECEDES | MEETS)."""
        return cls.PRECEDES | cls.MEETS

    @classmethod
    def AFTER(cls) -> "IntervalRelation":
        """A is strictly after B (PRECEDED_BY | MET_BY)."""
        return cls.PRECEDED_BY | cls.MET_BY

    @classmethod
    def OVERLAPPING(cls) -> "IntervalRelation":
        """A and B overlap (all except PRECEDES, PRECEDED_BY, MEETS, MET_BY)."""
        return (
            cls.OVERLAPS
            | cls.STARTS
            | cls.DURING
            | cls.FINISHES
            | cls.EQUAL
            | cls.OVERLAPPED_BY
            | cls.STARTED_BY
            | cls.CONTAINS
            | cls.FINISHED_BY
        )

    @classmethod
    def SUBSET(cls) -> "IntervalRelation":
        """A is a subset of B (STARTS | DURING | FINISHES | EQUAL)."""
        return cls.STARTS | cls.DURING | cls.FINISHES | cls.EQUAL

    @classmethod
    def SUPERSET(cls) -> "IntervalRelation":
        """A is a superset of B (STARTED_BY | CONTAINS | FINISHED_BY | EQUAL)."""
        return cls.STARTED_BY | cls.CONTAINS | cls.FINISHED_BY | cls.EQUAL

    @classmethod
    def ALL(cls) -> "IntervalRelation":
        """All possible relations."""
        return cls((1 << 13) - 1)

    @classmethod
    def NONE(cls) -> "IntervalRelation":
        """No relations."""
        return cls(0)


# Initialize class-level constants
IntervalRelation.PRECEDES = IntervalRelation(_PRECEDES)
IntervalRelation.MEETS = IntervalRelation(_MEETS)
IntervalRelation.OVERLAPS = IntervalRelation(_OVERLAPS)
IntervalRelation.STARTS = IntervalRelation(_STARTS)
IntervalRelation.DURING = IntervalRelation(_DURING)
IntervalRelation.FINISHES = IntervalRelation(_FINISHES)
IntervalRelation.EQUAL = IntervalRelation(_EQUAL)
IntervalRelation.PRECEDED_BY = IntervalRelation(_PRECEDED_BY)
IntervalRelation.MET_BY = IntervalRelation(_MET_BY)
IntervalRelation.OVERLAPPED_BY = IntervalRelation(_OVERLAPPED_BY)
IntervalRelation.STARTED_BY = IntervalRelation(_STARTED_BY)
IntervalRelation.CONTAINS = IntervalRelation(_CONTAINS)
IntervalRelation.FINISHED_BY = IntervalRelation(_FINISHED_BY)


# =============================================================================
# interval_compare: Compute Allen relation between two intervals
# =============================================================================


def interval_compare(a: Interval, b: Interval) -> IntervalRelation:
    """Compare two intervals and return their Allen relation.

    This function computes which of Allen's 13 interval relations holds between
    intervals a and b.

    Parameters
    ----------
    a : Interval
        First interval to compare
    b : Interval
        Second interval to compare

    Returns
    -------
    IntervalRelation
        The Allen relation between a and b. For array-valued intervals,
        the result has the same shape as the inputs.

    Examples
    --------
    >>> a = interval(1.0, 2.0)
    >>> b = interval(3.0, 4.0)
    >>> rel = interval_compare(a, b)
    >>> rel == IntervalRelation.PRECEDES
    True
    """
    a = interval(a)
    b = interval(b)

    # Compare endpoints
    al_lt_bl = a.lower < b.lower
    al_eq_bl = a.lower == b.lower
    al_gt_bl = a.lower > b.lower

    au_lt_bu = a.upper < b.upper
    au_eq_bu = a.upper == b.upper
    au_gt_bu = a.upper > b.upper

    au_lt_bl = a.upper < b.lower
    au_eq_bl = a.upper == b.lower
    au_gt_bl = a.upper > b.lower

    al_lt_bu = a.lower < b.upper
    al_eq_bu = a.lower == b.upper
    al_gt_bu = a.lower > b.upper

    # Determine relation based on endpoint comparisons
    # Each relation is mutually exclusive, so we build the result by selecting

    # PRECEDES: A.upper < B.lower
    precedes = au_lt_bl

    # MEETS: A.upper == B.lower (and A.lower < B.lower implied)
    meets = au_eq_bl & al_lt_bl

    # OVERLAPS: A.lower < B.lower, A.upper > B.lower, A.upper < B.upper
    overlaps = al_lt_bl & au_gt_bl & au_lt_bu

    # STARTS: A.lower == B.lower, A.upper < B.upper
    starts = al_eq_bl & au_lt_bu

    # DURING: A.lower > B.lower, A.upper < B.upper
    during = al_gt_bl & au_lt_bu

    # FINISHES: A.lower > B.lower, A.upper == B.upper
    finishes = al_gt_bl & au_eq_bu

    # EQUAL: A.lower == B.lower, A.upper == B.upper
    equal = al_eq_bl & au_eq_bu

    # PRECEDED_BY: B.upper < A.lower (converse of PRECEDES)
    preceded_by = al_gt_bu

    # MET_BY: B.upper == A.lower (converse of MEETS)
    met_by = al_eq_bu & au_gt_bu

    # OVERLAPPED_BY: B.lower < A.lower < B.upper < A.upper
    overlapped_by = al_gt_bl & al_lt_bu & au_gt_bu

    # STARTED_BY: A.lower == B.lower, A.upper > B.upper (converse of STARTS)
    started_by = al_eq_bl & au_gt_bu

    # CONTAINS: A.lower < B.lower, A.upper > B.upper (converse of DURING)
    contains = al_lt_bl & au_gt_bu

    # FINISHED_BY: A.lower < B.lower, A.upper == B.upper (converse of FINISHES)
    finished_by = al_lt_bl & au_eq_bu

    # Build result using conditional selection
    result = jnp.zeros(jnp.broadcast_shapes(a.shape, b.shape), dtype=jnp.int32)
    result = jnp.where(precedes, _PRECEDES, result)
    result = jnp.where(meets, _MEETS, result)
    result = jnp.where(overlaps, _OVERLAPS, result)
    result = jnp.where(starts, _STARTS, result)
    result = jnp.where(during, _DURING, result)
    result = jnp.where(finishes, _FINISHES, result)
    result = jnp.where(equal, _EQUAL, result)
    result = jnp.where(preceded_by, _PRECEDED_BY, result)
    result = jnp.where(met_by, _MET_BY, result)
    result = jnp.where(overlapped_by, _OVERLAPPED_BY, result)
    result = jnp.where(started_by, _STARTED_BY, result)
    result = jnp.where(contains, _CONTAINS, result)
    result = jnp.where(finished_by, _FINISHED_BY, result)

    return IntervalRelation(result)


# =============================================================================
# Custom JAX Primitive for Interval-Aware Less-Than Comparison
# =============================================================================

irx_lt_p = Primitive("irx_lt_p")


def _irx_lt_impl(x, y, relation_mask):
    """Concrete implementation: standard < comparison for arrays."""
    return lax.lt(x, y)


irx_lt_p.def_impl(_irx_lt_impl)


def _irx_lt_abstract_eval(x_aval, y_aval, relation_mask_aval):
    """Abstract evaluation: result is boolean with broadcasted shape."""
    shape = jnp.broadcast_shapes(x_aval.shape, y_aval.shape)
    return jax.core.ShapedArray(shape, jnp.bool_)


irx_lt_p.def_abstract_eval(_irx_lt_abstract_eval)


# MLIR lowering for JIT compilation
def _irx_lt_lowering(ctx, x, y, relation_mask):
    """Lower to standard < comparison for XLA."""
    return mlir.lower_fun(_irx_lt_impl, multiple_results=False)(
        ctx, x, y, relation_mask
    )


mlir.register_lowering(irx_lt_p, _irx_lt_lowering)


# Batching rule for vmap
def _irx_lt_batching(vector_arg_values, batch_axes):
    """Batching rule: vmap over the comparison."""
    x, y, relation_mask = vector_arg_values
    x_bdim, y_bdim, rm_bdim = batch_axes

    # Broadcast relation_mask if not batched but others are
    if rm_bdim is None and (x_bdim is not None or y_bdim is not None):
        # relation_mask is not batched, replicate it
        result = jax.vmap(
            lambda xi, yi: irx_lt_p.bind(xi, yi, relation_mask),
            in_axes=(x_bdim, y_bdim),
        )(x, y)
    else:
        result = jax.vmap(
            lambda xi, yi, rmi: irx_lt_p.bind(xi, yi, rmi),
            in_axes=(x_bdim, y_bdim, rm_bdim),
        )(x, y, relation_mask)

    return result, 0


batching.primitive_batchers[irx_lt_p] = _irx_lt_batching


# JVP rules (comparison has zero gradient)
def _irx_lt_jvp_x(tangent, x, y, relation_mask):
    """JVP for x: comparison has zero tangent."""
    shape = jnp.broadcast_shapes(jnp.shape(x), jnp.shape(y))
    return jnp.zeros(shape, dtype=jnp.bool_)


def _irx_lt_jvp_y(tangent, x, y, relation_mask):
    """JVP for y: comparison has zero tangent."""
    shape = jnp.broadcast_shapes(jnp.shape(x), jnp.shape(y))
    return jnp.zeros(shape, dtype=jnp.bool_)


def _irx_lt_jvp_rm(tangent, x, y, relation_mask):
    """JVP for relation_mask: comparison has zero tangent."""
    shape = jnp.broadcast_shapes(jnp.shape(x), jnp.shape(y))
    return jnp.zeros(shape, dtype=jnp.bool_)


ad.defjvp(irx_lt_p, _irx_lt_jvp_x, _irx_lt_jvp_y, _irx_lt_jvp_rm)


# Inclusion function for interval comparison
def _inclusion_irx_lt_p(x, y, relation_mask) -> jax.Array:
    """Inclusion function for interval-aware less-than comparison.

    This is called when natif encounters irx_lt_p with Interval arguments.
    It computes the Allen relation between the intervals and returns whether
    this relation matches any relation in the mask.

    Returns a boolean array (NOT an Interval), since the comparison result
    is unambiguous once we specify which relations we're checking for.
    """
    # Convert to intervals
    x = interval(x)
    y = interval(y)

    # Compute Allen relation
    relation = interval_compare(x, y)

    # Check if relation matches mask - returns boolean array
    return relation.matches(relation_mask)


nif.inclusion_registry[irx_lt_p] = _inclusion_irx_lt_p


# =============================================================================
# Public API: Comparison Functions
# =============================================================================

# Default relation masks for standard comparisons
_LT_MASK = IntervalRelation.PRECEDES
_LE_MASK = IntervalRelation.PRECEDES | IntervalRelation.MEETS | IntervalRelation.EQUAL
_EQ_MASK = IntervalRelation.EQUAL
_GT_MASK = IntervalRelation.PRECEDED_BY
_GE_MASK = (
    IntervalRelation.PRECEDED_BY | IntervalRelation.MET_BY | IntervalRelation.EQUAL
)
_NE_MASK = ~IntervalRelation.EQUAL & IntervalRelation.ALL()


def lt(
    x: Real[Array, "#*dims"],
    y: Real[Array, "#*dims"],
    relation_mask: IntervalRelation = IntervalRelation.PRECEDES,
) -> Bool[Array, "#*dims"]:
    """Interval-aware less-than comparison.

    For JAX arrays, performs standard < comparison.
    For Intervals, computes the Allen relation and checks if it matches the mask.

    When used inside a function transformed by natif, the inclusion function
    will handle the interval comparison automatically.

    Parameters
    ----------
    x : jax.Array | Interval
        First comparand
    y : jax.Array | Interval
        Second comparand
    relation_mask : IntervalRelation, optional
        Relation mask to check against. Default is PRECEDES.

    Returns
    -------
    jax.Array
        Boolean result of comparison
    """

    return irx_lt_p.bind(jnp.asarray(x), jnp.asarray(y), relation_mask.value)
