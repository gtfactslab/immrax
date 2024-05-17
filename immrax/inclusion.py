from functools import wraps
import jax
import jax.numpy as jnp
from jax import core
from jax import lax
from jax.core import Primitive
from jax import jit, vmap
from jax.typing import ArrayLike
from jax._src.util import safe_map
from jax.tree_util import register_pytree_node_class
from typing import Tuple, Callable, Sequence, Iterable, Any
import numpy as np
from functools import partial
from itertools import accumulate, product
from itertools import permutations as perms
from jax._src import ad_util, source_info_util, config
from jax._src.api import api_boundary
from jax._src.core import Jaxpr, Literal, Var, Atom, typecheck, last_used, clean_up_dead_vars

@register_pytree_node_class
class Interval :
    """Interval: A class to represent an interval in :math:`\\mathbb{R}^n`.
    
    Use the helper functions :func:`interval`, :func:`icentpert`, :func:`i2centpert`, :func:`i2lu`, :func:`i2ut`, and :func:`ut2i` to create and manipulate intervals.
    
    Use the transforms :func:`natif`, :func:`jacif`, :func:`mjacif`, :func:`mjacM`, to create inclusion functions.
    
    Composable with typical jax transforms, such as :func:`jax.jit`, :func:`jax.grad`, and :func:`jax.vmap`.
    """
    lower: jax.Array
    upper: jax.Array
    def __init__(self, lower:jax.Array, upper:jax.Array) -> None:
        self.lower = lower
        self.upper = upper
    def tree_flatten(self) :
        return ((self.lower, self.upper), 'Interval')
    @classmethod
    def tree_unflatten(cls, aux_data, children) :
        return cls(*children)
    
    @property
    def dtype (self) -> jnp.dtype :
        return self.lower.dtype
    
    @property
    def shape (self) -> Tuple[int] :
        return self.lower.shape

    @property
    def width(self) -> jax.Array :
        return self.upper - self.lower

    def __len__ (self) -> int :
        return len(self.lower)

    def reshape(self, *args, **kwargs) :
        return interval(self.lower.reshape(*args, **kwargs), self.upper.reshape(*args, **kwargs))
    
    def atleast_1d(self) -> 'Interval' :
        return interval(jnp.atleast_1d(self.lower), jnp.atleast_1d(self.upper))
    
    def atleast_2d(self) -> 'Interval' :
        return interval(jnp.atleast_2d(self.lower), jnp.atleast_2d(self.upper))
    
    def atleast_3d(self) -> 'Interval' :
        return interval(jnp.atleast_3d(self.lower), jnp.atleast_3d(self.upper))
    
    @property
    def ndim (self) -> int :
        return self.lower.ndim
    
    def transpose (self, *args) -> 'Interval' :
        return Interval(self.lower.transpose(*args), self.upper.transpose(*args))
    @property
    def T (self) -> 'Interval' :
        return self.transpose()
    
    def __and__ (self, other:'Interval') -> 'Interval' :
        return interval(jnp.maximum(self.lower, other.lower), jnp.minimum(self.upper, other.upper))
    
    def __or__ (self, other:'Interval') -> 'Interval' :
        return interval(jnp.minimum(self.lower, other.lower), jnp.maximum(self.upper, other.upper))
  
    def __str__(self) -> str:
        return np.array([[(l,u)] for (l,u) in 
                        zip(self.lower.reshape(-1),self.upper.reshape(-1))], 
                        dtype=np.dtype([('f1',float), ('f2', float)])).reshape(self.shape + (1,)).__str__()
        # return self.lower.__str__() + ' <= x <= ' + self.upper.__str__()
    
    def __repr__(self) -> str:
        return np.array([[(l,u)] for (l,u) in 
                        zip(self.lower.reshape(-1),self.upper.reshape(-1))], 
                        dtype=np.dtype([('f1',float), ('f2', float)])).reshape(self.shape + (1,)).__str__()
                        # dtype=np.dtype([('f1',float), ('f2', float)])).reshape(self.shape + (1,)).__repr__()
        # return self.lower.__str__() + ' <= x <= ' + self.upper.__str__()
    
    def __getitem__(self, i:int) :
        return Interval(self.lower[i], self.upper[i])

# HELPER FUNCTIONS 

def interval (lower:ArrayLike, upper:ArrayLike=None) :
    """interval: Helper to create a Interval from a lower and upper bound.

    Parameters
    ----------
    lower : ArrayLike
        Lower bound of the interval.
    upper : ArrayLike
        Upper bound of the interval. Set to lower bound if None. Defaults to None.
    lower:ArrayLike :

    Returns
    -------
    Interval
        [lower, upper], or [lower, lower] if upper is None.

    """
    if isinstance (lower, Interval) and upper is None :
        return lower
    if upper is None :
        return Interval(jnp.asarray(lower), jnp.asarray(lower))
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)
    if lower.dtype != upper.dtype :
        raise Exception(f'lower and upper dtype should match, {lower.dtype} != {upper.dtype}')
    if lower.shape != upper.shape :
        raise Exception(f'lower and upper shape should match, {lower.shape} != {upper.shape}')
    return Interval(jnp.asarray(lower), jnp.asarray(upper))

def icopy (i:Interval) -> Interval :
    """icopy: Helper to copy an interval.

    Parameters
    ----------
    i : Interval
        interval to copy

    Returns
    -------
    Interval
        copy of the interval

    """
    return Interval(jnp.copy(i.lower), jnp.copy(i.upper))

def icentpert (cent:ArrayLike, pert:ArrayLike) -> Interval :
    """icentpert: Helper to create a Interval from a center of an interval and a perturbation.

    Parameters
    ----------
    cent : ArrayLike
        Center of the interval, i.e., (l + u)/2
    pert : ArrayLike
        l-inf perturbation from the center, i.e., (u - l)/2

    Returns
    -------
    Interval
        Interval [cent - pert, cent + pert]

    """
    cent = jnp.asarray(cent)
    pert = jnp.asarray(pert)
    return interval(cent - pert, cent + pert)


def i2centpert (i:Interval) -> Tuple[jax.Array, jax.Array] :
    """i2centpert: Helper to get the center and perturbation from the center of a Interval.

    Parameters
    ----------
    i : Interval
        _description_

    Returns
    -------
    Tuple[jax.Array, jax.Array]
        ((l + u)/2, (u - l)/2)

    """
    return (i.lower + i.upper)/2, (i.upper - i.lower)/2

def i2lu (i:Interval) -> Tuple[jax.Array, jax.Array] :
    """i2lu: Helper to get the lower and upper bound of a Interval.

    Parameters
    ----------
    interval : Interval
        _description_

    Returns
    -------
    Tuple[jax.Array, jax.Array]
        (l, u)

    """
    return (i.lower, i.upper)

def lu2i (l:jax.Array, u:jax.Array) -> Interval :
    """lu2i: Helper to create a Interval from a lower and upper bound.

    Parameters
    ----------
    l : jax.Array
        Lower bound of the interval.
    u : jax.Array
        Upper bound of the interval.

    Returns
    -------
    Interval
        [l, u]

    """
    return interval(l, u)

def i2ut (i:Interval) -> jax.Array :
    """i2ut: Helper to convert an interval to an upper triangular coordinate in :math:`\\mathbb{R}\\times\\mathbb{R}`.

    Parameters
    ----------
    interval : Interval
        interval to convert

    Returns
    -------
    jax.Array
        upper triangular coordinate in :math:`\\mathbb{R}\\times\\mathbb{R}`

    """
    return jnp.concatenate((i.lower, i.upper))

def ut2i (coordinate:jax.Array, n:int=None) -> Interval :
    """ut2i: Helper to convert an upper triangular coordinate in :math:`\\mathbb{R}\\times\\mathbb{R}` to an interval.

    Parameters
    ----------
    coordinate : jax.Array
        upper triangular coordinate to convert
    n : int
        length of interval, automatically determined if None. Defaults to None.

    Returns
    -------
    Interval
        interval representation of the coordinate

    """
    if n is None :
        n = len(coordinate) // 2
    return interval(coordinate[:n], coordinate[n:])

def izeros (shape:Tuple[int], dtype:np.dtype=jnp.float32) -> Interval :
    """izeros: Helper to create a Interval of zeros.

    Parameters
    ----------
    shape : Tuple[int]
        shape of the interval
    dtype : np.dtype
        dtype of the interval. Defaults to jnp.float32.

    Returns
    -------
    Interval
        interval of zeros

    """
    return interval(jnp.zeros(shape, dtype), jnp.zeros(shape, dtype))

def iconcatenate (intervals:Iterable[Interval], axis:int=0) -> Interval :
    """iconcatenate: Helper to concatenate intervals (cartesian product).

    Parameters
    ----------
    intervals : Iterable[Interval]
        intervals to concatenate
    axis : int
        axis to concatenate on. Defaults to 0.

    Returns
    -------
    Interval
        concatenated interval

    """
    return interval(jnp.concatenate([i.lower for i in intervals], axis=axis),
                    jnp.concatenate([i.upper for i in intervals], axis=axis))

inclusion_registry = {}

def _make_inclusion_passthrough_p (primitive:Primitive) -> Callable[..., Interval] :
    """Creates an inclusion function that applies to the lower and upper bounds individually

    Parameters
    ----------
    primitive : Primitive
        Primitive to wrap

    Returns
    -------
    Callable[..., Interval]
        Inclusion Function to bind in natif

    """
    def _inclusion_p (*args, **kwargs) -> Interval :
        args_l = [(arg.lower if isinstance(arg, Interval) else arg) for arg in args]
        args_u = [(arg.upper if isinstance(arg, Interval) else arg) for arg in args]
        return Interval(primitive.bind(*args_l, **kwargs), primitive.bind(*args_u, **kwargs))
    return _inclusion_p
def _add_passthrough_to_registry (primitive:Primitive) -> None :
    inclusion_registry[primitive] = _make_inclusion_passthrough_p(primitive)
_add_passthrough_to_registry(lax.copy_p)
_add_passthrough_to_registry(lax.reshape_p)
_add_passthrough_to_registry(lax.slice_p)
_add_passthrough_to_registry(lax.dynamic_slice_p)
_add_passthrough_to_registry(lax.squeeze_p)
_add_passthrough_to_registry(lax.transpose_p)
_add_passthrough_to_registry(lax.broadcast_in_dim_p)
_add_passthrough_to_registry(lax.concatenate_p)
_add_passthrough_to_registry(lax.gather_p)
_add_passthrough_to_registry(lax.scatter_p)
if hasattr(lax, 'select_p') :
    _add_passthrough_to_registry(lax.select_p)
if hasattr(lax, 'select_n_p') :
    _add_passthrough_to_registry(lax.select_n_p)
_add_passthrough_to_registry(lax.iota_p)
_add_passthrough_to_registry(lax.eq_p)
_add_passthrough_to_registry(lax.convert_element_type_p)
# *([lax.select_p] if hasattr(lax, 'select_p') else []),
# *([lax.select_n_p] if hasattr(lax, 'select_n_p') else []),
# synthetic_primitives.convert_float32_p,
_add_passthrough_to_registry(lax.reduce_max_p)
_add_passthrough_to_registry(lax.reduce_min_p)
_add_passthrough_to_registry(lax.max_p)
_add_passthrough_to_registry(lax.min_p)
_add_passthrough_to_registry(lax.exp_p)
_add_passthrough_to_registry(jax.experimental.pjit.pjit_p)

def _inclusion_add_p (x:Interval, y:Interval) -> Interval :
    if isinstance(x, Interval) and isinstance (y, Interval) :
        return Interval(x.lower + y.lower, x.upper + y.upper)
    elif isinstance(x, Interval) :
        return Interval(x.lower + y, x.upper + y)
    elif isinstance(y, Interval) :
        return Interval(x + y.lower, x + y.upper)
    else :
        return x + y
inclusion_registry[lax.add_p] = _inclusion_add_p
inclusion_registry[ad_util.add_any_p] = _inclusion_add_p
Interval.__add__ = _inclusion_add_p

def _inclusion_sub_p (x:Interval, y:Interval) -> Interval :
    if isinstance(x, Interval) and isinstance (y, Interval) :
        return Interval(x.lower - y.upper, x.upper - y.lower)
    elif isinstance(x, Interval) :
        return Interval(x.lower - y, x.upper - y)
    elif isinstance(y, Interval) :
        return Interval(x - y.upper, x - y.lower)
    else :
        return x - y
inclusion_registry[lax.sub_p] = _inclusion_sub_p
Interval.__sub__ = _inclusion_sub_p

def _inclusion_neg_p (x:Interval) -> Interval :
    return Interval(-x.upper, -x.lower)
inclusion_registry[lax.neg_p] = _inclusion_neg_p
Interval.__neg__ = _inclusion_neg_p

def _inclusion_mul_p (x:Interval, y:Interval) -> Interval :
    if isinstance(x, Interval) and isinstance(y, Interval) :
        _1 = x.lower*y.lower
        _2 = x.lower*y.upper
        _3 = x.upper*y.lower
        _4 = x.upper*y.upper
        return Interval(jnp.minimum(jnp.minimum(_1,_2),jnp.minimum(_3,_4)),
                        jnp.maximum(jnp.maximum(_1,_2),jnp.maximum(_3,_4)))
    elif isinstance(x,Interval) :
        _1 = x.lower*y
        _2 = x.upper*y
        return Interval(jnp.minimum(_1,_2), jnp.maximum(_1,_2))
    elif isinstance(y,Interval) :
        _1 = x*y.lower
        _2 = x*y.upper
        return Interval(jnp.minimum(_1,_2), jnp.maximum(_1,_2))
    else :
        return x*y
inclusion_registry[lax.mul_p] = _inclusion_mul_p
Interval.__mul__ = _inclusion_mul_p

def _inclusion_div_p (x:Interval, y:Interval) -> Interval :
    if isinstance(x, Interval) and isinstance(y, Interval) :
        return _inclusion_mul_p(x, _inclusion_reciprocal_p(y))
    elif isinstance(x,Interval) :
        return _inclusion_mul_p(x, 1/y)
    elif isinstance(y,Interval) :
        return _inclusion_mul_p(x, _inclusion_reciprocal_p(y))
    else :
        return x/y
inclusion_registry[lax.div_p] = _inclusion_div_p
Interval.__truediv__ = _inclusion_div_p

def _inclusion_reciprocal_p (x: Interval) -> Interval :
    if not isinstance (x, Interval) :
        return 1/x
    c = jnp.logical_or(jnp.logical_and(x.lower > 0, x.upper > 0),
                       jnp.logical_and(x.lower < 0, x.upper < 0))
    return Interval(jnp.where(c, (1./x.upper), -jnp.inf), jnp.where(c, (1./x.lower), jnp.inf))

def _inclusion_integer_pow_p (x:Interval, y: int) -> Interval :
    if not isinstance (x, Interval) :
        return x**y
    def _inclusion_integer_pow_impl (x: Interval, y:int) -> Interval :
        l_pow = lax.integer_pow(x.lower, y)
        u_pow = lax.integer_pow(x.upper, y)
    
        def even () :
            contains_zero = jnp.logical_and(
                jnp.less_equal(x.lower, 0), jnp.greater_equal(x.upper, 0))
            lower = jnp.where(contains_zero, jnp.zeros_like(x.lower),
                                jnp.minimum(l_pow, u_pow))
            upper = jnp.maximum(l_pow, u_pow)
            return (lower, upper)
        odd = lambda : (l_pow, u_pow)

        return lax.cond(jnp.all(y % 2), odd, even)

    def _pos_pow () :
        return _inclusion_integer_pow_impl(x, y)
    def _neg_pow () :
        return _inclusion_integer_pow_impl(_inclusion_reciprocal_p(x), -y)

    ol, ou = lax.cond(jnp.all(y < 0), _neg_pow, _pos_pow)
    return Interval(ol, ou)
inclusion_registry[lax.integer_pow_p] = _inclusion_integer_pow_p
Interval.__pow__ = _inclusion_integer_pow_p

# """
# The following two functions were taken and modified from jax_verify.
# _move_axes, and _inclusion_dot_general_p
# """

# def _move_axes(
#     bound: Interval, cdims: Tuple[int, ...], bdims: Tuple[int, ...],
#     orig_axis: int, new_axis: int,
# ) -> Tuple[Interval, Tuple[int, ...], Tuple[int, ...]]:
#   bound = interval(bound)
#   def new_axis_fn(old_axis):
#     if old_axis == orig_axis:
#       # This is the axis being moved. Return its new position.
#       return new_axis
#     elif (old_axis < orig_axis) and (old_axis >= new_axis):
#       # The original axis being moved was after, but it has now moved to before
#       # (or at this space). This means that this axis gets shifted back
#       return old_axis + 1
#     elif (old_axis > orig_axis) and (old_axis <= new_axis):
#       # The original axis being moved was before this one, but it has now moved
#       # to after. This means that this axis gets shifted forward.
#       return old_axis - 1
#     else:
#       # Nothing should be changing.
#       return old_axis

#   mapping = {old_axis: new_axis_fn(old_axis)
#              for old_axis in range(len(bound.lower.shape))}
#   permutation = sorted(range(len(bound.lower.shape)), key=lambda x: mapping[x])
#   new_bound = Interval(jax.lax.transpose(bound.lower, permutation),
#                        jax.lax.transpose(bound.upper, permutation))
#   new_cdims = tuple(new_axis_fn(old_axis) for old_axis in cdims)
#   new_bdims = tuple(new_axis_fn(old_axis) for old_axis in bdims)
#   return new_bound, new_cdims, new_bdims
# def _inclusion_dot_general_p(lhs: Interval, rhs: Interval, **kwargs) -> Interval:
#   (lhs_cdims, rhs_cdims), (lhs_bdims, rhs_bdims) = kwargs['dimension_numbers']

#   # Move the contracting dimensions to the front.
#   for cdim_index in range(len(lhs_cdims)):
#     lhs, lhs_cdims, lhs_bdims = _move_axes(lhs, lhs_cdims, lhs_bdims,
#                                            lhs_cdims[cdim_index], cdim_index)
#     rhs, rhs_cdims, rhs_bdims = _move_axes(rhs, rhs_cdims, rhs_bdims,
#                                            rhs_cdims[cdim_index], cdim_index)

#   # Because we're going to scan over the contracting dimensions, the
#   # batch dimensions are appearing len(cdims) earlier.
#   new_lhs_bdims = tuple(bdim - len(lhs_cdims) for bdim in lhs_bdims)
#   new_rhs_bdims = tuple(bdim - len(rhs_cdims) for bdim in rhs_bdims)

#   merge_cdims = lambda x: x.reshape((-1,) + x.shape[len(lhs_cdims):])
#   operands = ((merge_cdims(lhs.lower), merge_cdims(lhs.upper)),
#               (merge_cdims(rhs.lower), merge_cdims(rhs.upper)))
#   batch_shape = tuple(lhs.lower.shape[axis] for axis in lhs_bdims)
#   lhs_contr_shape = tuple(dim for axis, dim in enumerate(lhs.lower.shape)
#                           if axis not in lhs_cdims + lhs_bdims)
#   rhs_contr_shape = tuple(dim for axis, dim in enumerate(rhs.lower.shape)
#                           if axis not in rhs_cdims + rhs_bdims)
#   out_shape = batch_shape + lhs_contr_shape + rhs_contr_shape
#   init_carry = (jnp.zeros(out_shape), jnp.zeros(out_shape))

#   new_dim_numbers = (((), ()), (new_lhs_bdims, new_rhs_bdims))
#   unreduced_dotgeneral = partial(jax.lax.dot_general,
#                                            dimension_numbers=new_dim_numbers)

#   def scan_fun(carry: Tuple[ArrayLike, ArrayLike],
#                inp: Tuple[Tuple[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]]
#                ) -> Tuple[Tuple[ArrayLike, ArrayLike], None]:
#     """Accumulates the minimum and maximum as inp traverse the first dimension.
    
#     (The first dimension is where we have merged all contracting dimensions.)

#     Parameters
#     ----------
#     carry :
#         Current running sum of the lower bound and upper bound
#     carry: Tuple[ArrayLike :
        
#     ArrayLike] :
        
#     inp: Tuple[Tuple[ArrayLike :
        
#     Tuple[ArrayLike :
        
#     ArrayLike]] :
        

#     Returns
#     -------
#     updated_carry
#         New version of the running sum including these elements.
#     updated_carry
#         New version of the running sum including these elements.
#         None

#     """

#     (lhs_low, lhs_up), (rhs_low, rhs_up) = inp
#     carry_min, carry_max = carry
#     opt_1 = unreduced_dotgeneral(lhs_low, rhs_low)
#     opt_2 = unreduced_dotgeneral(lhs_low, rhs_up)
#     opt_3 = unreduced_dotgeneral(lhs_up, rhs_low)
#     opt_4 = unreduced_dotgeneral(lhs_up, rhs_up)
#     elt_min = jnp.minimum(jnp.minimum(jnp.minimum(opt_1, opt_2), opt_3), opt_4)
#     elt_max = jnp.maximum(jnp.maximum(jnp.maximum(opt_1, opt_2), opt_3), opt_4)
#     return (carry_min + elt_min, carry_max + elt_max), None

#   (lower, upper), _ = jax.lax.scan(scan_fun, init_carry, operands)
#   return Interval(lower, upper)

# def _inclusion_dot_p (A: Interval, B: Interval, **kwargs) -> Interval :

    # lAp = jnp.maximum(A.lower, 0)
    # lAn = jnp.minimum(A.lower, 0)
    # uAp = jnp.maximum(A.upper, 0)
    # uAn = jnp.minimum(A.upper, 0)
    # lBp = jnp.maximum(B.lower, 0)
    # lBn = jnp.minimum(B.lower, 0)
    # uBp = jnp.maximum(B.upper, 0)
    # uBn = jnp.minimum(B.upper, 0)
    # dg = partial(lax.dot_general, **kwargs)

    # lplp = dg(lAp, lBp)
    # lpup = dg(lAp, uBp)
    # uplp = dg(uAp, lBp)
    # upup = dg(uAp, uBp)
    # lnln = dg(lAn, lBn)
    # lnun = dg(lAn, uBn)
    # unln = dg(uAn, lBn)
    # unun = dg(uAn, uBn)
    # lpln = dg(lAp, lBn)
    # lpun = dg(lAp, uBn)
    # upln = dg(uAp, lBn)
    # upun = dg(uAp, uBn)
    # lnlp = dg(lAn, lBp)
    # unlp = dg(uAn, lBp)
    # unup = dg(uAn, uBp)

    # # I can use jnp.where instead....


    # # lA > 0, uA > 0, lB > 0, uB > 0
    # lplp, upup

    # # lA > 0, uA > 0, lB < 0, uB < 0
    # upln, lpun

    # # lA > 0, uA > 0, lB < 0, uB > 0

    # jnp.minimum(dg(lAn, uBp), dg(uAp, lBn))


    # return Interval(
    # )

def _inclusion_dot_general_p (A: Interval, B: Interval, **kwargs) -> Interval :
    # Current implementation only works for 2D matrix multiplication.
    # TODO: Generalize to work for any call to dot_general.

    if A.ndim > 2 or B.ndim > 2 :
        raise NotImplementedError("dot_general inclusion function currently only supported for vectors and matrices.")

    retdim = 2

    if A.ndim == 1 :
        A = A.reshape(1, -1)
        retdim = retdim - 1
    if B.ndim == 1 :
        B = B.reshape(-1, 1)
        retdim = retdim - 1    

    # Extract the contraction and batch dimensions
    # ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) = kwargs['dimension_numbers']

    # Move the batch dimensions to the front and the contraction dimensions to the back
    # A = Interval(jnp.moveaxis(A.lower, lhs_contract, 0), jnp.moveaxis(A.upper, lhs_contract, 0))
    # B = Interval(jnp.moveaxis(B.lower, rhs_contract, 0), jnp.moveaxis(B.upper, rhs_contract, 0))

    def _mul (a, b) :
        _1 = a.lower*b.lower
        _2 = a.lower*b.upper
        _3 = a.upper*b.lower
        _4 = a.upper*b.upper
        return Interval(jnp.minimum(jnp.minimum(_1,_2),jnp.minimum(_3,_4)),
                        jnp.maximum(jnp.maximum(_1,_2),jnp.maximum(_3,_4)))

    def _dot (a, b) :
        _mulres = vmap(_mul)(a, b)
        return Interval(jnp.sum(_mulres.lower, axis=0), jnp.sum(_mulres.upper, axis=0))

    def _arow (a) :
        return vmap(_dot, (None, -1))(a, B)

    res = vmap(_arow)(A)
   
    if retdim == 1 : res = res.reshape(-1)
    if retdim == 0 : res = res.reshape(())
    return res

inclusion_registry[lax.dot_general_p] = _inclusion_dot_general_p



def _inclusion_sin_p (x:Interval) -> Interval :
    if not isinstance (x, Interval) :
        return lax.sin(x)
    def _sin_if (l:jnp.float32, u:jnp.float32) :
        def case_lpi (l, u) :
            cl = jnp.cos(l); cu = jnp.cos(u)
            branch = jnp.array(cl >= 0, "int32") + 2*jnp.array(cu >= 0, "int32")
            case3 = lambda : (jnp.sin(l), jnp.sin(u)) # cl >= 0, cu >= 0
            case0 = lambda : (jnp.sin(u), jnp.sin(l)) # cl <= 0, cu <= 0
            case1 = lambda : (jnp.minimum(jnp.sin(l), jnp.sin(u)),  1.0) # cl >= 0, cu <= 0
            case2 = lambda : (-1.0, jnp.maximum(jnp.sin(l), jnp.sin(u))) # cl <= 0, cu >= 0
            return lax.switch(branch, [case0, case1, case2, case3])
        def case_pi2pi (l, u) :
            cl = jnp.cos(l); cu = jnp.cos(u)
            branch = jnp.array(cl >= 0, "int32") + 2*jnp.array(cu >= 0, "int32")
            case3 = lambda : (-1.0, 1.0) # cl >= 0, cu >= 0
            case0 = lambda : (-1.0, 1.0) # cl <= 0, cu <= 0
            case1 = lambda : (jnp.minimum(jnp.sin(l), jnp.sin(u)),  1.0) # cl >= 0, cu <= 0
            case2 = lambda : (-1.0, jnp.maximum(jnp.sin(l), jnp.sin(u))) # cl <= 0, cu >= 0
            return lax.switch(branch, [case0, case1, case2, case3])
        def case_else (l, u) :
            return -1.0, 1.0
        diff = u - l
        c = jnp.array(diff <= jnp.pi, "int32") + jnp.array(diff <= 2*jnp.pi, "int32")
        ol, ou = lax.switch(c, [case_else, case_pi2pi, case_lpi], l, u)
        return ol, ou
    _sin_if_vmap = jax.vmap(_sin_if,(0,0))
    _x, x_ = _sin_if_vmap(x.lower.reshape(-1), x.upper.reshape(-1))
    return Interval(_x.reshape(x.shape), x_.reshape(x.shape))
inclusion_registry[lax.sin_p] = _inclusion_sin_p

def _inclusion_cos_p (x:Interval) -> Interval :
    return _inclusion_sin_p(Interval(x.lower + jnp.pi/2, x.upper + jnp.pi/2))
inclusion_registry[lax.cos_p] = _inclusion_cos_p

def _inclusion_tan_p (x:Interval) -> Interval :
    l = x.lower; u = x.upper
    div = jnp.floor((u + jnp.pi/2) / (jnp.pi)).astype(int)
    l -= div*jnp.pi; u -= div*jnp.pi
    ol = jnp.where((l < -jnp.pi/2), -jnp.inf, jnp.tan(l))
    ou = jnp.where((l < -jnp.pi/2),  jnp.inf, jnp.tan(u))
    return Interval(ol, ou)
inclusion_registry[lax.tan_p] = _inclusion_tan_p

def _inclusion_atan_p (x:Interval) -> Interval :
    return Interval(jnp.arctan(x.lower), jnp.arctan(x.upper))
inclusion_registry[lax.atan_p] = _inclusion_atan_p

def _inclusion_sqrt_p (x:Interval) -> Interval :
    ol = jnp.where((x.lower < 0), -jnp.inf, jnp.sqrt(x.lower))
    ou = jnp.where((x.lower < 0), jnp.inf, jnp.sqrt(x.upper))
    return Interval (ol, ou)
inclusion_registry[lax.sqrt_p] = _inclusion_sqrt_p

def _inclusion_pow_p(x:Interval, y: int) -> Interval :
    # if isinstance (y, Interval) :
    #     # if y.lower == y.upper :
    #     if True :
    #         y = y.upper
    #     else :
    #         raise Exception('y must be a constant')
    def _inclusion_pow_impl (x:Interval, y:int) :
        l_pow = lax.pow(x.lower, y)
        u_pow = lax.pow(x.upper, y)
        cond = jnp.logical_and(x.lower >= 0, x.upper >= 0)
        ol = jnp.where(cond, l_pow, -jnp.inf)
        ou = jnp.where(cond, u_pow, jnp.inf)
        return (ol, ou)

    def _pos_pow () :
        return _inclusion_pow_impl(x, y)
    def _neg_pow () :
        return _inclusion_pow_impl(_inclusion_reciprocal_p(x), -y)

    ol, ou = lax.cond(jnp.all(y < 0), _neg_pow, _pos_pow)
    return Interval(ol, ou)
inclusion_registry[lax.pow_p] = _inclusion_pow_p

def _inclusion_tanh_p (x:Interval) -> Interval :
    return Interval(jnp.tanh(x.lower), jnp.tanh(x.upper))
inclusion_registry[lax.tanh_p] = _inclusion_tanh_p


# def natif_jaxpr (jaxpr, consts, *args) :
#     env = {}
#     def read (var) :
#         # Literals are values baked into the Jaxpr
#         if type(var) is core.Literal :
#             return var.val
#         return env[var]
#     def write (var, val) :
#         env[var] = val
    
#     # Bind args and consts to environment
#     # print(write)
#     # print(jaxpr.invars)
#     safe_map(write, jaxpr.invars, args)
#     safe_map(write, jaxpr.constvars, consts)

#     # Loop through equations (forward)
#     for eqn in jaxpr.eqns :
#         # Read inputs to equation from environment
#         invals = safe_map(read, eqn.invars)
#         # Check if primitive has an inclusion function
#         if eqn.primitive not in inclusion_registry :
#             raise NotImplementedError(
#                 f"{eqn.primitive} does not have a registered inclusion function")
#         outvals = inclusion_registry[eqn.primitive](*invals, **eqn.params)
#         # Primitives may return multiple outputs or not
#         if not eqn.primitive.multiple_results :
#             outvals = [outvals]
#         # Write the results of the primitive into the environment
#         safe_map(write, eqn.outvars, outvals)
#     return safe_map(read, jaxpr.outvars)

def natif_jaxpr (jaxpr: Jaxpr, consts, *args, propagate_source_info=True) -> list[Any]:
    def read(v: Atom) -> Any:
        return v.val if isinstance(v, Literal) else env[v]

    def write(v: Var, val: Any) -> None:
        if config.enable_checks.value and not config.dynamic_shapes.value:
            assert typecheck(v.aval, val), (v.aval, val)
        env[v] = val

    env: dict[Var, Any] = {}
    safe_map(write, jaxpr.constvars, consts)
    # print(jaxpr.invars)
    # print(args)
    safe_map(write, jaxpr.invars, args)
    lu = last_used(jaxpr)
    for eqn in jaxpr.eqns:
        subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
        name_stack = source_info_util.current_name_stack() + eqn.source_info.name_stack
        traceback = eqn.source_info.traceback if propagate_source_info else None
        with source_info_util.user_context(traceback, name_stack=name_stack):
            # ans = eqn.primitive.bind(*subfuns, *map(read, eqn.invars), **bind_params)
            ans = inclusion_registry[eqn.primitive](*subfuns, *safe_map(read, eqn.invars), **bind_params)
        if eqn.primitive.multiple_results:
            safe_map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)
        clean_up_dead_vars(eqn, env, lu)
    return safe_map(read, jaxpr.outvars)


def natif (f:Callable[..., jax.Array]) -> Callable[..., Interval] :
    """Creates a Natural Inclusion Function of f using natif.
    
    All positional arguments are assumed to be replaced with interval arguments for the inclusion function.

    Parameters
    ----------
    f : Callable[..., jax.Array]
        Function to construct Natural Inclusion Function from

    Returns
    -------
    Callable[..., Interval]
        Natural Inclusion Function of f

    """
    @wraps(f)
    def wrapped (*args, **kwargs) :
        f"""Natural inclusion function.

        Parameters
        ----------
        *args :
            
        **kwargs :

        Returns
        -------
        _type_
            _description_
        """
        getlower = lambda x : x.lower if isinstance(x, Interval) else x
        isinterval = lambda x : isinstance(x, Interval)
        buildargs = jax.tree_util.tree_map(getlower, args, is_leaf=isinterval)
        buildkwargs = jax.tree_util.tree_map(getlower, kwargs, is_leaf=isinterval)
        closed_jaxpr = jax.make_jaxpr(f)(*buildargs, **buildkwargs)
        out = natif_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        if len(out) == 1 :
            return out[0]
        return out

    return wrapped

Interval.__matmul__ = jit(natif(jnp.matmul))

def jacM (f:Callable[..., jax.Array]) -> Callable[..., Interval] :
    """Creates the M matrices for the Jacobian-based inclusion function.
    
    All positional arguments are assumed to be replaced with interval arguments for the inclusion function.

    Parameters
    ----------
    f : Callable[..., jax.Array]
        Function to construct Jacobian Inclusion Function from
        
    Returns
    -------
    Callable[..., Interval]
        Jacobian-Based Inclusion Function of f

    """

    @jit
    @api_boundary
    def F (*args, centers:jax.Array|Sequence[jax.Array]|None = None, **kwargs) -> Interval :
        """Jacobian-based Inclusion Function of f.
        
        All positional arguments from f should be replaced with interval arguments for the inclusion function.
        
        Additional Args:
            centers (jax.Array | Sequence[jax.Array] | None, optional): _description_. Defaults to None.

        Parameters
        ----------
        *args :
            
        centers:jax.Array|Sequence[jax.Array]|None :
             (Default value = None)
        **kwargs :
            

        Returns
        -------
        Interval
            Interval output from the Jacobian-based Inclusion Function

        """
        args = [interval(arg).atleast_1d() for arg in args]
        if centers is None :
            centers = [tuple([(x.lower + x.upper)/2 for x in args])]
        elif isinstance(centers, jax.Array) :
            centers = [centers]
        elif not isinstance(centers, Sequence) :
            raise Exception('Must pass jax.Array (one center), Sequence[jax.Array], or None (auto-centered) for the centers argument')

        return [natif(jax.jacfwd(f, i))(*args) for i in range(len(args))]
        # return [interval(jax.jacfwd(f, i)(*centers[0])) for i in range(len(args))]
    return F

def jacif (f:Callable[..., jax.Array]) -> Callable[..., Interval] :
    """Creates a Jacobian Inclusion Function of f using natif.
    
    All positional arguments are assumed to be replaced with interval arguments for the inclusion function.

    Parameters
    ----------
    f : Callable[..., jax.Array]
        Function to construct Jacobian Inclusion Function from
        
    Returns
    -------
    Callable[..., Interval]
        Jacobian-Based Inclusion Function of f

    """

    @jit
    @api_boundary
    def F (*args, centers:jax.Array|Sequence[jax.Array]|None = None, **kwargs) -> Interval :
        """Jacobian-based Inclusion Function of f.
        
        All positional arguments from f should be replaced with interval arguments for the inclusion function.
        
        Additional Args:
            centers (jax.Array | Sequence[jax.Array] | None, optional): _description_. Defaults to None.

        Parameters
        ----------
        *args :
            
        centers:jax.Array|Sequence[jax.Array]|None :
             (Default value = None)
        **kwargs :
            

        Returns
        -------
        Interval
            Interval output from the Jacobian-based Inclusion Function

        """
        if centers is None :
            centers = [tuple([(x.lower + x.upper)/2 for x in args])]
        elif isinstance(centers, jax.Array) :
            centers = [centers]
        elif not isinstance(centers, Sequence) :
            raise Exception('Must pass jax.Array (one center), Sequence[jax.Array], or None (auto-centered) for the centers argument')

        retl, retu = [], []
        df = [natif(jax.jacfwd(f, i))(*args) for i in range(len(args))]
        for center in centers :
            if len(center) != len(args) :
                raise Exception(f'Not enough points {len(center)=} != {len(args)=} to center the Jacobian-based inclusion function around')
            f0 = f(*center)
            sum = interval(f0)
            for i in range(len(args)) :
                # sum = natif(jnp.add)(sum, natif(jnp.matmul)(df[i], (interval(args[i].lower - center[i], args[i].upper - center[i]))))
                sum = sum + df[i] @ (interval(args[i].lower - center[i], args[i].upper - center[i]))
            retl.append(sum.lower)
            retu.append(sum.upper)
        retl, retu = jnp.array(retl), jnp.array(retu)
        return interval(jnp.max(retl,axis=0), jnp.min(retu,axis=0))
    return F

class Permutation (tuple) :
    """A tuple of :math:`n` numbers :math:`(o_i)_{i=1}^n` such that each :math:`0\\leq o_i \\leq n-1` and each :math:`o_i` is unique."""
    def __new__(cls, __iterable: Iterable = ()) -> 'Permutation':
        if sum([2**x for x in __iterable]) != (2**len(__iterable) - 1) :
            raise Exception(f'The permutation doesnt have every i from 0 to n-1: {__iterable}')
        return tuple.__new__(Permutation, (__iterable))
    def __str__(self) -> str:
        return 'Permutation' + super().__str__()
    def sub (self, i:int) -> 'Permutation' :
        """Returns the sub-permutation of the first i elements."""
        return self[:i+1]

    @property
    def arr (self) -> jax.Array :
        """Returns the Permutation in a jax.Array."""
        return jnp.asarray(self)

    @property
    def mat (self) -> jax.Array :
        """Returns the permutation matrix of the Permutation."""
        n = len(self)
        return jnp.array([[1 if j == self[i] else 0 for j in range(n)] for i in range(n)])

    @property
    def mtx (self) -> jax.Array :
        """Returns the replacement matrix of the Permutation."""
        n = len(self)
        return jnp.array([[1 if j in self.sub(i) else 0 for j in range(n)] for i in range(n)])

def standard_permutation (n:int) -> Tuple[Permutation] :
    """Returns the standard n-permutation :math:`(0,\\dots,n-1)`"""
    return (Permutation(range(n)),)

def two_permutations (n:int) -> Tuple[Permutation] :
    """Returns the two standard n-permutations :math:`(0,\\dots,n-1)` and :math:`(n-1,\\dots,0)`"""
    return (Permutation(range(n)), Permutation(tuple(reversed(range(n)))))

def all_permutations (n:int) -> Tuple[Permutation] :
    """Returns all n-permutations."""
    return tuple(Permutation(x) for x in perms(range(n)))

class Corner (tuple) :
    """A tuple of :math:`n` elements in :math:`\\{0,1\\}` representing the corners of an :math:`n`-dimensional hypercube. 0 is the lower bound, 1 is the upper bound"""
    def __new__(cls, __iterable: Iterable = ()) -> 'Corner':
        for x in __iterable :
            if x not in (0,1) :
                raise Exception(f'The corner elements need to be in 0,1: {__iterable}')
        return tuple.__new__(Corner, (__iterable))
    def __str__(self) -> str:
        return 'Corner' + super().__str__()

def bot_corner (n:int) -> Tuple[Corner] :
    """Returns the bottom corner of the n-dimensional hypercube."""
    return (Corner((0,)*n),)

def top_corner (n:int) -> Tuple[Corner] :
    """Returns the top corner of the n-dimensional hypercube."""
    return (Corner((1,)*n),)

def two_corners (n:int) -> Tuple[Corner] :
    """Returns the bottom and top corners of the n-dimensional hypercube."""
    return (Corner((0,)*n), Corner((1,)*n))

def all_corners (n:int) -> Tuple[Corner] :
    """Returns all corners of the n-dimensional hypercube."""
    return tuple(Corner(x) for x in product((0,1), repeat=n))


# def mjacM (f:Callable[..., jax.Array]) -> Callable :
#     """Creates the M matrices for the Mixed Jacobian-based inclusion function.
    
#     All positional arguments are assumed to be replaced with interval arguments for the inclusion function.

#     Parameters
#     ----------
#     f : Callable[..., jax.Array]
#         Function to construct Mixed Jacobian Inclusion Function from

#     Returns
#     -------
#     Callable[..., Interval]
#         Mixed Jacobian-Based Inclusion Function of f

#     """

#     # @partial(jit,static_argnames=['permutations', 'corners'])
#     @api_boundary
#     def F (*args, permutations:Tuple[Permutation]|None = None, centers:jax.Array|Sequence[jax.Array]|None = None, 
#            corners:Tuple[Corner]|None = None,**kwargs) -> Interval :
#         """_summary_

#         Parameters
#         ----------
#         permutations : Tuple[Permutation] | None, optional
#             _description_, by default None
#         centers : jax.Array | Sequence[jax.Array] | None, optional
#             _description_, by default None
#         corners : Tuple[Corner] | None, optional
#             _description_, by default None

#         Returns
#         -------
#         Interval
#             _description_

#         Raises
#         ------
#         Exception
#             _description_
#         Exception
#             _description_
#         Exception
#             _description_
#         Exception
#             _description_
#         Exception
#             _description_
#         """
#         args = [interval(arg).atleast_1d() for arg in args]
#         leninputsfull = tuple([len(x) for x in args])
#         leninputs = sum(leninputsfull)

#         if permutations is None :
#             permutations = standard_permutation(leninputs)
#         elif isinstance(permutations, Permutation) :
#             permutations = [permutations]
#         elif not isinstance(permutations, Tuple) :
#             raise Exception('Must pass jax.Array (one permutation), Sequence[jax.Array], or None (auto standard permutation) for the permutations argument')

#         cumsum = tuple(accumulate(leninputsfull))
#         permutations_pairs = []

#         # Split permutation into individual inputs and indices.
#         for permutation in permutations :
#             if len(permutation) != leninputs :
#                 raise Exception(f'The permutation is not the same length as the sum of the lengths of the inputs: {len(permutation)} != {leninputs}')
#             pairs = []
#             for o in permutation :
#                 a = 0
#                 while cumsum[a] - 1 < o :
#                     a += 1
#                 pairs.append((a,(o - cumsum[a-1] if a > 0 else o)))
#             permutations_pairs.append(tuple(pairs))

#         # Mixed Centered
#         if centers is None :
#             if corners is None :
#                 # Auto-centered
#                 centers = [tuple([(x.lower + x.upper)/2 for x in args])]
#             else :
#                 centers = []
#         elif isinstance(centers, jax.Array) :
#             centers = [centers]
#         elif not isinstance(centers, Sequence) :
#             raise Exception('Must pass jax.Array (one center), Sequence[jax.Array], or None (auto-centered) for the centers argument')

#         if corners is not None :
#             if not isinstance(corners, Tuple) :
#                 raise Exception('Must pass Tuple[Corner] or None for the corners argument')
#             centers.extend([tuple([(x.lower if c[i] == 0 else x.upper) for i,x in enumerate(args)]) for c in corners])

#         # centers.extend([tuple([(x.lower if c[i] == 0 else x.upper) for i,x in enumerate(args)]) for c in corners])

#         # multiple permutations/centers
#         ret = []

#         df_func = [natif(jax.jacfwd(f, i)) for i in range(len(args))]
#         # centers is an array of centers to check
#         for center in centers :
#             if len(center) != len(args) :
#                 raise Exception(f'Not enough points {len(center)=} != {len(args)=} to center the Jacobian-based inclusion function around')
#             f0 = f(*center)
#             for pairs in permutations_pairs :
#                 val_lower = [jnp.empty((len(f0), leninput)) for leninput in leninputsfull]
#                 val_upper = [jnp.empty((len(f0), leninput)) for leninput in leninputsfull]
#                 argr = [interval(c).atleast_1d() for c in center]

#                 for argi, subi in pairs :
#                     # Replacement operation
#                     l = argr[argi].lower.at[subi].set(args[argi].lower[subi])
#                     u = argr[argi].upper.at[subi].set(args[argi].upper[subi])
#                     argr[argi] = interval(l, u)
#                     # Mixed Jacobian matrix subi-th column. TODO: Make more efficient?
#                     Mi = interval(df_func[argi](*argr, **kwargs)[:,subi])
#                     val_lower[argi] = val_lower[argi].at[:,subi].set(Mi.lower)
#                     val_upper[argi] = val_upper[argi].at[:,subi].set(Mi.upper)

#                 ret.append([interval(val_lower[argi],val_upper[argi]) for argi in range(len(args))])
        
#         return ret
#     return F


def mjacM (f:Callable[..., jax.Array]) -> Callable :
    """Creates the M matrices for the Mixed Jacobian-based inclusion function.
    
    All positional arguments are assumed to be replaced with interval arguments for the inclusion function.

    Parameters
    ----------
    f : Callable[..., jax.Array]
        Function to construct Mixed Jacobian Inclusion Function from

    Returns
    -------
    Callable[..., Interval]
        Mixed Jacobian-Based Inclusion Function of f

    """

    # @partial(jit,static_argnames=['permutations', 'corners'])
    @api_boundary
    def F (*args, permutations:Tuple[Permutation]|None = None, centers:jax.Array|Sequence[jax.Array]|None = None, 
           corners:Tuple[Corner]|None = None,**kwargs) -> Interval :
        """_summary_

        Parameters
        ----------
        permutations : Tuple[Permutation] | None, optional
            _description_, by default None
        centers : jax.Array | Sequence[jax.Array] | None, optional
            _description_, by default None
        corners : Tuple[Corner] | None, optional
            _description_, by default None

        Returns
        -------
        Interval
            _description_

        Raises
        ------
        Exception
            _description_
        Exception
            _description_
        Exception
            _description_
        Exception
            _description_
        Exception
            _description_
        """
        args = [interval(arg).atleast_1d() for arg in args]
        leninputsfull = tuple([len(x) for x in args])
        leninputs = sum(leninputsfull)

        if permutations is None :
            permutations = standard_permutation(leninputs)
        elif isinstance(permutations, Permutation) :
            permutations = [permutations]
        elif not isinstance(permutations, Tuple) :
            raise Exception('Must pass jax.Array (one permutation), Sequence[jax.Array], or None (auto standard permutation) for the permutations argument')

        cumsum = tuple(accumulate(leninputsfull))

        # Mixed Centered
        if centers is None :
            if corners is None :
                # Auto-centered
                centers = [tuple([(x.lower + x.upper)/2 for x in args])]
            else :
                centers = []
        elif isinstance(centers, jax.Array) :
            centers = [centers]
        elif not isinstance(centers, Sequence) :
            raise Exception('Must pass jax.Array (one center), Sequence[jax.Array], or None (auto-centered) for the centers argument')

        if corners is not None :
            if not isinstance(corners, Tuple) :
                raise Exception('Must pass Tuple[Corner] or None for the corners argument')
            centers.extend([tuple([(x.lower if c[i] == 0 else x.upper) for i,x in enumerate(args)]) for c in corners])

        # multiple permutations/centers
        ret = []

        def arg2z (*args) :
            return jnp.concatenate(args)

        def z2arg (z, **kwargs) :
            return jnp.split(z, cumsum[:-1], axis=-1)
        
        df_func = [natif(jax.jacfwd(partial(f, **kwargs), i)) for i in range(len(args))]
        _z = arg2z(*[arg.lower for arg in args])
        z_ = arg2z(*[arg.upper for arg in args])

        for center in centers :
            if len(center) != len(args) :
                raise Exception(f'Not enough points {len(center)=} != {len(args)=} to center the Jacobian-based inclusion function around')
            # f0 = f(*center)
            zc = arg2z(*center)
            for sig in permutations :
                Z = interval(
                    jnp.where(sig.mtx, jnp.tile(_z, (len(sig),1)), jnp.tile(zc, (len(sig),1))),
                    jnp.where(sig.mtx, jnp.tile(z_, (len(sig),1)), jnp.tile(zc, (len(sig),1)))
                )
                _cumsum = (0,) + cumsum
                retc = []
                npsig = np.asarray(sig)

                # # Using jax.lax.scan to build columns
                # for i in range(len(args)) :
                #     idx = np.logical_and(npsig >= _cumsum[i], npsig < _cumsum[i+1])
                #     def to_scan (_, arg) :
                #         sigj, z = arg
                #         return None, df_func[i](*natif(z2arg)(z))[:,sigj]
                #     _, Mi = jax.lax.scan(to_scan, None, (npsig[idx], Z[idx])) #
                #     # print(Mi.shape)
                #     # print(npsig[idx])
                #     retc.append(Mi[npsig[idx]-_cumsum[i]].T)

                # Using vmap to build columns
                for i in range(len(args)) :
                    idx = np.logical_and(npsig >= _cumsum[i], npsig < _cumsum[i+1])
                    Mi = jax.vmap(df_func[i])(*natif(z2arg)(Z[idx]))
                    # sig.arr[idx]-_cumsum[i] rearranges/extracts the columns of Mi
                    retc.append(Mi[np.arange(leninputsfull[i]),:,npsig[idx]-_cumsum[i]].T)
 
                ret.append(retc)
        return ret
    return F

def mjacif (f:Callable[..., jax.Array]) -> Callable[..., Interval] :
    """Creates a Mixed Jacobian Inclusion Function of f using natif.
    
    All positional arguments are assumed to be replaced with interval arguments for the inclusion function.

    Parameters
    ----------
    f : Callable[..., jax.Array]
        Function to construct Mixed Jacobian Inclusion Function from

    Returns
    -------
    Callable[..., Interval]
        Mixed Jacobian-Based Inclusion Function of f

    """

    # @wraps(f)
    @partial(jit,static_argnames=['permutations','corners'])
    @api_boundary
    def F (*args, permutations:Tuple[Permutation]|None=None, centers:jax.Array|Sequence[jax.Array]|None = None,
           corners:Tuple[Corner]|None = None,**kwargs) -> Interval :
        """_summary_

        Parameters
        ----------
        permutations : Tuple[Permutation] | None, optional
            _description_, by default None
        centers : jax.Array | Sequence[jax.Array] | None, optional
            _description_, by default None
        corners : Tuple[Corner] | None, optional
            _description_, by default None

        Returns
        -------
        Interval
            _description_

        Raises
        ------
        Exception
            _description_
        Exception
            _description_
        Exception
            _description_
        Exception
            _description_
        Exception
            _description_
        """

        leninputsfull = tuple([len(x) for x in args])
        leninputs = sum(leninputsfull)

        if permutations is None :
            permutations = standard_permutation(leninputs)
        elif isinstance(permutations, Permutation) :
            permutations = [permutations]
        elif not isinstance(permutations, Tuple) :
            raise Exception('Must pass jax.Array (one permutation), Sequence[jax.Array], or None (auto standard permutation) for the permutations argument')

        cumsum = tuple(accumulate(leninputsfull))
        permutations_pairs = []

        # Split each permutation into individual inputs and indices.
        # Permutation is of the length of taking each input and concatenating them.
        # The result in permutations_pairs is a list of tuples of 2-tuples (argi, subi)
        # argi is the argument index, subi is the subindex of that argument.
        for permutation in permutations :
            if len(permutation) != leninputs :
                raise Exception(f'The permutation is not the same length as the sum of the lengths of the inputs: {len(permutation)} != {leninputs}')
            pairs = []
            for o in permutation :
                a = 0
                while cumsum[a] - 1 < o :
                    a += 1
                pairs.append((a,(o - cumsum[a-1] if a > 0 else o)))
            permutations_pairs.append(tuple(pairs))

        # Mixed Centered
        if centers is None :
            if corners is None :
                # Auto-centered
                centers = [tuple([(x.lower + x.upper)/2 for x in args])]
            else :
                centers = []
        elif isinstance(centers, jax.Array) :
            centers = [centers]
        elif not isinstance(centers, Sequence) :
            raise Exception('Must pass jax.Array (one center), Sequence[jax.Array], or None (auto-centered) for the centers argument')

        if corners is not None :
            if not isinstance(corners, Tuple) :
                raise Exception('Must pass Tuple[Corner] or None for the corners argument')
            centers.extend([tuple([(x.lower if c[i] == 0 else x.upper) for i,x in enumerate(args)]) for c in corners])

 
        # multiple permutations/centers, need to take min/max for final inclusion function output.
        retl, retu = [], []

        # This is the \sfJ_x for each argument. Natural inclusion on the Jacobian tree.
        df_func = [natif(jax.jacfwd(f, i)) for i in range(len(args))]

        # centers is an array of centers to check
        for center in centers :
            if len(center) != len(args) :
                raise Exception(f'Not enough points {len(center)=} != {len(args)=} to center the Jacobian-based inclusion function around')
            f0 = f(*center)
            for pairs in permutations_pairs :
                # argr initialized at the center, will be slowly relaxed to the whole interval
                argr = [interval(c) for c in center]
                # val is the eventual output, M ([\ulx,\olx] - \overcirc{x}).
                # Will be built column-by-column of the matrix multiplication.
                val = interval(f0)

                for argi, subi in pairs :
                    # Perform the replacement operation using (argi, subi)
                    l = argr[argi].lower.at[subi].set(args[argi].lower[subi])
                    u = argr[argi].upper.at[subi].set(args[argi].upper[subi])
                    # Set the "running" interval
                    argr[argi] = interval(l, u)
                    # Mixed Jacobian matrix subi-th column. TODO: Make more efficient?
                    # Extracting a column here for the right shape for the multiplication
                    Mi = df_func[argi](*argr, **kwargs)[:,(subi,)]
                    # Mi ([\ulx,\olx]_i - \overcirc{x}_i)
                    val = natif(jnp.add)(val, natif(jnp.matmul)(Mi, (interval(args[argi].lower[subi] - center[argi][subi], args[argi].upper[subi] - center[argi][subi]).reshape(-1))))

                # (\sfJ_x, \overcirc{x}, \calO)-Mixed Jacobian-based added to the potential
                retl.append(val.lower)
                retu.append(val.upper)
        
        retl, retu = jnp.array(retl), jnp.array(retu)
        return interval(jnp.max(retl,axis=0), jnp.min(retu,axis=0))
    return F
