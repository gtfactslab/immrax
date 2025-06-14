from typing import List
import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
from typing import Tuple, Iterable
from jaxtyping import ArrayLike
import numpy as onp


@register_pytree_node_class
class Interval:
    """Interval: A class to represent an interval in :math:`\\mathbb{R}^n`.

    Use the helper functions :func:`interval`, :func:`icentpert`, :func:`i2centpert`, :func:`i2lu`, :func:`i2ut`, and :func:`ut2i` to create and manipulate intervals.

    Use the transforms :func:`natif`, :func:`jacif`, :func:`mjacif`, :func:`mjacM`, to create inclusion functions.

    Composable with typical jax transforms, such as :func:`jax.jit`, :func:`jax.grad`, and :func:`jax.vmap`.
    """

    lower: jax.Array
    upper: jax.Array

    def __init__(self, lower: jax.Array, upper: jax.Array) -> None:
        self.lower = lower
        self.upper = upper

    def tree_flatten(self):
        return ((self.lower, self.upper), "Interval")

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    @property
    def dtype(self) -> jnp.dtype:
        return self.lower.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.lower.shape

    @property
    def size(self) -> int:
        return self.lower.size

    @property
    def width(self) -> jax.Array:
        return self.upper - self.lower

    @property
    def center(self) -> jax.Array:
        return (self.lower + self.upper) / 2

    def __matmul__(self, _: "Interval") -> "Interval": ...
    def __truediv__(self, _: "Interval") -> "Interval": ...
    def __neg__(self) -> "Interval": ...

    def __len__(self) -> int:
        return len(self.lower)

    def reshape(self, *args, **kwargs):
        return interval(
            self.lower.reshape(*args, **kwargs), self.upper.reshape(*args, **kwargs)
        )

    def ravel(self) -> List["Interval"]:
        return [interval(l, u) for l, u in zip(self.lower.ravel(), self.upper.ravel())]

    def atleast_1d(self) -> "Interval":
        return interval(jnp.atleast_1d(self.lower), jnp.atleast_1d(self.upper))

    def atleast_2d(self) -> "Interval":
        return interval(jnp.atleast_2d(self.lower), jnp.atleast_2d(self.upper))

    def atleast_3d(self) -> "Interval":
        return interval(jnp.atleast_3d(self.lower), jnp.atleast_3d(self.upper))

    @property
    def ndim(self) -> int:
        return self.lower.ndim

    def transpose(self, *args) -> "Interval":
        return Interval(self.lower.transpose(*args), self.upper.transpose(*args))

    @property
    def T(self) -> "Interval":
        return self.transpose()

    def __and__(self, other: "Interval") -> "Interval":
        return interval(
            jnp.maximum(self.lower, other.lower), jnp.minimum(self.upper, other.upper)
        )

    def __or__(self, other: "Interval") -> "Interval":
        return interval(
            jnp.minimum(self.lower, other.lower), jnp.maximum(self.upper, other.upper)
        )

    def __str__(self) -> str:
        return (
            onp.array(
                [
                    [(l, u)]
                    for (l, u) in zip(self.lower.reshape(-1), self.upper.reshape(-1))
                ],
                dtype=onp.dtype([("f1", float), ("f2", float)]),
            )
            .reshape(self.shape + (1,))
            .__str__()
        )
        # return self.lower.__str__() + ' <= x <= ' + self.upper.__str__()

    def __repr__(self) -> str:
        # return onp.array([[(l,u)] for (l,u) in
        #                 zip(self.lower.reshape(-1),self.upper.reshape(-1))],
        #                 dtype=onp.dtype([('f1',float), ('f2', float)])).reshape(self.shape + (1,)).__str__()
        # dtype=np.dtype([('f1',float), ('f2', float)])).reshape(self.shape + (1,)).__repr__()
        return self.lower.__str__() + " <= x <= " + self.upper.__str__()

    def __getitem__(self, i: slice | ArrayLike) -> "Interval":
        return Interval(self.lower[i], self.upper[i])

    def __iter__(self):
        """Return an iterator over the interval elements."""
        # Use the actual length to create a proper iterator
        # This avoids the infinite loop issue by using explicit indexing
        length = int(len(self))
        return (self[i] for i in range(length))


# HELPER FUNCTIONS


def interval(lower: ArrayLike, upper: ArrayLike | None = None) -> Interval:
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
    if isinstance(lower, Interval) and upper is None:
        return lower
    if upper is None:
        return Interval(jnp.asarray(lower), jnp.asarray(lower))
    lower = jnp.asarray(lower)
    upper = jnp.asarray(upper)
    if lower.dtype != upper.dtype:
        raise Exception(
            f"lower and upper dtype should match, {lower.dtype} != {upper.dtype}"
        )
    if lower.shape != upper.shape:
        raise Exception(
            f"lower and upper shape should match, {lower.shape} != {upper.shape}"
        )
    return Interval(jnp.asarray(lower), jnp.asarray(upper))


def icopy(i: Interval) -> Interval:
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


def icentpert(cent: ArrayLike, pert: ArrayLike) -> Interval:
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


centpert2i = icentpert


def i2centpert(i: Interval) -> Tuple[jax.Array, jax.Array]:
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
    return (i.lower + i.upper) / 2, (i.upper - i.lower) / 2


def i2lu(i: Interval) -> Tuple[jax.Array, jax.Array]:
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


def lu2i(l: jax.Array, u: jax.Array) -> Interval:
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


def i2ut(i: Interval) -> jax.Array:
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


def ut2i(coordinate: jax.Array, n: int | None = None) -> Interval:
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
    if n is None:
        n = len(coordinate) // 2
    return interval(coordinate[:n], coordinate[n:])


def izeros(shape: Tuple[int], dtype: onp.dtype = jnp.float32) -> Interval:
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


def iconcatenate(intervals: Iterable[Interval], axis: int = 0) -> Interval:
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
    return interval(
        jnp.concatenate([i.lower for i in intervals], axis=axis),
        jnp.concatenate([i.upper for i in intervals], axis=axis),
    )
