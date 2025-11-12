import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
from jaxtyping import Integer, Float, Array, ArrayLike
from typing import Tuple, Iterable, Union, List, Callable, Literal, Mapping
from ..inclusion import Interval, interval

@register_pytree_node_class
class Parametope :
    r"""A Parametope is the set

    .. math::
        {x : g(\alpha, x - \mathring{x}) <= y}

    where :math:`g` is a nonlinearity parameterized by :math:`\alpha`, :math:`\mathring{x}` is the center, and :math:`y` is the offset.
        
    In subclasses, define the nonlinearity :math:`g` by implementing the `g` method, as well as the `from_parametope` classmethod to unpack flattened data.

    We avoid making this class abstract because we need Parametopes as a data container for JAX's internal pytree mechanism. 
    """

    ox: ArrayLike  # Center
    alpha: ArrayLike  # Parameters
    y: ArrayLike  # Offset

    def __init__ (self, ox:ArrayLike, alpha:ArrayLike, y:ArrayLike) :
        self.ox = jnp.asarray(ox)
        self.alpha = jnp.asarray(alpha)
        self.y = jnp.asarray(y)
    
    def g (self, x:Array) -> Array :
        r"""Evaluates the nonlinearity :math:`g(\alpha, x - \mathring{x})` at x

        Parameters
        ----------
        alpha : ArrayLike
            _description_
        x : ArrayLike
            _description_
        """
        raise NotImplementedError("Parametope.g must be implemented in subclasses")

    def contains (self, x:Array, **kwargs) -> bool :
        """Checks if x is in the parametope."""
        return jnp.all(jnp.logical_or(self.g(x) <= self.y, jnp.isclose(self.g(x), self.y, **kwargs)))

    @classmethod
    def from_parametope (cls, pt:'Parametope') :
        r"""Create an instance of the subclass from a Parametope instance"""
        return cls(pt.ox, pt.alpha, pt.y)

    # Always flatten parametope data into (ox, alpha, y)
    def tree_flatten (self) :
        return ((self.ox, self.alpha, self.y), type(self).__name__)

    @classmethod
    def tree_unflatten (cls, aux_data, children) :
        return cls.from_parametope(Parametope(*children))
    
    @property
    def dtype (self) -> jnp.dtype :
        return self.ox.dtype

    def __str__(self):
        ox_str = self.ox
        alpha_str = self.alpha
        y_str = self.y

        # ox_str = str(self.ox)
        # alpha_str = str(self.alpha)
        # y_str = str(self.y)

        return self.__class__.__name__ + f'(ox={ox_str}, alpha={alpha_str}, y={y_str})'

def g_parametope (g:Callable, name=None) -> Parametope :
    """Creates a Parametope subclass from a given nonlinearity g"""

    # Validation of g
    if not callable(g):
        raise ValueError("input g to g_parametope must be a callable function")

    # Check that g has the right signature
    import inspect
    sig = inspect.signature(g)
    params = sig.parameters
    if len(params) != 2 :
        raise ValueError("input g to g_parametope must have signature g(alpha, x)")

    if name is None :
        name = g.__name__ + 'Parametope'

    @register_pytree_node_class
    class gParametope (Parametope) :
        def g (self, x:ArrayLike) :
            return g(self.alpha, x - self.ox)
        
        @classmethod
        def from_parametope (cls, pt:Parametope) :
            return gParametope(pt.ox, pt.alpha, pt.y)

        def __str__ (self) :
            return name + f'(ox={self.ox}, alpha={self.alpha}, y={self.y})'

    return gParametope

