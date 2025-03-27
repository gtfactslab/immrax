import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
from typing import Tuple, Iterable, List, Callable
from jaxtyping import ArrayLike
import numpy as onp
from ..inclusion import Interval, interval

@register_pytree_node_class
class Parametope :
    """Parametope. Defines the set
    
    {x : g(alpha, x - ox) <= y}
    
    """
    def __init__ (self, ox, alpha, y) :
        self.ox = ox
        self.alpha = alpha
        self.y = y

    def g (self, alpha:ArrayLike, x:ArrayLike) :
        """Evaluates the nonlinearity g at alpha, x

        Parameters
        ----------
        alpha : ArrayLike
            _description_
        x : ArrayLike
            _description_
        """
        pass

    # Always flatten parametope data into (ox, alpha, y)
    def tree_flatten (self) :
        return ((self.ox, self.alpha, self.y), type(self).__name__)
    
    # Override in subclasses to unpack the flattened data
    @classmethod
    def from_parametope (cls, pt:'Parametope') :
        return pt
    
    @classmethod
    def tree_unflatten (cls, aux_data, children) :
        return cls.from_parametope(Parametope(*children))

    @property
    def dtype (self) -> jnp.dtype :
        return self.ox.dtype

    def __str__(self):
        return f'Parametope(ox={self.ox}, alpha={self.alpha}, y={self.y})'

@register_pytree_node_class
class hParametope (Parametope) :
    """Defines a parametope with a particular structured nonlinearity
    
    g(alpha, x - ox) = h(alpha @ (x - ox))

    and y split into lower and upper bounds y = (ly, uy)

    """


    def h(self, z:ArrayLike) :
        """Evaluates the nonlinearity h at z

        Parameters
        ----------
        z : ArrayLike
            Input to the nonlinearity
        """
        pass

    def g(self, alpha:ArrayLike, x:ArrayLike) :
        """Evaluates the nonlinearity g at alpha, x

        Parameters
        ----------
        z : ArrayLike
            Input to the nonlinearity
        """
        return self.h(jnp.dot(alpha, x - self.ox))

    def hinv (self, iy:Interval) :
        """Overapproximating inverse image of the nonlinearity h

        Parameters
        ----------
        iy : ArrayLike
            _description_
        """
        pass
    
    def k_face (self, k:int) -> Interval :
        """Overapproximate the k-face of the hParametope"""

    # Override in subclasses to unpack the flattened data
    @classmethod
    def from_parametope (cls, pt:'hParametope') :
        return pt
    
    @classmethod
    def tree_unflatten (cls, aux_data, children) :
        return cls.from_parametope(hParametope(*children))
