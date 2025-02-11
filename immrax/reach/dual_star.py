import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
from typing import Tuple, Iterable, List, Callable
from jaxtyping import ArrayLike
import numpy as onp
# import immrax as irx

@register_pytree_node_class
class DualStar :
    """Dual Star Set. Defines the set

    {x : ly_k <= g_k(H_k(x - ox)) <= uy_k for every k}

    Raises
    ------
    ValueError
        _description_
    """

    ox: ArrayLike
    g: Tuple[Callable]
    H: Tuple[ArrayLike]
    ly: Tuple[ArrayLike]
    uy: Tuple[ArrayLike]
    K: int

    def __init__ (self, ox, g, H, ly, uy) :
        Ks = [len(g), len(H), len(ly), len(uy)]

        if len(set(Ks)) != 1:
            raise ValueError(f"Dimension mismatch: {Ks}")

        self.ox = ox
        self.g = g
        self.H = H
        self.ly = ly
        self.uy = uy
        self.K = Ks[0]
    
    def tree_flatten(self) :
        return ((self.ox, self.g, self.H, self.ly, self.uy), "DualStar")
    
    @classmethod
    def tree_unflatten(cls, _, children) :
        return cls(*children)

    @property
    def tree (self) :
        return jax.tree_util.tree_structure(self)

    def __str__ (self):
        return f"DualStar({self.ox}, {self.g}, {self.H}, {self.ly}, {self.uy})"

class DualStarIntersection :
    pass

class DualStarUnion :
    pass
