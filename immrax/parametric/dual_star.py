import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
from typing import Tuple, Iterable, List, Callable
from jaxtyping import ArrayLike
import numpy as onp
from ..inclusion import Interval, interval
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
    # g: Tuple[Callable]
    H: Tuple[ArrayLike]
    ly: Tuple[ArrayLike]
    uy: Tuple[ArrayLike]
    K: int

    def __init__ (self, ox, H, ly, uy) :
        Ks = [len(H), len(ly), len(uy)]

        if len(set(Ks)) != 1:
            raise ValueError(f"Dimension mismatch: {Ks}")

        self.ox = ox
        self.H = H
        self.ly = ly
        self.uy = uy
        self.K = Ks[0]
    
    def g (self, i:int, a:ArrayLike) :
        """Evaluates the nonlinearity g_i at a

        Parameters
        ----------
        i : int
            Nonlinearity index, g_i
        a : ArrayLike
            Input to the nonlinearity
        """
        pass

    def ginv (self, i:int, iy:Interval) :
        """Overapproximating inverse image of the nonlineraity g_i

        Parameters
        ----------
        i : int
            Nonlinearity index, g_i
        iy : Interval
            Interval image on output
        """
        pass
    
    def iover (self) :
        """Overapproximate the DualStar with an axis aligned interval using ginv"""
        invs = [self.ginv(H@interval(ly, uy)) for H, ly, uy in zip(self.H, self.ly, self.uy)]
        # Entrywise max and min over each entry considered
        l = jnp.max(jnp.array([_.lower for _ in invs]), axis=0)
        u = jnp.min(jnp.array([_.upper for _ in invs]), axis=0)
        return interval(l, u)
    
    def tree_flatten(self) :
        return ((self.ox, self.H, self.ly, self.uy), type(self).__name__)
    
    @classmethod
    def from_ds (cls, ds:'DualStar') :
        return ds

    @classmethod
    def tree_unflatten(cls, _, children) :
        return cls.from_ds(DualStar(*children))

    @property
    def tree (self) :
        return jax.tree_util.tree_structure(self)

    @property
    def dtype(self) -> jnp.dtype:
        return self.H.dtype

    def __str__ (self):
        return f"DualStar({self.ox}, {self.H}, {self.ly}, {self.uy})"

def intersect_shared_ox (ds1, ds2) :
    """Intersects two dual stars, assuming they have the same ox"""
    @register_pytree_node_class
    class DualStarIntersection (DualStar) :
        def __init__ (self, _ds1, _ds2) :
            super().__init__(_ds1.ox, _ds1.H + _ds2.H, _ds1.ly + _ds2.ly, _ds1.uy + _ds2.uy)
    return DualStarIntersection

@register_pytree_node_class
class HODualStar :
    """Higher Order Dual Star. Defines the set

    {x : ly <= H1[x - ox] + H2[(x - ox)^{x2} + ... + HN[(x - ox)^{xN}]] <= uy}
    """
    ox: ArrayLike
    Hs: Tuple[ArrayLike]
    ly: ArrayLike
    uy: ArrayLike
    N: int

    def __init__ (self, ox, Hs, ly, uy) :
        self.ox = ox
        self.Hs = Hs
        self.ly = ly
        self.uy = uy
        self.N = len(Hs)

    def p (self, x:ArrayLike) :
        dx = x - self.ox
        # ret_Hs = [H.dot(dx) for H in self.Hs]
        ret_Hs = []

        for i in range (self.N) :
            res = self.Hs[i]
            for j in range (i+1) :
                res = res.dot(dx)
            ret_Hs.append(res)

        return jnp.sum(jnp.array(ret_Hs), axis=0)

    def g (self, a:ArrayLike) :
        pass

    def tree_flatten(self) :
        return ((self.ox, self.Hs, self.ly, self.uy), type(self).__name__)
    
    @classmethod
    def from_hods (cls, hods:'DualStar') :
        return hods

    @classmethod
    def tree_unflatten(cls, _, children) :
        return cls.from_hods(HODualStar(*children))

    @property
    def tree (self) :
        return jax.tree_util.tree_structure(self)

    @property
    def dtype(self) -> jnp.dtype:
        return self.H.dtype

    def __str__ (self):
        return f"HODualStar[{self.N}]({self.ox}, {self.Hs}, {self.ly}, {self.uy})"

    
