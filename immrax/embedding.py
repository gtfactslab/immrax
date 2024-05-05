import abc
import jax
from jax import lax
import jax.numpy as jnp
from jaxtyping import Integer, Float
import sympy
from typing import List, Literal, Union, Any, Callable, Tuple
from .system import System
from .inclusion import interval, natif, jacif, mjacif, i2ut, ut2i, Interval
from .control import Control, ControlledSystem
from functools import partial

class EmbeddingSystem (System, abc.ABC) :
    """EmbeddingSystem
    
    Embeds a System
    
    ..math::
        \\mathbb{R}^n \\times \\text{inputs} \\to\\mathbb{R}^n`
    
    into an Embedding System evolving on the upper triangle
    
    ..math::
        \\mathcal{T}^{2n} \\times \\text{embedded inputs} \\to \\mathbb{T}^{2n}.
    """
    sys:System

    @abc.abstractmethod
    def E(self, t:Union[Integer,Float], x: jax.Array, *args, **kwargs) -> jax.Array :
        """The right hand side of the embedding system.

        Parameters
        ----------
        t : Union[Integer, Float]
            The time of the embedding system.
        x : jax.Array
            The state of the embedding system.
        *args :
            interval-valued control inputs, disturbance inputs, etc. Depends on parent class.
        **kwargs :
            

        Returns
        -------
        jax.Array
            The time evolution of the state on the upper triangle

        """

    def Ei(self, i:int, t:Union[Integer,Float], x: jax.Array, *args, **kwargs) -> jax.Array :
        """The right hand side of the embedding system.

        Parameters
        ----------
        i : int
            component
        t : Union[Integer, Float]
            The time of the embedding system.
        x : jax.Array
            The state of the embedding system.
        *args :
            interval-valued control inputs, disturbance inputs, etc. Depends on parent class.

        Returns
        -------
        jax.Array
            The i-th component of the time evolution of the state on the upper triangle

        """
        return self.E(t, x, *args, **kwargs)[i]

    def f(self, t:Union[Integer,Float], x: jax.Array, *args, **kwargs) -> jax.Array:
        return self.E(t, x, *args, **kwargs)

    def fi(self, i: int, t:Union[Integer,Float], x: jax.Array, *args, **kwargs) -> jax.Array:
        return self.Ei(i, t, x, *args, **kwargs)

class InclusionEmbedding (EmbeddingSystem) :
    """EmbeddingSystem
    
    Embeds a System
    
    ..math::
        \\mathbb{R}^n \\times \\text{inputs} \\to\\mathbb{R}^n`,
    
    into an Embedding System evolving on the upper triangle
    
    ..math::
        \\mathcal{T}^{2n} \\times \\text{embedded inputs} \\to \\mathbb{T}^{2n},
    
    using an Inclusion Function for the dynamics f.
    """
    sys:System
    F:Callable[..., Interval]
    Fi:List[Callable[..., Interval]]
    def __init__(self, sys:System, F:Callable[..., Interval], Fi:Callable[..., Interval] = None) -> None:
        """Initialize an EmbeddingSystem using a System and an inclusion function for f.

        Args:
            sys (System): The system to be embedded
            if_transform (InclusionFunction): An inclusion function for f.
        """
        self.sys = sys
        self.F = F
        self.Fi = Fi if Fi is not None else (lambda i, t, x, *args, **kwargs : self.F(t,x,*args,**kwargs)[i])
        self.evolution = sys.evolution
        self.xlen = sys.xlen * 2 

    # def E(self, t: Any, x: jax.Array, *args, 
    #         refine:Callable[[Interval], Interval]|None=None, **kwargs) -> jax.Array:
    #     if refine is None :
    #         refine = lambda x : x

    #     if self.evolution == 'continuous' :
    #         n = self.sys.xlen
    #         ret = jnp.empty(self.xlen)
    #         for i in range(n) :
    #             _xi = refine(ut2i(jnp.copy(x).at[i+n].set(x[i])))
    #             ret = ret.at[i].set(self.Fi(i, interval(t), _xi, *args, **kwargs).lower)
    #             # ret = ret.at[i].set(self.F(interval(t), ut2i(_xi), *args, **kwargs).lower[i])
    #             x_i = refine(ut2i(jnp.copy(x).at[i].set(x[i+n])))
    #             ret = ret.at[i+n].set(self.Fi(i, interval(t), x_i, *args, **kwargs).upper)
    #             # ret = ret.at[i+n].set(self.F(interval(t), ut2i(x_i), *args, **kwargs).upper[i])
    #         return ret
    #     elif self.evolution == 'discrete' :
    #         # Convert x from ut to i, compute through F, convert back to ut.
    #         return i2ut(self.F(interval(t), refine(ut2i(x)), *args, **kwargs))
    #     else :
    #         raise Exception("evolution needs to be 'continuous' or 'discrete'")
        
    def E(self, t: Any, x: jax.Array, *args, 
            refine:Callable[[Interval], Interval]|None=None, **kwargs) -> jax.Array:
        if refine is None :
            refine = lambda x : x

        if self.evolution == 'continuous' :
            n = self.sys.xlen
            _x = x[:n]; x_ = x[n:]

            Fkwargs = partial(self.Fi, **kwargs)

            # Computing F on the faces of the hyperrectangle

            _X = interval(jnp.tile(_x, (n,1)), jnp.where(jnp.eye(n), _x, jnp.tile(x_, (n,1))))
            _E = jax.vmap(Fkwargs, (0, None, 0) + (None,)*len(args))(jnp.arange(n), t, _X, *args)

            X_ = interval(jnp.where(jnp.eye(n), x_, jnp.tile(_x, (n,1))), jnp.tile(x_, (n,1)))
            E_ = jax.vmap(Fkwargs, (0, None, 0) + (None,)*len(args))(jnp.arange(n), t, X_, *args)

            # return jnp.concatenate((jnp.diag(_E.lower), jnp.diag(E_.upper)))
            return jnp.concatenate((jnp.diag(_E.lower), jnp.diag(E_.upper)))

        elif self.evolution == 'discrete' :
            # Convert x from ut to i, compute through F, convert back to ut.
            return i2ut(self.F(interval(t), refine(ut2i(x)), *args, **kwargs))
        else :
            raise Exception("evolution needs to be 'continuous' or 'discrete'")
        
    # def Ei(self, i: int, t: Any, x: jax.Array, *args, **kwargs) -> jax.Array:
    #     if self.evolution == 'continuous':
    #         n = self.sys.xlen
    #         if i < n :
    #             _xi = jnp.copy(x).at[i+n].set(x[i])
    #             return self.Fi(i, interval(t), ut2i(_xi), *args, **kwargs).lower
    #         else :
    #             x_i = jnp.copy(x).at[i].set(x[i+n])
    #             return self.Fi(i, interval(t), ut2i(x_i), *args, **kwargs).upper
    #     elif self.evolution == 'discrete' :
    #         if i < self.sys.xlen :
    #             return self.Fi(i, interval(t), ut2i(x), *args, **kwargs).lower
    #         else :
    #             return self.Fi(i, interval(t), ut2i(x), *args, **kwargs).upper
    #     else :
    #         raise Exception("evolution needs to be 'continuous' or 'discrete'")

def ifemb (sys:System, F:Callable[..., Interval]) :
    """Creates an EmbeddingSystem using an inclusion function for the dynamics of a System.

    Parameters
    ----------
    sys : System
        System to embed
    F : Callable[..., Interval]
        Inclusion function for the dynamics of sys.

    Returns
    -------
    EmbeddingSystem
        Embedding system from the inclusion function transform.

    """
    return InclusionEmbedding(sys, F)

class TransformEmbedding (InclusionEmbedding) :
    def __init__(self, sys:System, if_transform = natif) -> None:
        """Initialize an EmbeddingSystem using a System and an inclusion function transform.

        Parameters
        ----------
        sys : System
            _description_
        if_transform : IFTransform
            _description_. Defaults to natif.

        Returns
        -------

        """
        F = if_transform(sys.f)
        Fi = [if_transform(sys.fi[i]) for i in range(sys.xlen)]
        super().__init__(sys, F, Fi) 
    
def natemb (sys:System) :
    """Creates an EmbeddingSystem using the natural inclusion function of the dynamics of a System.

    Parameters
    ----------
    sys : System
        System to embed

    Returns
    -------
    EmbeddingSystem
        Embedding system from the natural inclusion function transform.

    """
    return TransformEmbedding(sys, if_transform=natif)

def jacemb (sys:System) :
    """Creates an EmbeddingSystem using the Jacobian-based inclusion function of the dynamics of a System.

    Parameters
    ----------
    sys : System
        System to embed

    Returns
    -------
    EmbeddingSystem
        Embedding system from the Jacobian-based inclusion function transform.

    """
    return TransformEmbedding(sys, if_transform=jacif)

def mjacemb (sys:System) :
    """Creates an EmbeddingSystem using the Mixed Jacobian-based inclusion function of the dynamics of a System.

    Parameters
    ----------
    sys : System
        System to embed

    Returns
    -------
    EmbeddingSystem
        Embedding system from the Mixed Jacobian-based inclusion function transform.

    """
    return TransformEmbedding(sys, if_transform=mjacif)

# class InterconnectedEmbedding (EmbeddingSystem) :
#     def __init__(self, sys:System, if_transform:IFTransform = natif) -> None:
#         self.sys = sys
#         self.F = if_transform(sys.f)
#         self.Fi = [if_transform(partial(sys.fi, i)) for i in range(sys.xlen)]
#         self.evolution = sys.evolution
#         self.xlen = sys.xlen * 2 


