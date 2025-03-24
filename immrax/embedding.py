import abc
from functools import partial
from typing import Any, Callable, Literal, Union

import jax
import jax.numpy as jnp
from jaxtyping import Float, Integer

from .refinement import SampleRefinement, LinProgRefinement
from .inclusion import Interval, i2ut, interval, jacif, mjacif, natif, ut2i
from .system import LiftedSystem, System

__all__ = [
    "EmbeddingSystem",
    "InclusionEmbedding",
    "TransformEmbedding",
    "ifemb",
    "natemb",
    "jacemb",
    "mjacemb",
    "embed",
]

class EmbeddingSystem(System, abc.ABC):
    """EmbeddingSystem

    Embeds a System

    ..math::
        \\mathbb{R}^n \\times \\text{inputs} \\to\\mathbb{R}^n`

    into an Embedding System evolving on the upper triangle

    ..math::
        \\mathcal{T}^{2n} \\times \\text{embedded inputs} \\to \\mathbb{T}^{2n}.
    """

    sys: System

    @abc.abstractmethod
    def E(self, t: Union[Integer, Float], x: jax.Array, *args, **kwargs) -> jax.Array:
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

    def Ei(
        self, i: int, t: Union[Integer, Float], x: jax.Array, *args, **kwargs
    ) -> jax.Array:
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

    def f(self, t: Union[Integer, Float], x: jax.Array, *args, **kwargs) -> jax.Array:
        return self.E(t, x, *args, **kwargs)

    def fi(
        self, i: int, t: Union[Integer, Float], x: jax.Array, *args, **kwargs
    ) -> jax.Array:
        return self.Ei(i, t, x, *args, **kwargs)


class InclusionEmbedding(EmbeddingSystem):
    """EmbeddingSystem

    Embeds a System

    ..math::
        \\mathbb{R}^n \\times \\text{inputs} \\to\\mathbb{R}^n`,

    into an Embedding System evolving on the upper triangle

    ..math::
        \\mathcal{T}^{2n} \\times \\text{embedded inputs} \\to \\mathbb{T}^{2n},

    using an Inclusion Function for the dynamics f.
    """

    sys: System
    F: Callable[..., Interval]
    Fi: Callable[..., Interval]

    def __init__(
        self,
        sys: System,
        F: Callable[..., Interval],
        Fi: Callable[..., Interval] | None = None,
    ) -> None:
        """Initialize an EmbeddingSystem using a System and an inclusion function for f.

        Args:
            sys (System): The system to be embedded
            if_transform (InclusionFunction): An inclusion function for f.
        """
        self.sys = sys
        self.F = F
        self.Fi = (
            Fi
            if Fi is not None
            else (lambda i, t, x, *args, **kwargs: self.F(t, x, *args, **kwargs)[i])
        )
        self.evolution = sys.evolution
        self.xlen = sys.xlen * 2

    def E(
        self,
        t: Any,
        x: jax.Array,
        *args,
        refine: Callable[[Interval], Interval] | None = None,
        **kwargs,
    ) -> jax.Array:
        t = interval(t)

        if refine is not None:
            convert = lambda x: refine(ut2i(x))
            Fkwargs = lambda t, x, *args: self.F(t, refine(x), *args, **kwargs)
        else:
            convert = ut2i
            Fkwargs = partial(self.F, **kwargs)

        x_int = convert(x)

        if self.evolution == "continuous":
            n = self.sys.xlen
            _x = x_int.lower
            x_ = x_int.upper

            # Computing F on the faces of the hyperrectangle

            _X = interval(
                jnp.tile(_x, (n, 1)), jnp.where(jnp.eye(n), _x, jnp.tile(x_, (n, 1)))
            )
            _E = jax.vmap(Fkwargs, (None, 0) + (None,) * len(args))(t, _X, *args)
            # _E = jnp.array([self.Fi[i](t, _X[i], *args, **kwargs).lower for i in range(n)])

            X_ = interval(
                jnp.where(jnp.eye(n), x_, jnp.tile(_x, (n, 1))), jnp.tile(x_, (n, 1))
            )
            E_ = jax.vmap(Fkwargs, (None, 0) + (None,) * len(args))(t, X_, *args)
            # E_ = jnp.array([self.Fi[i](t, X_[i], *args, **kwargs).upper for i in range(n)])

            # return jnp.concatenate((_E, E_))
            return jnp.concatenate((jnp.diag(_E.lower), jnp.diag(E_.upper)))

        elif self.evolution == "discrete":
            # Convert x from ut to i, compute through F, convert back to ut.
            return i2ut(self.F(interval(t), x_int, *args, **kwargs))
        else:
            raise Exception("evolution needs to be 'continuous' or 'discrete'")


def ifemb(sys: System, F: Callable[..., Interval]):
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

def embed (F: Callable[..., Interval]) :
    def E (
        t: Any,
        x: jax.Array,
        *args,
        refine: Callable[[Interval], Interval] | None = None,
        **kwargs,
    ) :
        n = len(x) // 2
        _x = x[:n]
        x_ = x[n:]

        if refine is not None:
            Fkwargs = lambda t, x, *args: F(t, refine(x), *args, **kwargs)
        else:
            Fkwargs = partial(F, **kwargs)

        # Computing F on the faces of the hyperrectangle

        if n > 1 :
            _X = interval(
                jnp.tile(_x, (n, 1)), jnp.where(jnp.eye(n), _x, jnp.tile(x_, (n, 1)))
            )
            _E = interval(jax.vmap(Fkwargs, (None, 0) + (None,) * len(args))(t, _X, *args))

            X_ = interval(
                jnp.where(jnp.eye(n), x_, jnp.tile(_x, (n, 1))), jnp.tile(x_, (n, 1))
            )
            E_ = interval(jax.vmap(Fkwargs, (None, 0) + (None,) * len(args))(t, X_, *args))
            return jnp.concatenate((jnp.diag(_E.lower), jnp.diag(E_.upper)))
        else :
            _E = Fkwargs(t, interval(_x)).lower
            E_ = Fkwargs(t, interval(x_)).upper
            return jnp.array([_E, E_])
        
    return E


class TransformEmbedding(InclusionEmbedding):
    def __init__(self, sys: System, if_transform=natif) -> None:
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
        # Fi = [if_transform(sys.fi[i]) for i in range(sys.xlen)]
        super().__init__(sys, F)


def natemb(sys: System):
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


def jacemb(sys: System):
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


def mjacemb(sys: System):
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


class AuxVarEmbedding(InclusionEmbedding):
    """
    Embedding system defined by auxiliary variables.n

    Attributes:
        H: Matrix of auxiliary variables to add
        Hp: psuedo-inverse of H
    """

    def __init__(
        self,
        sys: System,
        H: jax.Array,
        mode: Literal["sample", "linprog"] = "sample",
        if_transform: Callable[[Callable[..., jnp.ndarray]], Callable[..., Interval]]
        | None = None,
        F: Callable[..., Interval] | None = None,
        num_samples=10,
    ) -> None:
        """
        Embedding system defined by auxiliary variables. Given a base system with dimension n
        and matrix H m by n, the base system is first lifted to dimension m by adding m-n
        auxiliary variables. Each aux var is a linear combination of some of the real state
        variables, defined by the coefficients of the rows of H. Because of this, the subspace
        defined by y = Hx is invariant in the lifted state under the base system dynamics.

        The lifted system is then embedded onto the upper triangle with either the inclusion
        function F or if_transform given.

        The intervals of the embedded system can then be refined by the subspace invariance of
        the lifted system. There are two methods to do this, "sample" and "linprog", and the
        method is chosen by the mode argument.

        Args:
            sys: Base system to embed
            H: Matrix of auxiliary variables to add
            mode: Whether to refine by sampling or solving a LP. Defaults to sample
            if_transform: How to construct the inclusion function for the embedding system
            F: For greater control, allows you to pass an inclusion function directly. NOTE:
            is required to be an inclusion function for the *lifted* system, not the bases system
            num_samples (): How many samples to take for sampling refinement. Defaults to 10
        """
        self.H = H
        # self.Hp = jnp.linalg.pinv(H)
        self.Hp = jnp.hstack(
            (jnp.eye(H.shape[1]), jnp.zeros((H.shape[1], H.shape[0] - H.shape[1])))
        )

        liftsys = LiftedSystem(sys, self.H, self.Hp)

        if mode == "sample":
            self.IH = SampleRefinement(H, num_samples).get_refine_func()
        elif mode == "linprog":
            self.IH = LinProgRefinement(H).get_refine_func()
        else:
            raise ValueError(
                "Invalid mode argument. Mode must be either 'sample' or 'linprog'."
            )

        if F is None and if_transform is None:
            F = natif(liftsys.f)  # default to natif
        elif if_transform is not None and F is None:
            F = if_transform(liftsys.f)
        elif F is not None and if_transform is None:
            pass  # do nothing, take F as given
        else:
            raise ValueError(
                "Cannot specify both an inclusion function F and if_transform"
            )

        super().__init__(liftsys, F)

    def E(
        self,
        t: Any,
        x: jax.Array,
        *args,
        refine: Callable[[Interval], Interval] | None = None,
        **kwargs,
    ) -> jax.Array:
        if refine is not None:
            raise (
                Exception(
                    "Class AuxVarEmbedding does not support passing refine as an argument, since the refinement is calculated from the auxillary variables."
                )
            )

        return super().E(t, x, *args, refine=self.IH, **kwargs)
