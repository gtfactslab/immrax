import jax
import jax.numpy as jnp
from ..system import System
from .parametope import Parametope, hParametope
from abc import ABC, abstractmethod
from immutabledict import immutabledict
from jaxtyping import Integer, Float, ArrayLike
from typing import Tuple, Union, List, Callable, Literal, Mapping
from diffrax import AbstractSolver, ODETerm, Euler, Dopri5, Tsit5, SaveAt, diffeqsolve
from ..inclusion import Permutation, standard_permutation, mjacM, interval, ut2i, i2ut, natif, jacM, icentpert, i2lu, lu2i
from ..embedding import embed
from ..refinement import SampleRefinement
from functools import partial
import equinox as eqx
from jax.experimental.jet import jet

class ParametopeEmbedding (ABC) :
    sys: System

    def __init__ (self, sys:System) :
        self.sys = sys

    @abstractmethod
    def _initialize (self, pt0:Parametope) -> ArrayLike :
        """Initialize the Embedding System for a particular initial set pt0

        Parameters
        ----------
        pt0 : hParametope
            _description_

        Returns
        -------
        ArrayLike
            aux0: Auxilliary states to evolve with the embedding system
        """
    
    @abstractmethod
    def _dynamics (self, t, state, *args) :
        """Embedding dynamics

        Parameters
        ----------
        t : _type_
            _description_
        state : _type_
            _description_
        """
    
    # @partial(jax.jit, static_argnums=(0, 4), static_argnames=("solver", "f_kwargs"))
    def compute_reachset (
        self,
        t0: Union[Integer, Float],
        tf: Union[Integer, Float],
        pt0: Parametope,
        inputs: List[Callable[[int, jax.Array], jax.Array]] = [],
        dt: float = 0.01,
        *,
        solver: Union[Literal["euler", "rk45", "tsit5"], AbstractSolver] = "tsit5",
        f_kwargs: immutabledict = immutabledict({}),
        **kwargs,
    ) :
        def func (t, x, args) :
            # Unpack the inputs
            return self._dynamics(t, x, *[u(t, x) for u in inputs], **f_kwargs)

        term = ODETerm(func)
        if solver == "euler":
            solver = Euler()
        elif solver == "rk45":
            solver = Dopri5()
        elif solver == "tsit5":
            solver = Tsit5()
        elif isinstance(solver, AbstractSolver):
            pass
        else:
            raise Exception(f"{solver=} is not a valid solver")

        aux0 = self._initialize(pt0)

        saveat = SaveAt(t0=True, t1=True, steps=True)
        return diffeqsolve(term, solver, t0, tf, dt, (pt0, aux0), saveat=saveat, **kwargs)

class AdjointEmbedding (ParametopeEmbedding) :
    def __init__(self, sys, kap:float=0.1, 
                 refine_factory:Callable[[ArrayLike], Callable]=partial(SampleRefinement, num_samples=10)):
        super().__init__(sys)
        # self.perm = perm if perm is not None else standard_permutation(sys.n)
        self.Jf_x = jax.jacfwd(sys.f, 1)
        self.Mf = mjacM(sys.f)
        self.Jf = jacM(sys.f)
        self.kap = kap
        self.refine_factory = refine_factory

    def _initialize (self, pt0:hParametope) -> ArrayLike :
        if not isinstance(pt0, hParametope):
            raise ValueError(f"{pt0=} is not a hParametope needed for AdjointEmbedding")

        # Setup refinement 
        alpha = pt0.alpha
        _id = lambda x : x
        self.refine = self.refine_factory(alpha).get_refine_func() if alpha.shape[0] > alpha.shape[1] else _id 
        return jnp.linalg.pinv(alpha)


    def _dynamics (self, t, state:Tuple[hParametope, ArrayLike], *args) :
        pt, aux = state
        ox = pt.ox

        K = len(pt.y) // 2
        ly = -pt.y[:K] # negative for lower bound
        uy = pt.y[K:]
        iy = lu2i(ly, uy)
        alpha = pt.alpha
        alpha_p = aux

        ## Adjoint dynamics + LICQ CBF


        J = self.Jf_x(t, ox)
        u0 = -alpha@J
        u0flat = u0.reshape(-1)

        # CBF: Enforce pairwise independence on the rows of alpha
        def barrier_LICQ (alpha) :
            # Normalize rows of alpha
            alpha = alpha / jnp.linalg.norm(alpha, axis=1, keepdims=True)
            # offdiagonal inner products of rows of alpha
            aaT = alpha @ alpha.T - jnp.eye(alpha.shape[0])
            # safe set defined by non unit offdiagonal inner products
            return 1. - jnp.max(aaT) - 1e-5

        balpha = barrier_LICQ(alpha)

        pLfh, Lfh = jax.jvp(barrier_LICQ, (alpha,), (u0,))
        unroll = lambda v : jax.jvp(barrier_LICQ, (alpha,), (v.reshape(alpha.shape),))
        pLgh, Lgh = jax.vmap(unroll)(jnp.eye(alpha.size))

        # Solution to QP
        ustar = jnp.where(Lfh + Lgh@u0flat + self.kap*balpha >= 0.,
                          jnp.zeros_like(u0flat), # constraint inactive
                          - (Lfh + Lgh@u0flat + self.kap*balpha)*Lgh.T/(Lgh@Lgh.T)).reshape(alpha.shape)
        # ustar = jnp.zeros_like(u0)

        ## Offset Dynamics given ustar
        
        big_iz = pt.hinv(pt.y)
        Jh = natif(jax.jacfwd(lambda z : jnp.asarray(pt.h(z))))
        def F_second (t, iy, *args) :
            iz = pt.hinv(i2ut(iy))
            def _get_second (oz, z) :
                primals = (t, alpha_p@oz + ox)
                series = ((0., 0.), (alpha_p@z, jnp.zeros_like(alpha_p@z)))
                _, coeffs = jet(self.sys.f, primals, series)
                return coeffs[1]
            res = natif(_get_second)(big_iz, iz)/2

            # Post first order cancellation
            return interval(Jh(iz))@(interval(ustar)@alpha_p@iz + interval(alpha)@res)
        E = embed(F_second)
        E_res = E(t, i2ut(iy), *args, refine=self.refine) 
        mul = jnp.concatenate((-jnp.ones_like(ly), jnp.ones_like(uy)))

        # hParametope dynamics in same pytree structure as pt
        pt_dot = pt.from_parametope(hParametope(self.sys.f(t, ox), u0 + ustar, E_res*mul))
        # sets d/dt [alpha_p @ alpha] = 0, so alpha_p @ alpha = I
        alpha_p_dot = alpha_p@(u0 + ustar)@alpha_p
        # alpha_p_dot = J@alpha_p


        return (pt_dot, alpha_p_dot)
