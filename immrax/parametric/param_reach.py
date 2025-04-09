import jax
import jax.numpy as jnp
from ..system import System
from .parametope import Parametope, hParametope
from abc import ABC, abstractmethod
from immutabledict import immutabledict
from jaxtyping import Integer, Float, ArrayLike
from typing import Tuple, Union, List, Callable, Literal, Mapping
from diffrax import AbstractSolver, ODETerm, Euler, Dopri5, Tsit5, SaveAt, diffeqsolve
from ..inclusion import Interval, Permutation, standard_permutation, mjacM, interval, ut2i, i2ut, natif, jacM, icentpert, i2lu, lu2i, icopy
from ..embedding import embed
from ..refinement import SampleRefinement
from ..utils import null_space
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
    def __init__(self, sys, alpha_p0, N0, kap:float=0.1, permutation=None) :
                #  refine_factory:Callable[[ArrayLike], Callable]=partial(SampleRefinement, num_samples=10)):
        super().__init__(sys)
        self.Jf_x = jax.jacfwd(sys.f, 1)
        self.Mf = mjacM(sys.f)
        self.Jf = jacM(sys.f)
        self.kap = kap
        self.alpha_p0 = alpha_p0
        self.N0 = N0
        self.permutation = permutation

    def _initialize (self, pt0:hParametope) -> ArrayLike :
        if not isinstance(pt0, hParametope):
            raise ValueError(f"{pt0=} is not a hParametope needed for AdjointEmbedding")

        # Setup refinement 
        alpha = pt0.alpha
        # _id = lambda x : x
        # self.refine = self.refine_factory(alpha).get_refine_func() if alpha.shape[0] > alpha.shape[1] else _id 
        # alpha_p = jnp.linalg.pinv(alpha)
        # N = null_space(alpha.T)
        # print(N@alpha)
        # return (alpha_p, N)
        return (self.alpha_p0, self.N0)

    def _dynamics (self, t, state:Tuple[hParametope, ArrayLike], *args, **kwargs) :
        pt, aux = state
        ox = pt.ox

        K = len(pt.y) // 2
        # ly = -pt.y[:K] # negative for lower bound
        # uy = pt.y[K:]
        # iy = lu2i(ly, uy)
        y = pt.y
        alpha = pt.alpha
        alpha_p, N = aux

        ## Adjoint dynamics + LICQ CBF

        args_centers = (arg.center for arg in args) 
        centers = (jnp.array([t]), ox) + tuple(args_centers)

        J = self.Jf_x(*centers)
        u0 = -alpha@J
        u0flat = u0.reshape(-1)

        # CBF: Enforce pairwise independence on the rows of alpha

        PENALTY = 0

        if PENALTY == 0 :
            ustar = jnp.zeros_like(u0)

        elif PENALTY == 1 :
            def soft_overmax (x, eps=1e-5) :
                return jnp.max(jnp.exp(x) / jnp.sum(jnp.exp(x)))

            def barrier_LICQ (alpha) :
                # Normalize rows of alpha
                alpha = alpha / jnp.linalg.norm(alpha, axis=1, keepdims=True)
                # offdiagonal inner products of rows of alpha
                aaT = alpha @ alpha.T - jnp.eye(alpha.shape[0])
                # safe set defined by non unit offdiagonal inner products
                # return 1. - soft_overmax(aaT) - 0.01
                return 1. - jnp.max(aaT) - 0.01
                # return jnp.sum(aaT)

            balpha = barrier_LICQ(alpha)
            k = self.kap*balpha

            pLfh, Lfh = jax.jvp(barrier_LICQ, (alpha,), (u0,))
            unroll = lambda v : jax.jvp(barrier_LICQ, (alpha,), (v.reshape(alpha.shape),))
            pLgh, Lgh = jax.vmap(unroll)(jnp.eye(alpha.size))

            # Solution to QP
            ustar = jnp.where(Lfh + Lgh@u0flat + k*balpha >= 0.,
                            jnp.zeros_like(u0flat), # constraint inactive
                            - (Lfh + Lgh@u0flat + k*balpha)*Lgh.T/(Lgh@Lgh.T)).reshape(alpha.shape)
        elif PENALTY == 2 :
            ustar = jnp.zeros_like(u0)
            def soft (H) :
                HHT = H @ H.T 
                return jnp.sum((HHT - jnp.eye(H.shape[0]))**2)
            ustar = -self.kap*jax.grad(soft)(alpha)


        ## Offset Dynamics given ustar

        MJACM = False
        
        # For properly handling signs in lower offsets
        K_2 = len(y) // 2
        mul = jnp.concatenate((-jnp.ones(K_2), jnp.ones(K_2)))
        big_iz = pt.hinv(pt.y)
        Jh = natif(jax.jacfwd(lambda z : jnp.asarray(pt.h(z))))

        def refine (y: Interval):
            if len(N) > 0 :
                refinements = _mat_refine_all(N, jnp.arange(len(y)), y)
                return interval(
                    jnp.max(refinements.lower, axis=0), jnp.min(refinements.upper, axis=0)
                )
            else :
                return y

        if MJACM :
            if self.permutation is None :
                lenperm = sum([len(arg) for arg in centers])
                self.permutation = standard_permutation(lenperm)

            MM = self.Mf(t, interval(alpha_p)@big_iz + ox, *args, \
                        centers=(centers,), permutations=self.permutation)[0]
            ls = []; us = []
            for M, arg in zip(MM[2:], args) :
                term = interval(M)@arg 
                ls.append(term.lower); us.append(term.upper)
            dist = interval(jnp.sum(jnp.asarray(ls)), jnp.sum(jnp.asarray(us)))

            def F (t, iy, *args) :
                iy = refine(iy)
                iz = pt.hinv(i2ut(iy)*mul)

                # _, iJx = self.Jf(interval(t), interval(Hp)@iz + ox, *args)
                # MM = self.Mf(t, interval(alpha_p)@big_iz + ox, *args, \
                #             centers=(centers,), permutations=self.permutation)[0]
                Mx = MM[1]

                empty = jnp.any(iy.lower > iy.upper)
                def _zero () :
                    return interval(jnp.zeros_like(iz.lower))
                def _ret () :
                    # Post first order cancellation
                    PH = Jh(iz)
                    return interval(PH[len(PH)//2:,:])@( (interval(alpha)@(Mx - J) + ustar)@(interval(alpha_p)@iz) + dist)

                return jax.lax.cond(empty, _zero, _ret)
                
            E = embed(F)
        else :
            def F_second (t, iy, *args) :
                iy = refine(iy)
                iz = pt.hinv(i2ut(iy)*mul)

                empty = jnp.any(iy.lower > iy.upper)
                def _zero () :
                    return interval(jnp.zeros_like(iz.lower))

                def _ret () :
                    def _get_second (oz, z) :
                        primals = (t, alpha_p@oz + ox)
                        series = ((0., 0.), (alpha_p@z, jnp.zeros_like(alpha_p@z)))
                        _, coeffs = jet(self.sys.f, primals, series)
                        return coeffs[1]
                    res = natif(_get_second)(big_iz, iz)

                    # Post first order cancellation
                    PH = Jh(iz)
                    return interval(PH[len(PH)//2:,:])@(interval(ustar)@alpha_p@iz + interval(alpha)@res)

                return jax.lax.cond(empty, _zero, _ret)
            
            E = embed(F_second)


        E_res = E(t, y*mul, *args) * mul
        # E_res = jnp.zeros_like(mul)

        # hParametope dynamics in same pytree structure as pt
        pt_dot = pt.from_parametope(hParametope(self.sys.f(*centers), u0 + ustar, E_res))
                                                # jnp.where(jnp.logical_and(y <= 1e-2, E_res <= 0), jnp.zeros_like(y), E_res)))
                                                # jnp.where(E_res <= 0, jnp.zeros_like(y), E_res)))

        # sets d/dt [alpha_p @ alpha] = 0, so alpha_p @ alpha = I
        alpha_p_dot = -alpha_p@(u0 + ustar)@alpha_p
        # alpha_p_dot = J@alpha_p

        # sets d/dt [N @ alpha] = 0, so N @ alpha = 0
        N_dot = -N@(u0 + ustar)@alpha_p
        # N_dot = jnp.zeros_like(N)


        return (pt_dot, (alpha_p_dot, N_dot))

def _vec_refine(null_vector: jax.Array, var_index: jax.Array, y: Interval):
    ret = icopy(y)

    # Set up linear algebra computations for the refinement
    bounding_vars = interval(null_vector.at[var_index].set(0))
    ref_var = interval(null_vector[var_index])
    b1 = lambda: ((-bounding_vars @ ret) / ref_var) & ret[var_index]
    b2 = lambda: ret[var_index]

    # Compute refinement based on null vector, if possible
    ndb0 = jnp.abs(null_vector[var_index]) > 1e-10
    ret = jax.lax.cond(ndb0, b1, b2)

    # fix fpe problem with upper < lower
    retu = jnp.where(ret.upper >= ret.lower, ret.upper, ret.lower)
    return interval(ret.lower, retu)

_mat_refine = jax.vmap(_vec_refine, in_axes=(0, None, None), out_axes=0)
_mat_refine_all = jax.vmap(_mat_refine, in_axes=(None, 0, None), out_axes=1)
