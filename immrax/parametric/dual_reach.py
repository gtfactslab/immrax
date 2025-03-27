import jax
import jax.numpy as jnp
from ..system import System
from .dual_star import DualStar, HODualStar
from abc import ABC, abstractmethod
from immutabledict import immutabledict
from jaxtyping import Integer, Float, ArrayLike
from typing import Union, List, Callable, Literal
from diffrax import AbstractSolver, ODETerm, Euler, Dopri5, Tsit5, SaveAt, diffeqsolve
from ..inclusion import Permutation, standard_permutation, mjacM, interval, ut2i, i2ut, natif, jacM, icentpert
from ..embedding import embed
from ..refinement import SampleRefinement
from functools import partial
import equinox as eqx
from jax.experimental.jet import jet

# def star_reach (sys: System, ds0: DualStar) :
#     def reach ()

#     return

class DualStarEmbeddingSystem(ABC) :
    sys: System

    def __init__ (self, sys: System) :
        self.sys = sys
        self.jacf = jax.jacfwd(sys.f, 1)
        self.aux0 = None
    
    @abstractmethod
    def _initialize (self, ds0: DualStar) :
        """Initialize the Embedding System for a particular initial set ds0

        Parameters
        ----------
        ds0 : DualStar
            The initial set

        Returns
        -------
        aux0 : Any
            Auxilliary states to evolve with the system
        """

    @abstractmethod
    def _dynamics (self, t, state, *args) :
        pass

    @partial(jax.jit, static_argnums=(0, 4), static_argnames=("solver", "f_kwargs"))
    # @partial(eqx.filter_jit, static_argnums=(0, 4), static_argnames=("solver", "f_kwargs"))
    def compute_reachset (
        self,
        t0: Union[Integer, Float],
        tf: Union[Integer, Float],
        ds0: DualStar,
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

        # self.ds0 = ds0
        # print(ds0.H)
        # state = (ds0.ox, ds0.H, [jnp.linalg.pinv(H) for H in ds0.H], ds0.ly, ds0.uy)

        aux0 = self._initialize(ds0)
        # print(f'finished _initialize {aux0}')

        saveat = SaveAt(t0=True, t1=True, steps=True)
        # print(ds0, aux0)
        return diffeqsolve(term, solver, t0, tf, dt, (ds0, aux0), saveat=saveat, **kwargs)
        # return Trajectory.from_diffrax(
        #     diffeqsolve(term, solver, t0, tf, dt, x0, saveat=saveat, **kwargs)
        # )


class DualStarMJacMEmbeddingSystem (DualStarEmbeddingSystem) :
    def __init__ (self, sys: System, permutation: Permutation=None, 
                  refine_factory:Callable[[ArrayLike], Callable]=partial(SampleRefinement, num_samples=10)) :
        super().__init__(sys)
        # TODO: fix standard permutation logic
        self.permutation = permutation if permutation is not None else standard_permutation(sys.xlen)
        self.Jf_x = jax.jacfwd(sys.f, 1)
        self.Mf = mjacM(sys.f)
        self.Jf = jacM(sys.f)
        self.refine = None
        self.refine_factory = refine_factory

    def _initialize (self, ds0) :
        # Setup refinement 
        _id = lambda x : x
        # self.refine = [self.refine_factory(H) if H.shape[0] > H.shape[1] else _id for H in ds0.H]
        # print(ds0.H[0].shape)
        # print(ds0.H)
        self.refine = [self.refine_factory(H).get_refine_func() if H.shape[0] > H.shape[1] else _id for H in ds0.H]
        # print(self.refine)
        # return ([(jnp.linalg.pinv(H), jax.jacfwd(self.sys.f, 1)(0, ds0.ox)) for H in ds0.H])
        return ([(jnp.linalg.pinv(H), jnp.zeros_like(jax.jacfwd(self.sys.f, 1)(0, ds0.ox))) for H in ds0.H])

    def _dynamics (self, t, state, *args):
        # ox, Hs, Hps, lys, uys = state
        ds, aux = state
        # Hps, A = aux
        ox = ds.ox

        J = self.Jf_x(t, ox)

        retHs = []
        retlys = []
        retuys = []
        ret_aux = []

        # ret = [self.sys.f(t, ox), [], [], [], []]
        # ret =
        # ret_aux = Hps

        # print(lys)


        # print(retHs)
        for i in range (len(ds.H)) :
            H = ds.H[i]; #Hp = Hps[i]
            Hp, Ai = aux[i]
            # Ai = jnp.zeros_like(J)
            # Ai = J if i == 0 else jnp.zeros_like(J)
            Ai = J
            # Ai = (J - J.T)/2
            ly = ds.ly[i]; uy = ds.uy[i]
            iy = interval(jnp.atleast_1d(ly), jnp.atleast_1d(uy))
            # print('ly, uy', ly.shape, uy.shape)
            Jg = natif(jax.jacfwd(lambda z : jnp.asarray(ds.g(i, z))))

            def h (H) :
                # Penalize linear dependence between rows of H
                HHT = H@H.T
                return jnp.sum((HHT - jnp.eye(HHT.shape[0]))**2)

                # # # Normalize the rows of H
                # # H_norm = H / jnp.linalg.norm(H, axis=1, keepdims=True)
                
                # # # Compute the cosine similarity matrix
                # # cosine_sim_matrix = jnp.dot(H_norm, H_norm.T) - jnp.eye(H.shape[0])
                
                # return jnp.sum(cosine_sim_matrix)
                
            ex_Hdot = -0.005*jax.grad(h)(H)
            # ex_Hdot = -0.01*jax.grad(h)(H)
            # ex_Hdot = -0.02*jax.grad(h)(H)
            # ex_Hdot = -2*jax.grad(h)(H)
            # ex_Hdot = -6*jax.grad(h)(H)
            # ex_Hdot = -4*jax.grad(h)(H)
            # ex_Hdot = -0.02*jax.grad(h)(H)
            # ex_Hdot = jnp.zeros_like(H)



            def F (t, iy, *args) :
                iz = ds.ginv(i, iy)
                _, iJx = self.Jf(interval(t), interval(Hp)@iz + ox, *args)

                iz = ds.ginv(i, iy)
                MM = self.Mf(interval(t), interval(Hp)@iz + ox, *args, 
                            centers=((jnp.zeros(1), ox),), permutations=self.permutation)[0]
                Mx = MM[1]
                
                # return interval(Jg(iz))@interval(H)@(iJx - Ai)@(interval(Hp)@iz)
                return interval(Jg(iz))@(interval(H)@(Mx - Ai) + ex_Hdot)@(interval(Hp)@iz)
            
            big_iz = ds.ginv(i, iy)

            def F_second (t, iy, *args) :
                iz = ds.ginv(i, iy)
                def _get_second (oz, z) :
                    primals = (t, Hp@oz + ox)
                    # series = ((0., 0.), (z, jnp.zeros_like(z)))
                    series = ((0., 0.), (Hp@z, jnp.zeros_like(Hp@z)))
                    _, coeffs = jet(self.sys.f, primals, series)
                    return coeffs[1]
                res = natif(_get_second)(big_iz, iz)/2

                # print(((interval(H)@(interval(J) - Ai))@Hp@iz).shape)

                return interval(Jg(iz))@((interval(H)@(interval(J) - Ai) + ex_Hdot)@Hp@iz + interval(H)@res)

            # E = embed(F)
            E = embed(F_second)
            # print((-H@Ai).shape)
            retHs.append(-H@Ai + ex_Hdot)
            print(f'here {i}, {iy}')
            try:
                E_res = E(t, i2ut(iy), *args, refine=self.refine[i])
                # print(F_second(t, iy, *args))
                # E_res = i2ut(F_second(t, iy, *args).atleast_1d())
            except Exception as e:
                print(f"in exception {e}")
                raise e

            print(E_res)
            # E_res = i2ut(F(t, self.refine[i](iy), *args))
            retlys.append(E_res[:len(ly)])
            retuys.append(E_res[len(ly):])

            ret_aux.append((Ai@Hp, .3*(J - Ai)))
        # print(retHs, retlys, retuys)
        ret = ds.__class__.from_ds(DualStar(self.sys.f(t, ox), retHs, retlys, retuys))

        print(state)
        print(ret)

        print('in _dynamics')
        return (ret, ret_aux)

def ds_mjacemb (sys: System, permutation:Permutation=None) :
    return DualStarMJacMEmbeddingSystem(sys, permutation)

class BaseHODualStarEmbeddingSystem(ABC) :
    sys: System

    def __init__ (self, sys: System) :
        self.sys = sys
    
    @abstractmethod
    def _initialize (self, ds0: HODualStar) :
        """Initialize the Embedding System for a particular initial set ds0

        Parameters
        ----------
        ds0 : DualStar
            The initial set

        Returns
        -------
        aux0 : Any
            Auxilliary states to evolve with the system
        """

    @abstractmethod
    def _dynamics (self, t, state, *args) :
        pass

    @partial(jax.jit, static_argnums=(0, 4), static_argnames=("solver", "f_kwargs"))
    def compute_reachset (
        self,
        t0: Union[Integer, Float],
        tf: Union[Integer, Float],
        hods0: HODualStar,
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

        aux0 = self._initialize(hods0)
        saveat = SaveAt(t0=True, t1=True, steps=True)
        return diffeqsolve(term, solver, t0, tf, dt, hods0, saveat=saveat, **kwargs)


_fact = jnp.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880
], dtype=float)

class HODualStarEmbeddingSystem (BaseHODualStarEmbeddingSystem) :
    def __init__ (self, sys:System) :
        super().__init__(sys)
        self.N = None
        self.Dn_f = []

    def _initialize(self, ds0):
        self.N = ds0.N
        self.Dn_f = [jax.jacfwd(self.sys.f, 1)]

        # N+1 derivatives---set dynamics to order N, N+1 for Lagrange remainder
        for i in range (self.N) :
            self.Dn_f.append(jax.jacfwd(self.Dn_f[-1], 1))
        return (0.,)

    def _dynamics(self, t, state, *args):
        # hods, aux = state
        hods = state

        # SET DYNAMICS

        # Use Dn_f to cancelling orders <= N.
        retHs = []
        f0 = self.sys.f(t, hods.ox)
        Dfs = [] 
        for i in range (self.N) :
            # Order i + 1 term 

            # Dfs[i] = (1/i!) D^i f (t, ox)
            Dfs.append(self.Dn_f[i](t, hods.ox)/_fact[i])
            
            # \dot{H}_i = ...
            retHs.append(jnp.sum(jnp.array([
                -(n+1)*hods.Hs[n]@Dfs[i-n]
                for n in range (i+1)
            ]), axis=0))
    
        # Use interval analysis to overapproximate offset dynamics
        # TODO: Can we use jet to make this more efficient? Maybe we don't need dense computations.
        retlys = jnp.zeros_like(hods.ly)
        retuys = jnp.zeros_like(hods.uy)

        terms = []

        ix = icentpert(jnp.zeros(2), 1.)
        Z = natif()

        for i in range (self.N) :
            # Order N + i + 1 term 
            
            # termi = 
            pass

        ret = hods.__class__.from_hods(HODualStar(
            f0, tuple(retHs), retlys, retuys))
        ret_aux = (0.,)

        # return (ret, ret_aux)
        return ret