import jax
import jax.numpy as jnp
from ..system import System
from .dual_star import DualStar
from abc import ABC, abstractmethod
from immutabledict import immutabledict
from jaxtyping import Integer, Float
from typing import Union, List, Callable, Literal
from diffrax import AbstractSolver, ODETerm, Euler, Dopri5, Tsit5, SaveAt, diffeqsolve
from ..inclusion import Permutation, standard_permutation

# def star_reach (sys: System, ds0: DualStar) :
#     def reach ()

#     return

class DualStarEmbeddingSystem(ABC) :
    sys: System

    def __init__ (self, sys: System) :
        self.sys = sys
        self.jacf = jax.jacfwd(sys.f, 1)
        self.g = None
    
    @abstractmethod
    def _dynamics (self, t, state, *args) :
        pass

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

        self.g = ds0.g
        state = (ds0.ox, ds0.H, jnp.linalg.pinv(ds0.H), ds0.ly, ds0.uy)

        saveat = SaveAt(t0=True, t1=True, steps=True)
        return diffeqsolve(term, solver, t0, tf, dt, state, saveat=saveat, **kwargs)
        # return Trajectory.from_diffrax(
        #     diffeqsolve(term, solver, t0, tf, dt, x0, saveat=saveat, **kwargs)
        # )


class DualStarMJacMEmbeddingSystem (DualStarEmbeddingSystem) :
    def __init__ (self, sys: System, permutation: Permutation=None) :
        super().__init__(sys)
        self.permutation = permutation if permutation is not None else standard_permutation(sys.xlen)

    def _dynamics (self, t, state, *args):
        ox, H, Hp, ly, uy = state

        return 

def ds_mjacemb (sys: System, permutation:Permutation=None) :
    return DualStarMJacMEmbeddingSystem(sys)
