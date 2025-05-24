import abc
from functools import partial
from typing import Any, Callable, List, Literal, Union

from diffrax import (
    AbstractSolver,
    Dopri5,
    Euler,
    ODETerm,
    SaveAt,
    Solution,
    Tsit5,
    diffeqsolve,
)
from immutabledict import immutabledict
import jax
from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
from jaxtyping import Float, Integer
import sympy
from sympy2jax import SymbolicModule

__all__ = [
    "System",
    "ReversedSystem",
    "LinearTransformedSystem",
    "LiftedSystem",
    "OpenLoopSystem",
    "SympySystem",
    "Trajectory",
]

@register_pytree_node_class
class Trajectory:
    _ts: jnp.ndarray
    _ys: jnp.ndarray
    tfinite: jnp.ndarray

    def __init__(self, ts: jax.Array, ys: jax.Array, tfinite: jax.Array) -> None:
        self._ts = ts
        self._ys = ys
        self.tfinite = tfinite

    @staticmethod
    def from_diffrax(sol: Solution) -> "Trajectory":
        ts = sol.ts if sol.ts is not None else jnp.empty(0)
        ys = sol.ys if sol.ys is not None else jnp.empty(0)
        tfinite = jnp.isfinite(ts)
        return Trajectory(ts, ys, tfinite)

    def tree_flatten(self):
        return ((self._ts, self._ys, self.tfinite), "Trajectory")

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    # NOTE: Currently, Trajectory objects that were produced by vmapping over time will 
    # behave badly. Because this class is a pytree, a "list" of Trajectory objects is 
    # automatically flattened into just one object, and the dimensions of the children 
    # arrays are adjusted to accomodate this. If the indices of every trajectory do not 
    # correspond to the same time point, this will result in a Trajectory object where
    # some sub-trajectories might have different indices that are finite, making the 
    # below boolean index give a ragged result. 
    # 
    # This is difficult to deal with, and we don't currently have a good solution.
    #
    # Also note that if the trajectory was produced with an adaptive step size controller, 
    # then different trajectories might also have indices that map to different times, 
    # causing the same issue. (Though, in this case the offset is likely to be smaller.)
    @property
    def ts(self):
        filtered = self._ts[self.tfinite]
        shape = list(self._ts.shape)
        shape[-1] = jnp.sum(self.tfinite[(0,) * (self.tfinite.ndim - 1)]).item()
        return filtered.reshape(shape)

    @property
    def ys(self):
        filtered = self._ys[self.tfinite]
        shape = list(self._ys.shape)
        shape[-2] = jnp.sum(self.tfinite[(0,) * (self.tfinite.ndim - 1)]).item()
        return filtered.reshape(shape)


class EvolutionError(Exception):
    def __init__(self, t: Any, evolution: Literal["continuous", "discrete"]) -> None:
        super().__init__(
            f"Time {t} of type {type(t)} does not match evolution type {evolution}"
        )


class System(abc.ABC):
    """System

    A dynamical system of one of the following forms:

    .. math::
        \\dot{x} = f(t, x, \\dots), \\text{ or } \\x^+ = f(t, x, \\dots).

    where :math:`t\\in\\T\\in\\{\\mathbb{Z},\\mathbb{R}\\}` is a discrete or continuous time variable, :math:`x\\in\\mathbb{R}^n` is the state of the system, and :math:`\\dots` are some other inputs, perhaps control and disturbance.

    There are two main attributes that need to be defined in a subclass:

    - `evolution` : Literal['continuous', 'discrete'], which specifies whether the system is continuous or discrete.
    - `xlen` : int, which specifies the dimension of the state space.

    The main method that needs to be defined is `f(t, x, *args, **kwargs)`, which returns the time evolution of the state at time `t` and state `x`.
    """

    evolution: Literal["continuous", "discrete"]
    xlen: int

    @abc.abstractmethod
    def f(self, t: Union[Integer, Float], x: jax.Array, *args, **kwargs) -> jax.Array:
        """The right hand side of the system

        Parameters
        ----------
        t : Union[Integer, Float]
            The time of the system
        x : jax.Array
            The state of the system
        *args :
            Inputs (control, disturbance, etc.) as positional arguments depending on parent class.
        **kwargs :
            Other keyword arguments depending on parent class.

        Returns
        -------
        jax.Array
            The time evolution of the state

        """

    def fi(
        self, i: int, t: Union[Integer, Float], x: jax.Array, *args, **kwargs
    ) -> jax.Array:
        """The i-th component of the right hand side of the system

        Parameters
        ----------
        i : int
            component
        t : Union[Integer, Float]
            The time of the system
        x : jax.Array
            The state of the system
        *args :
            control inputs, disturbance inputs, etc. Depends on parent class.
        **kwargs :


        Returns
        -------
        jax.Array
            The i-th component of the time evolution of the state

        """

        return self.f(t, x, *args, **kwargs)[i]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.f(*args, **kwargs)

    @partial(jax.jit, static_argnums=(0, 4), static_argnames=("solver", "f_kwargs"))
    def compute_trajectory(
        self,
        t0: Union[Integer, Float],
        tf: Union[Integer, Float],
        x0: jax.Array,
        inputs: List[Callable[[int, jax.Array], jax.Array]] = [],
        dt: float = 0.01,
        *,
        solver: Union[Literal["euler", "rk45", "tsit5"], AbstractSolver] = "tsit5",
        f_kwargs: immutabledict = immutabledict({}),
        **kwargs,
    ) -> Trajectory:
        """Computes the trajectory of the system from time t0 to tf with initial condition x0.

        Parameters
        ----------
        t0 : Union[Integer,Float]
            Initial time
        tf : Union[Integer,Float]
            Final time
        x0 : jax.Array
            Initial condition
        inputs : Tuple[Callable[[int,jax.Array], jax.Array]], optional
            A tuple of Callables u(t,x) returning time/state varying inputs as positional arguments into f, by default ()
        dt : float, optional
            Time step, by default 0.01
        solver : Union[Literal['euler', 'rk45', 'tsit5'], AbstractSolver], optional
            Solver to use for diffrax, by default 'tsit5'
        f_kwargs : immutabledict, optional
            An immutabledict to pass as keyword arguments to the dynamics f, by default {}
        **kwargs :
            Additional kwargs to pass to the solver from diffrax

        Returns
        -------
        Trajectory
            _description_
        """

        def func(t, x, args):
            # Unpack the inputs
            return self.f(t, x, *[u(t, x) for u in inputs], **f_kwargs)

        if self.evolution == "continuous":
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

            saveat = SaveAt(t0=True, t1=True, steps=True)
            return diffeqsolve(term, solver, t0, tf, dt, x0, saveat=saveat, **kwargs)
            # return Trajectory.from_diffrax(
            #     diffeqsolve(term, solver, t0, tf, dt, x0, saveat=saveat, **kwargs)
            # )

        elif self.evolution == "discrete":
            if not isinstance(t0, int) or not isinstance(tf, int):
                raise Exception(
                    f"Times {t0=} and {tf=} must be integers for discrete evolution, got {type(t0)=} and {type(tf)=}"
                )

            # Use jax.lax.scan to compute the trajectory of the discrete system
            def step(x, t):
                xtp1 = func(t, x, None)
                return xtp1, xtp1

            times = jnp.arange(t0, tf + 1)
            _, traj = jax.lax.scan(step, x0, times)
            return Trajectory(times, jnp.vstack((x0, traj)), jnp.ones_like(times, dtype=bool))
        else:
            raise Exception(
                f"Evolution needs to be 'continuous' or 'discrete', got {self.evolution=}"
            )


class ReversedSystem(System):
    """ReversedSystem
    A system with time reversed dynamics, i.e. :math:`\\dot{x} = -f(t,x,...)`.
    """

    sys: System

    def __init__(self, sys: System) -> None:
        self.evolution = sys.evolution
        self.xlen = sys.xlen
        self.sys = sys

    def f(self, t: Union[Integer, Float], x: jax.Array, *args, **kwargs) -> jax.Array:
        return -self.sys.f(t, x, *args, **kwargs)


class LinearTransformedSystem(System):
    """Linear Transformed System
    A system with dynamics :math:`\\dot{x} = Tf(t, T^{-1}x, ...)` where :math:`T` is an invertible linear transformation.
    """

    sys: System

    def __init__(self, sys: System, T: jax.Array) -> None:
        self.evolution = sys.evolution
        self.xlen = sys.xlen
        self.sys = sys
        self.T = T
        self.Tinv = jnp.linalg.inv(T)

    def f(self, t: Union[Integer, Float], x: jax.Array, *args, **kwargs) -> jax.Array:
        return self.T @ self.sys.f(t, self.Tinv @ x, *args, **kwargs)


class LiftedSystem(System):
    """Lifted System
    A system with dynamics :math:`\\dot{x} = Hf(t, H^+x, ...)` where H^+Hx = x.
    """

    sys: System
    H: jax.Array
    Hp: jax.Array

    def __init__(self, sys: System, H: jax.Array, Hp: jax.Array) -> None:
        self.evolution = sys.evolution
        self.xlen = H.shape[0]
        self.sys = sys
        self.H = H
        self.Hp = Hp

    def f(self, t: Union[Integer, Float], x: jax.Array, *args, **kwargs) -> jax.Array:
        return self.H @ self.sys.f(t, self.Hp @ x, *args, **kwargs)


class OpenLoopSystem(System, abc.ABC):
    """OpenLoopSystem
    An open-loop nonlinear dynamical system of the form

    .. math::

        \\dot{x} = f(x,u,w),

    where :math:`x\\in\\mathbb{R}^n` is the state of the system, :math:`u\\in\\mathbb{R}^p` is a control input to the system, and :math:`w\\in\\mathbb{R}^q` is a disturbance input.
    """

    @abc.abstractmethod
    def f(
        self, t: Union[Integer, Float], x: jax.Array, u: jax.Array, w: jax.Array
    ) -> jax.Array:
        """The right hand side of the open-loop system

        Parameters
        ----------
        t : Union[Integer, Float]
            The time of the system
        x : jax.Array
            The state of the system
        u : jax.Array
            The control input to the system
        w : jax.Array
            The disturbance input to the system

        Returns
        -------
        jax.Array
            The time evolution of the state

        """

    def fi(
        self, i: int, t: Union[Integer, Float], x: jax.Array, u: jax.Array, w: jax.Array
    ) -> jax.Array:
        """The right hand side of the open-loop system

        Parameters
        ----------
        i : int
            component
        t : Union[Integer, Float]
            The time of the system
        x : jax.Array
            The state of the system
        u : jax.Array
            The control input to the system
        w : jax.Array
            The disturbance input to the system

        Returns
        -------
        jax.Array
            The i-th component of the time evolution of the state

        """
        return self.f(t, x, u, w)[i]


class SympySystem(OpenLoopSystem):
    t_var: sympy.Symbol
    x_vars: sympy.Matrix
    u_vars: sympy.Matrix
    w_vars: sympy.Matrix
    xlen: int
    ulen: int
    wlen: int
    f_eqn: List[sympy.Expr]
    _f: SymbolicModule
    _fi: List[SymbolicModule]

    def __init__(
        self,
        t_var: sympy.Symbol,
        x_vars: List[sympy.Symbol],
        u_vars: List[sympy.Symbol],
        w_vars: List[sympy.Symbol],
        f_eqn: List[sympy.Expr],
        evolution: Literal["continuous", "discrete"] = "continuous",
    ) -> None:
        """Initialize an open-loop system using Sympy.

        Parameters
        ----------
        t_var : sympy.Symbol
            Symbolic variable for the time
        x_vars : List[sympy.Symbol]
            Symbolic variables for the state
        u_vars : List[sympy.Symbol]
            Symbolic variables for the control input
        w_vars : List[sympy.Symbol]
            Symbolic variables for the disturbance input
        f_eqn : List[sympy.Expr]
            Symbolic expressions for the RHS
        evolution : Literal['continuous','discrete'], optional
            Type of time evolution, by default 'continuous'
        """
        self.evolution = evolution

        self.t_var = t_var
        self.x_vars = sympy.Matrix(x_vars)
        self.u_vars = sympy.Matrix(u_vars)
        self.w_vars = sympy.Matrix(w_vars)

        self.xlen = len(x_vars)
        self.ulen = len(u_vars)
        self.wlen = len(w_vars)

        self.f_eqn = [sympy.sympify(f_eqn_i) for f_eqn_i in f_eqn]
        self._f = SymbolicModule(self.f_eqn)
        self._fi = [SymbolicModule(f_eqn_i) for f_eqn_i in self.f_eqn]

    def _txuw_to_kwargs(
        self, t: Union[Integer, Float], x: jax.Array, u: jax.Array, w: jax.Array
    ) -> dict:
        """Return the kwargs to the SymbolicModule from sympy2jax given a specific value.

        Returns
        -------
        dict
            {name: value} pairs for each symbol and value
        """
        return {
            self.t_var.name: t,
            **{xvar.name: xval for xvar, xval in zip(self.x_vars, x)},
            **{uvar.name: uval for uvar, uval in zip(self.u_vars, u)},
            **{wvar.name: wval for wvar, wval in zip(self.w_vars, w)},
        }

    def f(
        self, t: Union[Integer, Float], x: jax.Array, u: jax.Array, w: jax.Array
    ) -> jax.Array:
        """Get the value of the RHS of the dynamical system.

        .. math ::

            f (t, x, u, w)

        Parameters
        ----------
        t : Union[Integer, Float]
            state value
        x : jax.Array
            state value
        u : jax.Array
            control value
        w : jax.Array
            disturbance value

        Returns
        -------
        jax.Array
            :math:`f(t,x,u,w)`

        """
        x = jnp.asarray(x)
        u = jnp.asarray(u)
        w = jnp.asarray(w)
        return jnp.asarray(self._f(**self._txuw_to_kwargs(t, x, u, w)))

    def fi(
        self, i: int, t: Union[Integer, Float], x: jax.Array, u: jax.Array, w: jax.Array
    ) -> jax.Array:
        """Get the value of the i-th component of the RHS of the dynamical system.

        Parameters
        ----------
        i : int
            component
        t : Union[Integer, Float]
            state value
        x : jax.Array
            state value
        u : jax.Array
            control value
        w : jax.Array
            disturbance value

        Returns
        -------
        jax.Array
            :math:`f_i(t,x,u,w)`

        """
        x = jnp.asarray(x)
        u = jnp.asarray(u)
        w = jnp.asarray(w)
        return jnp.asarray(self._fi[i](**self._txuw_to_kwargs(t, x, u, w)))
