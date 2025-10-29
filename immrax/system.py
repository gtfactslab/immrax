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
import numpy as np
from jaxtyping import Float, Integer

__all__ = [
    "System",
    "ReversedSystem",
    "LinearTransformedSystem",
    "LiftedSystem",
    "OpenLoopSystem",
    "RawTrajectory",
    "RawContinuousTrajectory",
    "RawDiscreteTrajectory",
    "Trajectory",
    "ContinuousTrajectory",
    "DiscreteTrajectory",
]


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
class RawTrajectory(abc.ABC):
    """Abstract base class for raw trajectories.

    These trajectories directly store the padded arrays used in JAX computations.
    """

    @abc.abstractmethod
    def to_convenience(self) -> "Trajectory":
        """Converts a raw trajectory to a convenience trajectory."""
        ...


@register_pytree_node_class
class RawContinuousTrajectory(RawTrajectory):
    """Raw continuous trajectory, wrapping a diffrax.Solution."""

    def __init__(self, solution: Solution):
        self.solution = solution

    def __getattr__(self, name: str) -> Any:
        return getattr(self.solution, name)

    def tree_flatten(self):
        children = (self.solution,)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0])

    def to_convenience(self) -> "ContinuousTrajectory":
        return ContinuousTrajectory(self)


@register_pytree_node_class
class RawDiscreteTrajectory(RawTrajectory):
    """Raw discrete trajectory."""

    def __init__(self, ts: jax.Array, ys: jax.Array):
        self.ts = ts
        self.ys = ys

    def tree_flatten(self):
        children = (self.ts, self.ys)
        aux_data = None
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def to_convenience(self) -> "DiscreteTrajectory":
        return DiscreteTrajectory(self)


class Trajectory:
    """Convenience wrapper for trajectories.

    This class provides access to the valid, unpadded data from a raw trajectory.
    """

    ts: Union[jax.Array, List[jax.Array]]
    ys: Union[jax.Array, List[jax.Array]]

    def is_ragged(self) -> bool:
        """Returns True if the trajectory is ragged, meaning `ts` and `ys` are lists of arrays.

        When a `RawTrajectory` is created from a `vmap`'d computation over parameters
        that affect the trajectory length (e.g., final time), the resulting batch of
        trajectories may be "ragged" - that is, different trajectories in the batch
        may have different numbers of time steps. To handle this, the `ts` and `ys`
        attributes of the `Trajectory` object will be lists of arrays, where each
        array corresponds to a single trajectory. The `is_ragged` method can be used
        to check for this case.
        """
        return isinstance(self.ts, list)


class ContinuousTrajectory(Trajectory):
    """Convenience wrapper for continuous trajectories."""

    def __init__(self, raw_trajectory: RawContinuousTrajectory):
        """Initializes a convenience trajectory wrapper from a raw trajectory.

        This constructor handles both single and batched trajectories. For batched
        trajectories, it can handle both cases where trajectories have the same
        length (non-ragged) and different lengths (ragged).

        For single trajectories, `ts` and `ys` will be `jax.Array`.
        For non-ragged batched trajectories, `ts` and `ys` will be `jax.Array`
        with leading batch dimensions.
        For ragged batched trajectories, `ts` and `ys` will be lists of `jax.Array`,
        where each array in the list corresponds to a single trajectory in the batch.
        """
        raw_ts = raw_trajectory.ts
        raw_ys = raw_trajectory.ys

        if raw_ts is None:  # diffrax can return None ts
            self.ts = jnp.empty(0)
            self.ys = (
                jnp.empty((0, raw_ys.shape[-1]))
                if raw_ys is not None
                else jnp.empty((0, 0))
            )
            return

        tfinite_mask = jnp.isfinite(raw_ts)

        if raw_ts.ndim <= 1:
            self.ts = raw_ts[tfinite_mask]
            self.ys = raw_ys[tfinite_mask]
            return

        time_axis = -1
        num_finite = jnp.sum(tfinite_mask, axis=time_axis)

        is_ragged = (
            not jnp.all(num_finite == num_finite.flatten()[0])
            if num_finite.size > 0
            else False
        )

        if not is_ragged:
            k = num_finite.flatten()[0].item() if num_finite.size > 0 else 0
            self.ts = raw_ts[..., :k]
            self.ys = raw_ys[..., :k, :]
        else:
            self.ts = []
            self.ys = []
            batch_shape = raw_ts.shape[:-1]
            for batch_idx in np.ndindex(batch_shape):
                n = num_finite[batch_idx].item()
                self.ts.append(raw_ts[batch_idx, :n].squeeze())
                self.ys.append(raw_ys[batch_idx, :n].squeeze())


class DiscreteTrajectory(Trajectory):
    """Convenience wrapper for discrete trajectories."""

    def __init__(self, raw_trajectory: RawDiscreteTrajectory):
        """Initializes a convenience trajectory wrapper from a raw trajectory.

        This constructor handles both single and batched trajectories. For batched
        trajectories, it can handle both cases where trajectories have the same
        length (non-ragged) and different lengths (ragged).

        For single trajectories, `ts` and `ys` will be `jax.Array`.
        For non-ragged batched trajectories, `ts` and `ys` will be `jax.Array`
        with leading batch dimensions.
        For ragged batched trajectories, `ts` and `ys` will be lists of `jax.Array`,
        where each array in the list corresponds to a single trajectory in the batch.
        """
        raw_ts = raw_trajectory.ts
        raw_ys = raw_trajectory.ys

        if raw_ts is None:  # diffrax can return None ts
            self.ts = jnp.empty(0)
            self.ys = (
                jnp.empty((0, raw_ys.shape[-1]))
                if raw_ys is not None
                else jnp.empty((0, 0))
            )
            return

        tfinite_mask = jnp.isfinite(raw_ts)

        if raw_ts.ndim <= 1:
            self.ts = raw_ts[tfinite_mask]
            self.ys = raw_ys[tfinite_mask]
            return

        time_axis = -1
        num_finite = jnp.sum(tfinite_mask, axis=time_axis)

        is_ragged = (
            not jnp.all(num_finite == num_finite.flatten()[0])
            if num_finite.size > 0
            else False
        )

        if not is_ragged:
            k = num_finite.flatten()[0].item() if num_finite.size > 0 else 0
            self.ts = raw_ts[..., :k]
            self.ys = raw_ys[..., :k, :]
        else:
            self.ts = []
            self.ys = []
            batch_shape = raw_ts.shape[:-1]
            for batch_idx in np.ndindex(batch_shape):
                n = num_finite[batch_idx].item()
                self.ts.append(raw_ts[batch_idx, :n].squeeze())
                self.ys.append(raw_ys[batch_idx, :n].squeeze())


class EvolutionError(Exception):
    def __init__(self, t: Any, evolution: Literal["continuous", "discrete"]) -> None:
        super().__init__(
            f"Time {t} of type {type(t)} does not match evolution type {evolution}"
        )


class System(abc.ABC):
    r"""System

    A dynamical system of one of the following forms:

    .. math::
        \dot{x} = f(t, x, \dots), \text{ or } x^+ = f(t, x, \dots).

    where :math:`t\in T\in\{\mathbb{Z},\mathbb{R}\}` is a discrete or continuous time variable, :math:`x\in\mathbb{R}^n` is the state of the system, and :math:`\dots` are some other inputs, perhaps control and disturbance.

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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.f(*args, **kwargs)

    @partial(
        jax.jit, static_argnums=(0, 4), static_argnames=("solver", "f_kwargs", "inputs")
    )
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
    ) -> RawTrajectory:
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
        RawTrajectory
            Flow line / trajectory of the system from the initial condition x0 to the final time tf
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
            sol = diffeqsolve(term, solver, t0, tf, dt, x0, saveat=saveat, **kwargs)
            return RawContinuousTrajectory(sol)

        elif self.evolution == "discrete":
            if not (
                jnp.issubdtype(jnp.array(t0).dtype, jnp.integer)
                and jnp.issubdtype(jnp.array(tf).dtype, jnp.integer)
            ):
                raise Exception(
                    f"Times {t0=} and {tf=} must be integers for discrete evolution, got {type(t0)=} and {type(tf)=}"
                )

            max_steps = 4096
            times = jnp.where(
                jnp.arange(max_steps) <= tf - t0,
                t0 + jnp.arange(max_steps),
                jnp.inf * jnp.ones(max_steps),
            )

            # Use jax.lax.scan to compute the trajectory of the discrete system
            def step(x, t):
                xtp1 = jax.lax.cond(
                    t < tf, lambda: func(t, x, None), lambda: jnp.inf * x0
                )
                return xtp1, xtp1

            _, traj = jax.lax.scan(step, x0, times)
            return RawDiscreteTrajectory(times, jnp.vstack((x0, traj[:-1])))
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
