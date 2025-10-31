import abc
from typing import Any, Union, List

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from diffrax import Solution

import numpy as np

__all__ = [
    "RawTrajectory",
    "RawContinuousTrajectory",
    "RawDiscreteTrajectory",
    "Trajectory",
    "ContinuousTrajectory",
    "DiscreteTrajectory",
]


class RawTrajectory(abc.ABC):
    """Abstract base class for raw trajectories.

    These trajectories directly store the padded arrays used in JAX computations.
    """

    ts: jax.Array
    ys: jax.Array

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

    def __init__(self, raw_trajectory: RawTrajectory):
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

    def __init__(self, raw_trajectory: RawTrajectory):
        assert isinstance(raw_trajectory, RawContinuousTrajectory)
        super().__init__(raw_trajectory)


class DiscreteTrajectory(Trajectory):
    """Convenience wrapper for discrete trajectories."""

    def __init__(self, raw_trajectory: RawTrajectory):
        assert isinstance(raw_trajectory, RawDiscreteTrajectory)
        super().__init__(raw_trajectory)
