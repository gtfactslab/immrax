import abc
from itertools import permutations
from typing import Callable

import jax
import jax.numpy as jnp

from immrax.optim import linprog
from immrax.inclusion import Interval, icopy, interval
from immrax.utils import angular_sweep, null_space


class Refinement(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def get_refine_func(self) -> Callable[[Interval], Interval]:
        pass


class LinProgRefinement(Refinement):
    H: jnp.ndarray

    def __init__(self, H: jnp.ndarray) -> None:
        self.H = H
        super().__init__()

    def get_refine_func(self) -> Callable[[Interval], Interval]:
        A_ub = jnp.vstack((self.H, -self.H))

        def var_refine(idx: int, ret: Interval) -> Interval:
            # I update b_eq and b_ub here because ret is shrinking
            b_ub = jnp.concatenate(
                (ret.upper, -ret.lower)
            )  # TODO: try adding buffer region *inside* the bounds to collapsed face
            obj_vec_i = self.H[idx]

            sol_min = linprog(
                obj=obj_vec_i,
                A_ub=A_ub,
                b_ub=b_ub,
                unbounded=True,
            )

            sol_max = linprog(
                obj=-obj_vec_i,
                A_ub=A_ub,
                b_ub=b_ub,
                unbounded=True,
            )

            # If a vector that gives extra info on this var is found, refine bounds
            new_lower_i = jnp.where(
                sol_min.success,
                jnp.maximum(sol_min.fun, ret.lower[idx]),
                ret.lower[idx],
            )[0]
            retl = ret.lower.at[idx].set(new_lower_i)
            new_upper_i = jnp.where(
                sol_max.success,
                jnp.minimum(-sol_max.fun, ret.upper[idx]),
                ret.upper[idx],
            )[0]
            retu = ret.upper.at[idx].set(new_upper_i)

            return interval(retl, retu)

        def I_r(y: Interval) -> Interval:
            # for i in range(n):
            #     ret = var_refine(ret, i)

            return jax.lax.fori_loop(0, y.shape[0], var_refine, icopy(y))

        return I_r


class SampleRefinement(Refinement):
    H: jnp.ndarray
    Hp: jnp.ndarray
    N: jnp.ndarray
    A_lib: jnp.ndarray
    num_samples: int

    def __init__(self, H: jnp.ndarray, num_samples: int = 10) -> None:
        self.num_samples = num_samples
        self.H = H
        # self.Hp = jnp.linalg.pinv(H)
        self.Hp = jnp.hstack(
            (jnp.eye(H.shape[1]), jnp.zeros((H.shape[1], H.shape[0] - H.shape[1])))
        )

        self.N = jnp.array(
            [
                jnp.squeeze(null_space(jnp.vstack([jnp.eye(H.shape[1]), aug_var]).T))
                for aug_var in H[H.shape[1] :]
            ]
        ).T
        # assert not jnp.any(jnp.isnan(self.N))
        self.N = jnp.vstack([self.N[: H.shape[1]], jnp.diag(self.N[-1])])

        # Sample aux vars independently
        self.A_lib = self.N.T

        # Sample aux vars pairwise
        if self.N.shape[1] > 1:
            points = angular_sweep(num_samples)
            extended_points = jnp.hstack(
                [
                    points,
                    jnp.zeros((points.shape[0], self.N.shape[1] - points.shape[1])),
                ]
            )

            def permutation(mat: jnp.ndarray, perm):
                permuted_matrix = jnp.zeros_like(mat)
                for i, p in enumerate(perm):
                    permuted_matrix = permuted_matrix.at[:, p].set(mat[:, i])
                return permuted_matrix

            points_permutations = jax.vmap(permutation, in_axes=(None, 0))(
                extended_points,
                jnp.array(list(permutations(range(self.N.shape[1]), 2))),
            )
            points_permutations = points_permutations.reshape(
                -1, points_permutations.shape[-1]
            )
            points_permutations = points_permutations @ self.N.T
            self.A_lib = jnp.vstack([self.A_lib, points_permutations])
            # assert jnp.allclose(self.A_lib @ self.H, 0, atol=1e-6)

        super().__init__()

    def get_refine_func(self) -> Callable[[Interval], Interval]:
        def vec_refine(null_vector: jax.Array, var_index: jax.Array, y: Interval):
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

        mat_refine = jax.vmap(vec_refine, in_axes=(0, None, None), out_axes=0)
        mat_refine_all = jax.vmap(mat_refine, in_axes=(None, 0, None), out_axes=1)

        def best_refinement(y: Interval):
            refinements = mat_refine_all(self.A_lib, jnp.arange(len(y)), y)
            return interval(
                jnp.max(refinements.lower, axis=0), jnp.min(refinements.upper, axis=0)
            )

        return best_refinement
