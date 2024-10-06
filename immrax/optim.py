import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from typing import Tuple
from functools import partial

import scipy.optimize as opt


@register_pytree_node_class
class SimplexStep:
    feasible: jax.Array
    unbounded: jax.Array
    x: jax.Array
    basis: jax.Array
    n_basis: jax.Array
    B_inv: jax.Array
    b_hat: jax.Array
    simplex_muls: jax.Array
    reduced_cost_muls: jax.Array

    def __init__(
        self,
        feasible,
        unbounded,
        x,
        basis,
        n_basis,
        B_inv,
        b_hat,
        simplex_muls,
        reduced_cost_muls,
    ):
        self.feasible = feasible
        self.unbounded = unbounded
        self.x = x
        self.basis = basis
        self.n_basis = n_basis
        self.B_inv = B_inv
        self.b_hat = b_hat
        self.simplex_muls = simplex_muls
        self.reduced_cost_muls = reduced_cost_muls

    def tree_flatten(self):
        return (
            (
                self.feasible,
                self.unbounded,
                self.x,
                self.basis,
                self.n_basis,
                self.B_inv,
                self.b_hat,
                self.simplex_muls,
                self.reduced_cost_muls,
            ),
            "SimplexStep",
        )

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    @property
    def success(self) -> bool:
        return jnp.logical_and(self.feasible, jnp.logical_not(self.unbounded))


def fuzzy_argmin(arr: jnp.ndarray, tolerance: float = 1e-5) -> jax.Array:
    min_val = jnp.min(arr)
    within_tolerance = jnp.abs(arr - min_val) <= tolerance
    indices = jnp.arange(arr.shape[0])
    valid_indices = jnp.where(within_tolerance, indices, jnp.inf)
    return jnp.argmin(valid_indices)


@partial(jax.jit, static_argnames=["unbounded"])
def _standard_form(
    obj: jax.Array,
    A_eq: jax.Array,
    b_eq: jax.Array,
    A_ub: jax.Array,
    b_ub: jax.Array,
    unbounded: bool,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    if unbounded:
        # Convert from unbounded vars to non-negative vars
        obj = jnp.concatenate((obj, -obj))
        A_eq = jnp.hstack((A_eq, -A_eq))
        A_ub = jnp.hstack((A_ub, -A_ub))

    # Ensure RHS of equality constraints are positive
    idx = jnp.less(b_eq, 0).repeat(A_eq.shape[1]).reshape(A_eq.shape)
    A_eq = jax.lax.select(idx, -A_eq, A_eq)
    A_eq = A_eq.reshape((A_eq.shape[0], obj.shape[0]))  # hack to make stacking work
    b_eq = jnp.abs(b_eq)

    # Add slack vars to turn inequalities into equalities
    obj = jnp.concatenate((obj, jnp.zeros_like(b_ub)))
    A_eq = jnp.hstack((A_eq, jnp.zeros((A_eq.shape[0], b_ub.shape[0]))))

    idx = jnp.less(b_ub, 0)
    slack_signs = jax.lax.select(
        idx, -jnp.ones_like(b_ub), jnp.ones_like(b_ub)
    )  # track direction of inequalities
    idx = idx.repeat(A_ub.shape[1]).reshape(A_ub.shape)
    A_ub = jax.lax.select(idx, -A_ub, A_ub)  # flip signs to make RHS positive
    b_ub = jnp.abs(b_ub)

    A_ub = jnp.hstack((A_ub, jnp.diag(slack_signs)))
    A_ub = A_ub.reshape((A_ub.shape[0], obj.shape[0]))  # hack to make stacking work

    A = jnp.vstack((A_eq, A_ub))
    b = jnp.concatenate((b_eq, b_ub))

    return (A, b, obj)


def _iteration_needed(step: SimplexStep) -> jax.Array:
    _iteration_needed = jnp.less(
        step.reduced_cost_muls, -1e-5
    ).any()  # Stop when optimal solution found
    _iteration_needed = jnp.logical_and(
        _iteration_needed, step.feasible
    )  # Stop if infeasible
    _iteration_needed = jnp.logical_and(
        _iteration_needed, jnp.logical_not(step.unbounded)
    )  # Stop if unbounded
    return _iteration_needed[0]


def _simplex(
    A: jax.Array,
    b: jax.Array,
    c: jax.Array,
    basis: jax.Array,
    n_basis: jax.Array,
    feasible: jax.Array = jnp.array([1]).astype(bool).reshape(1),
) -> SimplexStep:
    def _simplex_step(step: SimplexStep) -> SimplexStep:
        # perform ratio test (Bland's rule)
        # Get index of first negative element in step.reduced_cost_muls
        neg_cost_mul_idx = jnp.where(
            step.reduced_cost_muls < -1e-5,
            size=step.reduced_cost_muls.size,
            fill_value=step.reduced_cost_muls.size,
        )[0]
        entering_idx = step.n_basis[jnp.min(neg_cost_mul_idx)]
        exiting_col = A[:, entering_idx]
        exiting_rates = step.B_inv @ exiting_col

        b_hat = jnp.maximum(
            step.b_hat, jnp.zeros_like(step.b_hat)
        )  # hack to make sure that fpe from matmul don't change basis selection
        div = jnp.divide(step.b_hat, exiting_rates)
        ratios = jax.lax.select(
            jnp.greater(exiting_rates, 1e-5), div, jnp.inf * jnp.ones_like(div)
        )  # Don't worry about constraints that entering var improves / doesn't affect
        exiting_idx = step.basis[fuzzy_argmin(ratios)]

        # Update basis set
        basis = step.basis.at[jnp.nonzero(step.basis == exiting_idx, size=1)].set(
            entering_idx
        )
        n_basis = step.n_basis.at[
            jnp.nonzero(step.n_basis == entering_idx, size=1)
        ].set(exiting_idx)

        # Find Solution
        B = A[:, basis]
        B_inv = jnp.linalg.inv(B)

        b_hat = B_inv @ b
        x = jnp.zeros(A.shape[1])
        x = x.at[basis].set(b_hat)

        N = A[:, n_basis]
        simplex_muls = c[basis] @ B_inv
        reduced_cost_muls = c[n_basis] - simplex_muls @ N

        dependent_constraints = jnp.any(jnp.isnan(B_inv))

        return SimplexStep(
            jnp.logical_and(step.feasible, jnp.logical_not(dependent_constraints)),
            jnp.less(exiting_rates, -1e-5).all().reshape(1),
            x,
            basis,
            n_basis,
            B_inv,
            b_hat,
            simplex_muls,
            reduced_cost_muls,
        )

    # Find Solution
    B = A[:, basis]
    B_inv = jnp.linalg.inv(B)
    b_hat = jnp.linalg.solve(B, b)

    b_hat = B_inv @ b
    x = jnp.zeros(A.shape[1])
    x = x.at[basis].set(b_hat)

    # Check if solution is optimal
    N = A[:, n_basis]
    simplex_muls = c[basis] @ B_inv
    reduced_cost_muls = c[n_basis] - simplex_muls @ N

    dependent_constraints = jnp.any(jnp.isnan(B_inv))

    val = SimplexStep(
        jnp.logical_and(feasible, jnp.logical_not(dependent_constraints)),
        jnp.array([0]).astype(bool),
        x,
        basis,
        n_basis,
        B_inv,
        b_hat,
        simplex_muls,
        reduced_cost_muls,
    )
    val = jax.lax.while_loop(_iteration_needed, _simplex_step, val)

    # Uncomment to debug
    # while _iteration_needed(val):
    #     val = _simplex_step(val)

    return val


@partial(jax.jit, static_argnames=["unbounded"])
def linprog(
    obj: jax.Array,
    A_eq: jax.Array = jnp.empty((0, 0)),
    b_eq: jax.Array = jnp.empty((0,)),
    A_ub: jax.Array = jnp.empty((0, 0)),
    b_ub: jax.Array = jnp.empty((0,)),
    unbounded: bool = False,
) -> SimplexStep:
    """
    Solves a linear program of the form: min c @ x s.t. A_eq @ x = b_eq, A_ub @ x <= b_ub

    Args:
        obj: The coefficients of the linear function to minimize
        A_eq: Equality constraint matrix
        b_eq: Equality constraint vector
        A_ub: Inequality constraint matrix
        b_ub: Inequality constraint vector
        unbounded: If False (default), only considers x >= 0. If True, will consider all x

    Returns:
        The vector x that minimizes c @ x subject to the constraints given.
    """
    A, b, c = _standard_form(obj, A_eq, b_eq, A_ub, b_ub, unbounded)

    # _simplex assumes that the last A.shape[0] variables form a feasible basis for the problem
    # This is not true in general (e.g. for problems with lots of equality constraints)
    # Therefore, we first solve a problem with auxiliary variables to find a feasible basis
    A_aux = jnp.hstack((A, jnp.eye(A.shape[0])))
    c_aux = jnp.concatenate((jnp.zeros_like(c), jnp.ones(A.shape[0])))
    n_basis, basis = jnp.split(
        jnp.arange(A_aux.shape[1]), (A_aux.shape[1] - A_aux.shape[0],)
    )
    aux_sol = _simplex(A_aux, b, c_aux, basis, n_basis)
    aux_sol.feasible = jnp.equal(c_aux @ aux_sol.x, 0).reshape(1)

    sol = _simplex(A, b, c, aux_sol.basis, aux_sol.n_basis, aux_sol.feasible)

    # Remove synthetic variables from returned result
    if unbounded:
        real_pos, real_neg, _ = jnp.split(sol.x, (len(obj), 2 * len(obj)))
        sol.x = real_pos - real_neg
    else:
        sol.x, _ = jnp.split(sol.x, (len(obj),))

    return sol


if __name__ == "__main__":

    def log(sol: SimplexStep):
        if not sol.feasible:
            print("Problem is infeasible")
        elif sol.unbounded:
            print("Problem is unbounded")
        else:
            print(sol.x)

    A = jnp.array(
        [
            [1, -1],
            [3, 2],
            [1, 0],
            [-2, 3],
        ]
    )
    b = jnp.array([1, 12, 2, 9])
    c = jnp.array([-4, -2])

    log(linprog(c, A_ub=A, b_ub=b))

    A = jnp.array(
        [
            [1],
        ]
    )
    b = jnp.array([10])
    c = jnp.array([1])

    log(linprog(c, A_ub=A, b_ub=b, unbounded=True))

    A = jnp.array(
        [
            [-1],
        ]
    )
    b = jnp.array([-10])
    c = jnp.array([-1])

    log(linprog(c, A_ub=A, b_ub=b))

    A = jnp.array(
        [
            [1, -1],
            [-1, 1],
        ]
    )
    b = jnp.array([1, -2])
    c = jnp.array([-4, -2])

    log(linprog(c, A_ub=A, b_ub=b))

    A = jnp.array(
        [
            [1.0, 1, -1, -1, 0, 0, 0, 0],
            [1, 0, -1, 0, 1, 0, 0, 0],
            [0, 1, 0, -1, 0, 1, 0, 0],
            [1, 0, -1, 0, 0, 0, -1, 0],
            [0, -1, 0, 1, 0, 0, 0, 1],
        ]
    )
    b = jnp.array([1.2, 1.1, 0.1, 0.9, 0.1])
    c = jnp.array([1.0, 0, -1, 0, 0, 0, 0, 0])

    sol = opt.linprog(c, A_eq=A, b_eq=b)
    print(sol.x)

    log(linprog(c, A_eq=A, b_eq=b))

    A_eq = jnp.array([[0.90096885, 0.43388376]])
    b_eq = jnp.array([0.7674836])
    A_ub = jnp.array([[1.0, 0], [0, 1], [-1, 0], [0, -1]])
    b_ub = jnp.array([0.9, 0.1, -0.9, 0.1])
    c = jnp.array([0.0, 1])

    log(linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, unbounded=True))
