import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
import scipy.optimize as opt

from typing import Tuple
from functools import partial


@register_pytree_node_class
class SimplexStep:
    tableau: jax.Array
    basis: jax.Array
    n_basis: jax.Array

    def __init__(
        self,
        tableau,
        basis,
        x,
        feasible,
        unbounded,
    ):
        self.tableau = tableau
        self.basis = basis
        self.x = x
        self.feasible = feasible
        self.unbounded = unbounded

    def tree_flatten(self):
        return (
            (
                self.tableau,
                self.basis,
                self.x,
                self.feasible,
                self.unbounded,
            ),
            "SimplexStep",
        )

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)

    @property
    def success(self) -> jax.Array:
        return jnp.logical_and(self.feasible, jnp.logical_not(self.unbounded))

    @property
    def fun(self) -> jax.Array:
        return -self.tableau[-1, -1]


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
        step.tableau[-1, :-1], -1e-5
    ).any()  # Stop when optimal solution found
    _iteration_needed = jnp.logical_and(
        _iteration_needed, step.feasible
    )  # Stop if infeasible
    _iteration_needed = jnp.logical_and(
        _iteration_needed, jnp.logical_not(step.unbounded)
    )  # Stop if unbounded
    return _iteration_needed[0]


def _simplex(step: SimplexStep, num_cost_rows: int) -> SimplexStep:
    def pivot(step: SimplexStep) -> SimplexStep:
        tableau = step.tableau

        # Find entering variable (with Bland's rule)
        neg_cost_mul_idx = jnp.where(
            tableau[-1, :-1] < -1e-5,
            size=tableau.shape[1] - 1,
            fill_value=tableau.shape[1] - 1,
        )[0]
        entering_col = jnp.min(neg_cost_mul_idx)

        # Find exiting variable / pivot row
        exiting_rates = tableau[:-num_cost_rows, entering_col]
        div = jnp.divide(tableau[:-num_cost_rows, -1], exiting_rates)
        ratios = jax.lax.select(
            jnp.greater(exiting_rates, 1e-5), div, jnp.inf * jnp.ones_like(div)
        )  # Don't worry about constraints that entering var improves / doesn't affect
        exiting_row = fuzzy_argmin(ratios)
        unbounded = jnp.all(exiting_rates < -1e-5).reshape(1)

        # Pivot
        pivot_val = tableau[exiting_row, entering_col]
        tableau = tableau.at[exiting_row].set(
            tableau[exiting_row] / pivot_val
        )  # normalize pivot val to 1
        other_rows = jnp.setdiff1d(
            jnp.arange(tableau.shape[0]),
            exiting_row,
            assume_unique=True,
            size=tableau.shape[0] - 1,
        )
        for i in other_rows:
            tableau = tableau.at[i].set(
                tableau[i] - tableau[i, entering_col] * tableau[exiting_row]
            )

        # Update basis set, BFS
        basis = step.basis.at[exiting_row].set(entering_col)
        x = jnp.zeros_like(step.x)
        x = x.at[basis].set(tableau[:-num_cost_rows, -1])

        return SimplexStep(tableau, basis, x, step.feasible, unbounded)

    # NOTE: this looses reverse mode autodiff. We can accomplish all the same calculations
    # with forward mode, which this does not lose, but they might be slightly less efficient.
    step = jax.lax.while_loop(_iteration_needed, pivot, step)

    # Uncomment to debug
    # while _iteration_needed(step):
    #     step = pivot(step)

    return step


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
    tableau = jnp.hstack((A, jnp.eye(A.shape[0]), b.reshape(-1, 1)))
    c_extended = jnp.concatenate((c, jnp.zeros(A.shape[0] + 1)))
    c_aux = jnp.concatenate((jnp.zeros_like(c), jnp.ones(A.shape[0]), jnp.zeros(1)))
    tableau = jnp.vstack((tableau, c_extended, c_aux))

    # Zero out reduced cost muls of initial basis
    for i in range(A.shape[0]):
        tableau = tableau.at[-1].set(tableau[-1] - tableau[i])

    basis = jnp.arange(A.shape[1], A.shape[1] + A.shape[0])
    x = jnp.concatenate((jnp.zeros_like(c), b))
    aux_start = SimplexStep(tableau, basis, x, jnp.array([True]), jnp.array([False]))
    aux_sol = _simplex(aux_start, num_cost_rows=2)
    x = aux_sol.x[: c.size]

    # Remove auxiliary variables from tableau for real problem
    tableau = aux_sol.tableau[:-1, :]
    tableau = jnp.delete(
        tableau,
        jnp.arange(A.shape[1], A.shape[1] + A.shape[0]),
        axis=1,
        assume_unique_indices=True,
    )

    redundant_cons = jnp.concatenate((aux_sol.basis > A.shape[1], jnp.array([False])))
    redundant_cons = redundant_cons.repeat(tableau.shape[1]).reshape(tableau.shape)
    # TODO: I don't know if this is correct in general. Really, what I want to do is
    # make the column corresponding to the auxiliary variable basic, but I'm not sure
    # of an efficient way to identify this column
    tableau = jnp.where(redundant_cons, -tableau, tableau)

    real_start = SimplexStep(
        tableau,
        aux_sol.basis,
        aux_sol.x[: c.size],
        jnp.allclose(c_aux[:-1] @ aux_sol.x, jnp.array([0])).reshape(1),
        jnp.array([False]),
    )
    sol = _simplex(real_start, num_cost_rows=1)

    # Remove synthetic variables from returned result
    if unbounded:
        real_pos, real_neg, _ = jnp.split(sol.x, (len(obj), 2 * len(obj)))
        sol.x = real_pos - real_neg
    else:
        sol.x, _ = jnp.split(sol.x, (len(obj),))

    return sol


def compare(my_sol: SimplexStep, sp_sol: opt.OptimizeResult) -> Tuple[bool, str]:
    if sp_sol.status == 2:
        if not my_sol.feasible:
            return True, "SUCCESS: problem is infeasible"
        else:
            return False, "FAILURE: we did not detect problem as infeasible"
    elif sp_sol.status == 3:
        if my_sol.unbounded:
            return True, "SUCCESS: problem is unbounded"
        else:
            return False, "FAILURE: we did not detect problem as unbounded"
    elif sp_sol.status == 0:
        if my_sol.success:
            correct = jnp.allclose(my_sol.fun, sp_sol.fun)

            if correct:
                return True, f"SUCCESS: x={my_sol.x}"
            else:
                return False, "FAILURE: objective value does not match"
        else:
            if not my_sol.feasible:
                return True, "FAILURE: we incorrectly identified problem as infeasible"
            elif my_sol.unbounded:
                return False, "FAILURE: we incorrectly identified problem as unbounded"

    return False, "FAILURE: unknown status"


if __name__ == "__main__":
    import scipy.optimize as opt

    def verify(
        c: jax.Array,
        A_eq: jax.Array = jnp.empty((0, 0)),
        b_eq: jax.Array = jnp.empty((0,)),
        A_ub: jax.Array = jnp.empty((0, 0)),
        b_ub: jax.Array = jnp.empty((0,)),
        unbounded: bool = False,
    ):
        my_sol = linprog(c, A_eq, b_eq, A_ub, b_ub, unbounded)

        # The method output by the first phase of the simplex method is not actually feasible
        # I do not detect this problem in my implementation
        # Consider changing to just linear solver, extract non-zero indices

        bounds = (None, None) if unbounded else (0, None)
        sp_A_eq = A_eq if A_eq.size > 0 else None
        sp_b_eq = b_eq if b_eq.size > 0 else None
        sp_A_ub = A_ub if A_ub.size > 0 else None
        sp_b_ub = b_ub if b_ub.size > 0 else None
        sp_sol = opt.linprog(c, sp_A_ub, sp_b_ub, sp_A_eq, sp_b_eq, bounds=bounds)

        print(compare(my_sol, sp_sol))

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

    verify(c, A_ub=A, b_ub=b)

    A = jnp.array(
        [
            [1],
        ]
    )
    b = jnp.array([10])
    c = jnp.array([1])

    verify(c, A_ub=A, b_ub=b, unbounded=True)

    A = jnp.array(
        [
            [-1],
        ]
    )
    b = jnp.array([-10])
    c = jnp.array([-1])

    verify(c, A_ub=A, b_ub=b)

    A = jnp.array(
        [
            [1, -1],
            [-1, 1],
        ]
    )
    b = jnp.array([1, -2])
    c = jnp.array([-4, -2])

    verify(c, A_ub=A, b_ub=b)

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

    verify(c, A_eq=A, b_eq=b)

    ## Programs with dependent constraints

    A = jnp.array([[1.0, 0], [1, 0]])
    b = jnp.array([3.0, 3])
    c = jnp.array([-1.0, 0])

    verify(c, A_ub=A, b_ub=b)

    A_eq = jnp.array([[1.0, 2]])
    b_eq = jnp.array([1.0])
    A_ub = jnp.array([[1.0, 0], [0, 1], [-1, 0], [0, -1]])
    b_ub = jnp.array([0.9, 0.1, -0.9, 0.1])
    c = jnp.array([0.0, 1])

    verify(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, unbounded=True)

    A_eq = jnp.array([[0.5, 0.5]])
    b_eq = jnp.array([0.4])
    A_ub = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -0.0], [-0.0, -1.0]])
    b_ub = jnp.array([0.9, 0.1, -0.9, 0.1])
    c = -jnp.array([0.0, 1.0])

    verify(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, unbounded=True)
