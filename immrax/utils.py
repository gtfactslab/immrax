import time
from functools import partial
from itertools import product
from math import exp, floor, log
from typing import Callable, Literal, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import shapely.geometry as sg
import shapely.ops as so
from jax._src.traceback_util import api_boundary
from jax._src.util import wraps
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pypoman import plot_polygon
from scipy.spatial import HalfspaceIntersection

import immrax as irx
from immrax.inclusion import Corner, Interval, all_corners, i2lu, i2ut, interval, ut2i
from immrax.system import Trajectory

# ================================================================================
# Function wrappers
# ================================================================================


def timed(f: Callable):
    @wraps(f)
    @api_boundary
    def f_timed(*args, **kwargs):
        t0 = time.time()
        ret = jax.block_until_ready(f(*args, **kwargs))
        tf = time.time()
        return ret, (tf - t0)

    return f_timed


def run_times(N: int, f: Callable, *args, **kwargs):
    f_timed = timed(f)
    times = []
    for i in range(N):
        ret, dt = f_timed(*args, **kwargs)
        times.append(dt)
    return ret, jnp.array(times)


# ================================================================================
# Plotting
# ================================================================================

sg_box = lambda x, xi=0, yi=1: sg.box(
    x[xi].lower, x[yi].lower, x[xi].upper, x[yi].upper
)
sg_boxes = lambda xx, xi=0, yi=1: [sg_box(x, xi, yi) for x in xx]


def draw_sg_union(ax, boxes, **kwargs):
    shape = so.unary_union(boxes)
    xs, ys = shape.exterior.xy
    kwargs.setdefault("ec", "tab:blue")
    kwargs.setdefault("fc", "none")
    kwargs.setdefault("lw", 2)
    kwargs.setdefault("alpha", 1)
    ax.fill(xs, ys, **kwargs)


def draw_iarray(ax, x, xi=0, yi=1, **kwargs):
    return draw_sg_union(ax, [sg_box(x, xi, yi)], **kwargs)


def draw_iarrays(ax, xx, xi=0, yi=1, **kwargs):
    return draw_sg_union(ax, sg_boxes(xx, xi, yi), **kwargs)


def draw_iarray_3d(ax, x, xi=0, yi=1, zi=2, **kwargs):
    Xl, Yl, Zl = x.lower[(xi, yi, zi),]
    Xu, Yu, Zu = x.upper[(xi, yi, zi),]
    poly_alpha = kwargs.pop("poly_alpha", 0.0)
    kwargs.setdefault("color", "tab:blue")
    kwargs.setdefault("lw", 0.75)
    faces = [
        onp.array(
            [[Xl, Yl, Zl], [Xu, Yl, Zl], [Xu, Yu, Zl], [Xl, Yu, Zl], [Xl, Yl, Zl]]
        ),
        onp.array(
            [[Xl, Yl, Zu], [Xu, Yl, Zu], [Xu, Yu, Zu], [Xl, Yu, Zu], [Xl, Yl, Zu]]
        ),
        onp.array(
            [[Xl, Yl, Zl], [Xu, Yl, Zl], [Xu, Yl, Zu], [Xl, Yl, Zu], [Xl, Yl, Zl]]
        ),
        onp.array(
            [[Xl, Yu, Zl], [Xu, Yu, Zl], [Xu, Yu, Zu], [Xl, Yu, Zu], [Xl, Yu, Zl]]
        ),
        onp.array(
            [[Xl, Yl, Zl], [Xl, Yu, Zl], [Xl, Yu, Zu], [Xl, Yl, Zu], [Xl, Yl, Zl]]
        ),
        onp.array(
            [[Xu, Yl, Zl], [Xu, Yu, Zl], [Xu, Yu, Zu], [Xu, Yl, Zu], [Xu, Yl, Zl]]
        ),
    ]
    for face in faces:
        ax.plot3D(face[:, 0], face[:, 1], face[:, 2], **kwargs)
        kwargs["alpha"] = poly_alpha
        ax.add_collection3d(Poly3DCollection([face], **kwargs))


def draw_iarrays_3d(ax, xx, xi=0, yi=1, zi=2, color="tab:blue"):
    for x in xx:
        draw_iarray_3d(ax, x, xi, yi, zi, color)


def plot_interval_t(ax, tt, x, **kwargs):
    xl, xu = i2lu(x)
    alpha = kwargs.pop("alpha", 0.25)
    label = kwargs.pop("label", None)
    ax.fill_between(tt, xl, xu, alpha=alpha, label=label, **kwargs)
    ax.plot(tt, xl, **kwargs)
    ax.plot(tt, xu, **kwargs)


def draw_trajectory_2d(traj: Trajectory, vars=(0, 1), **kwargs):
    n = traj.ys[0].shape[0] // 2
    y_int = [
        irx.ut2i(jnp.array([y[vars[0]], y[vars[1]], y[vars[0] + n], y[vars[1] + n]]))
        for y in traj.ys
    ]  # TODO: fix indexing
    alpha = kwargs.pop("alpha", 0.4)
    label = kwargs.pop("label", None)
    for bound in y_int:
        draw_iarray(plt.gca(), bound, alpha=alpha, label=label, **kwargs)
        label = "_nolegend_"  # Only label the first plot


def draw_refined_trajectory_2d(traj: Trajectory, H: jnp.ndarray, vars=(0, 1), **kwargs):
    ys_int = [irx.ut2i(y) for y in traj.ys]
    color = kwargs.pop("color", "tab:blue")
    for bound in ys_int:
        dx = 1e-3 * jnp.ones_like(bound.lower)
        cons = onp.hstack(
            (
                onp.vstack((-H, H)),
                onp.concatenate((bound.lower - dx, -bound.upper - dx)).reshape(-1, 1),
            )
        )
        hs = HalfspaceIntersection(cons, bound.center[0 : H.shape[1]])
        # try:
        #     hs = HalfspaceIntersection(cons, bound.center[0 : H.shape[1]])
        # except Exception:
        #     x = bound.center[0 : H.shape[1]]
        #     print(bound.lower[0 : H.shape[1]], H @ x, bound.upper[0 : H.shape[1]])

        vertices = hs.intersections[:, 0:2]
        vertices = onp.vstack(
            (hs.intersections[:, vars[0]], hs.intersections[:, vars[1]])
        ).T

        plot_polygon(vertices, fill=False, resize=True, color=color, **kwargs)


def get_half_intervals(x: Interval, N=1, ut=False):
    _xx_0 = i2ut(x) if ut is False else x
    n = len(_xx_0) // 2
    ret = [_xx_0]
    for i in range(N):
        newret = []
        for _xx_ in ret:
            cent = (_xx_[:n] + _xx_[n:]) / 2
            for part_i in range(2**n):
                part = jnp.copy(_xx_)
                for ind in range(n):
                    part = part.at[ind + n * ((part_i >> ind) % 2)].set(cent[ind])
                newret.append(part)
        ret = newret
    if ut:
        return ret
    else:
        return [ut2i(part) for part in ret]


# ================================================================================
# Math
# ================================================================================


# @partial(jax.jit,static_argnums=(1,))
def get_partitions_ut(x: jax.Array, N: int) -> jax.Array:
    n = len(x) // 2
    # c^n = N
    c = floor(exp(log(N) / n) + 1e-10)
    _x = x[:n]
    x_ = x[n:]
    xc = []
    for i in range(c + 1):
        xc.append(_x + i * (x_ - _x) / c)
    l = onp.arange(c)
    A = onp.array(onp.meshgrid(*[l for i in range(n)])).reshape((n, -1)).T
    ret = []
    for i in range(len(A)):
        _part = jnp.array([xc[A[i, j]][j] for j in range(n)])
        part_ = jnp.array([xc[A[i, j] + 1][j] for j in range(n)])
        ret.append(jnp.concatenate((_part, part_)))
    return jnp.array(ret)


def gen_ics(x0, N, key=jax.random.key(0)):
    # X = np.empty((N, len(x0)))
    X = []
    keys = jax.random.split(key, len(x0))
    for i in range(len(x0)):
        # X[:,i] = uniform_disjoint(range, N)
        X.append(
            jax.random.uniform(
                key=keys[i], shape=(N,), minval=x0.lower[i], maxval=x0.upper[i]
            )
        )
    return jnp.array(X).T


def set_columns_from_corner(corner: Corner, A: Interval):
    _Jx = jnp.where(jnp.asarray(corner) == 0, A.lower, A.upper)
    J_x = jnp.where(jnp.asarray(corner) == 0, A.upper, A.lower)
    return _Jx, J_x


def get_corners(x: Interval, corners: Tuple[Corner] | None = None):
    """Gets the specified corners of the interval x. Returns all corners if None."""
    corners = all_corners(len(x)) if corners is None else corners
    xut = i2ut(x)
    return jnp.array(
        [
            jnp.array([x.lower[i] if c[i] == 0 else x.upper[i] for i in range(len(x))])
            for c in corners
        ]
    )


def get_sparse_corners(x: Interval, verbose=False, **kwargs):
    """Returns a function returning the sparse corners of the interval

    Parameters
    ----------
    x : Interval
        Interval object to model the gsc off of---value of x should be static
    **kwargs : dict
        Additional keyword arguments to pass to jnp.isclose

    Returns
    -------
    function
        A function that takes an Interval object and returns the sparse corners based on
        the entries that are not constant in the test x
    """
    sh = x.shape

    # Static value usage here.
    ic = onp.isclose(x.lower.reshape(-1), x.upper.reshape(-1), **kwargs)
    cs = [
        irx.Corner(p)
        for p in product(*[(0,) if ic[i] else (0, 1) for i in range(len(ic))])
    ]
    if verbose:
        print(
            f"Found {len(cs)} corners, from {jnp.sum(jnp.logical_not(ic))} nonconstant entries."
        )

    @jax.jit
    def gsc(x: Interval):
        x = x.reshape(-1)
        return [
            jnp.array(
                [x.lower[i] if ci == 0 else x.upper[i] for i, ci in enumerate(c)]
            ).reshape(sh)
            for c in cs
        ]

    return gsc


@api_boundary
@partial(jax.jit, static_argnums=(1,))
def get_rohn_corners(A: Interval, sign: Literal["+", "-"] = "+"):
    """Gets the 2^n corners of [A] which upper or lower bound x^T A x depending on the chosen sign (+/-)"""
    if A.shape[0] != A.shape[1] or len(A.shape) != 2:
        raise Exception(f"A should be a square matrix, got {A.shape}")
    n = A.shape[0]
    Ac = A.center
    Ap = A.pert

    if sign == "+":
        return jnp.asarray(
            [
                Ac + jnp.diag(jnp.asarray(s)) @ Ap @ jnp.diag(jnp.asarray(s))
                for s in product(*[[-1, +1] for i in range(n)])
            ]
        )
    elif sign == "-":
        return jnp.asarray(
            [
                Ac - jnp.diag(jnp.asarray(s)) @ Ap @ jnp.diag(jnp.asarray(s))
                for s in product(*[[-1, +1] for i in range(n)])
            ]
        )
    else:
        raise Exception("pm should be '+' or '-'.")


def null_space(A, rcond=None, dim_null: int | None = None):
    """Taken from scipy, with some modifications to use jax.numpy"""
    u, s, vh = jnp.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = jnp.finfo(s.dtype).eps * max(M, N)
    tol = jnp.amax(s) * rcond
    num = jnp.sum(s > tol, dtype=int) if dim_null is None else len(s) - dim_null + 1
    # num = jnp.sum(s > tol, dtype=int)
    # print(num)
    Q = vh[num:, :].T.conj()

    return Q


def angular_sweep(N: int):
    """
    Returns an array of points on the unit circle, evenly spaced in angle, which is in [0, pi]
    Both 0 and pi are excluded.

    Args:
        N: The number of points to generate

    Returns:
        jnp.array of points
    """
    return jnp.array(
        [
            [jnp.cos(n * jnp.pi / (N + 1)), jnp.sin(n * jnp.pi / (N + 1))]
            for n in range(1, N + 1)
        ]
    )


def check_containment(x, y):
    """Checks if the interval x is contained in the interval y.

    Uses interval-aware comparison to determine the containment relation:
    - Compares x against y using the new interval comparison operators
    - x is contained in y if x.lower >= y.lower AND x.upper <= y.upper
    - x is outside y if x.lower > y.upper OR x.upper < y.lower

    Returns
    -------
    int
        1 if x is fully contained in y
        -1 if x is fully outside of y
        0 if x is partially contained in y
    """
    from immrax.comparison import IntervalRelation, interval_compare

    x = interval(x)
    y = interval(y)

    # Use interval_compare to get Allen's relation
    relation = interval_compare(x, y)

    # Check containment element-wise
    is_contained = relation.matches(IntervalRelation.SUBSET)
    is_outside = relation.matches(IntervalRelation.DISJOINT)

    fully_contained = jnp.all(is_contained).astype(int)
    fully_outside = jnp.any(is_outside).astype(int)

    return fully_contained - fully_outside


def d_metzler(A):
    diag = jnp.diag_indices_from(A)
    Am = jnp.clip(A, 0, jnp.inf).at[diag].set(A[diag])
    return Am, A - Am


def d_positive(B):
    return jnp.clip(B, 0, jnp.inf), jnp.clip(B, -jnp.inf, 0)
