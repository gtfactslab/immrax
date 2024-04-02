import jax
import jax.numpy as jnp
import time
from jax._src.util import wraps
from jax._src.traceback_util import api_boundary
from immrax.inclusion import Interval, Corner, all_corners
from immrax.inclusion import i2ut, i2lu, i2centpert, ut2i, icentpert, icopy, interval
from typing import Callable, List, Tuple
import shapely.geometry as sg
import shapely.ops as so
import numpy as onp
from math import floor, exp, log
from functools import partial

def timed (f:Callable) :
    @wraps(f)
    @api_boundary
    def f_timed (*args, **kwargs) :
        t0 = time.time()
        ret = jax.block_until_ready(f(*args, **kwargs))
        tf = time.time()
        return ret, (tf - t0)
    return f_timed

def run_times (N:int, f:Callable, *args, **kwargs) :
    f_timed = timed(f)
    times = []
    for i in range(N) :
        ret, dt = f_timed(*args, **kwargs)
        times.append(dt)
    return ret, jnp.array(times)

def d_metzler (A) :
    diag = jnp.diag_indices_from(A)
    Am = jnp.clip(A, 0, jnp.inf); Am = Am.at[diag].set(A[diag])
    return Am, A - Am

def d_positive (B) :
    return jnp.clip(B, 0, jnp.inf), jnp.clip(B, -jnp.inf, 0)

sg_box = lambda x, xi=0, yi=1 : sg.box(x[xi].lower,x[yi].lower,x[xi].upper,x[yi].upper)
sg_boxes = lambda xx, xi=0, yi=1 : [sg_box(x, xi, yi) for x in xx]

def draw_sg_union (ax, boxes, **kwargs) :
    shape = so.unary_union(boxes)
    xs, ys = shape.exterior.xy
    kwargs.setdefault('ec', 'tab:blue')
    kwargs.setdefault('fc', 'none')
    kwargs.setdefault('lw', 2)
    kwargs.setdefault('alpha', 1)
    ax.fill(xs, ys, **kwargs)

draw_iarray = lambda ax, x, xi=0, yi=1, **kwargs : draw_sg_union(ax, [sg_box(x, xi, yi)], **kwargs)
draw_iarrays = lambda ax, xx, xi=0, yi=1, **kwargs: draw_sg_union(ax, sg_boxes(xx, xi, yi), **kwargs)

def plot_interval_t (ax, tt, x, **kwargs) :
    xl, xu = i2lu(x)
    alpha = kwargs.pop('alpha', 0.25)
    label = kwargs.pop('label', None)
    ax.fill_between(tt, xl, xu, alpha=alpha, label=label, **kwargs)
    ax.plot(tt, xl, **kwargs)
    ax.plot(tt, xu, **kwargs)


# @ijit
def get_half_intervals (x:Interval, N=1, ut=False) :
    _xx_0 = i2ut(x) if ut is False else x
    n = len(_xx_0) // 2
    ret = [_xx_0]
    for i in range (N) :
        newret = []
        for _xx_ in ret :
            cent = (_xx_[:n] + _xx_[n:])/2
            for part_i in range(2**n) :
                part = jnp.copy(_xx_)
                for ind in range (n) :
                    part = part.at[ind + n*((part_i >> ind) % 2)].set(cent[ind])
                newret.append(part)
        ret = newret
    if ut :
        return ret
    else :
        return [ut2i(part) for part in ret]

# @partial(jax.jit,static_argnums=(1,))
def get_partitions_ut (x:jax.Array, N:int) -> jax.Array :
    n = len(x) // 2
    # c^n = N
    c = floor(exp(log(N)/n) + 1e-10)
    _x = x[:n]; x_ = x[n:]
    xc = []
    for i in range(c+1) :
        xc.append(_x + i*(x_ - _x)/c)
    l = onp.arange(c)
    A = onp.array(onp.meshgrid(*[l for i in range(n)])).reshape((n,-1)).T
    ret = []
    for i in range(len(A)) :
        _part = jnp.array([xc[A[i,j]][j] for j in range(n)])
        part_ = jnp.array([xc[A[i,j]+1][j] for j in range(n)])
        ret.append(jnp.concatenate((_part,part_)))
    return jnp.array(ret)

def gen_ics (x0, N, key=jax.random.PRNGKey(0)) :
    # X = np.empty((N, len(x0)))
    X = []
    keys = jax.random.split(key, len(x0))
    for i in range(len(x0)) :
        # X[:,i] = uniform_disjoint(range, N)
        X.append(jax.random.uniform(key=keys[i],shape=(N,),minval=x0.lower[i], maxval=x0.upper[i]))
    return jnp.array(X).T

def set_columns_from_corner (corner:Corner, A:Interval) :
    # Set i-th column of A based on corner
    _Jx = jnp.empty_like(A.lower); J_x = jnp.empty_like(A.upper)
    for i in range(len(corner)) :
        if corner[i] == 0 :
            _Jx = _Jx.at[:,i].set(A.lower[:,i]) # Use _A when cornered on ub
            J_x = J_x.at[:,i].set(A.upper[:,i]) # Use A_ when cornered on lb
        else :
            _Jx = _Jx.at[:,i].set(A.upper[:,i]) # Use A_ when cornered on ub
            J_x = J_x.at[:,i].set(A.lower[:,i]) # Use _A when cornered on lb
    return _Jx, J_x

def get_corners (x:Interval, corners:Tuple[Corner]|None=None) :
    corners = all_corners(len(x)) if corners is None else corners
    xut = i2ut(x)
    return jnp.array([jnp.array([x.lower[i] if c[i] == 0 else x.upper[i] for i in range(len(x))]) for c in corners])

def I_refine (A:jax.Array, y:Interval) -> Interval :
    A = interval(A)
    def I_r (y) :
        ret = icopy(y)
        for j in range(len(A)) :
            for i in range(len(y)) :
                # print(f'{ret[i]=}')
                b1 = lambda : ((-A[j,:i] @ ret[:i] - A[j,i+1:] @ ret[i+1:])/A[j,i]) & ret[i]
                b2 = lambda : ret[i]
                reti = jax.lax.cond(jnp.abs(A[j,i].lower) > 1e-10, b1, b2)
                # print(f'{reti=}')
                retl = ret.lower.at[i].set(reti.lower)
                retu = ret.upper.at[i].set(reti.upper)
                ret = interval(retl, retu)
                # print(f'{ret=}')
        return ret
    return I_r(y)
def null_space(A, rcond=None):
    """Taken from scipy, with some modifications to use jax.numpy"""
    u, s, vh = jnp.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = jnp.finfo(s.dtype).eps * max(M, N)
    tol = jnp.amax(s) * rcond
    num = jnp.sum(s > tol, dtype=int)
    Q = vh[num:,:].T.conj()
    return Q