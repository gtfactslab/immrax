import jax
import jax.numpy as jnp
import numpy as onp
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
    Am = jnp.clip(A, 0, jnp.inf).at[diag].set(A[diag])
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

def draw_iarray_3d (ax, x, xi=0, yi=1, zi=2, **kwargs) :
    Xl, Yl, Zl = x.lower[(xi,yi,zi),]
    Xu, Yu, Zu = x.upper[(xi,yi,zi),]
    poly_alpha = kwargs.pop('poly_alpha', 0.)
    kwargs.setdefault('color', 'tab:blue')
    kwargs.setdefault('lw', 0.75)
    faces = [ \
        onp.array([[Xl,Yl,Zl],[Xu,Yl,Zl],[Xu,Yu,Zl],[Xl,Yu,Zl],[Xl,Yl,Zl]]), \
        onp.array([[Xl,Yl,Zu],[Xu,Yl,Zu],[Xu,Yu,Zu],[Xl,Yu,Zu],[Xl,Yl,Zu]]), \
        onp.array([[Xl,Yl,Zl],[Xu,Yl,Zl],[Xu,Yl,Zu],[Xl,Yl,Zu],[Xl,Yl,Zl]]), \
        onp.array([[Xl,Yu,Zl],[Xu,Yu,Zl],[Xu,Yu,Zu],[Xl,Yu,Zu],[Xl,Yu,Zl]]), \
        onp.array([[Xl,Yl,Zl],[Xl,Yu,Zl],[Xl,Yu,Zu],[Xl,Yl,Zu],[Xl,Yl,Zl]]), \
        onp.array([[Xu,Yl,Zl],[Xu,Yu,Zl],[Xu,Yu,Zu],[Xu,Yl,Zu],[Xu,Yl,Zl]]) ]
    for face in faces :
        ax.plot3D(face[:,0], face[:,1], face[:,2], **kwargs)
        kwargs['alpha'] = poly_alpha
        ax.add_collection3d(Poly3DCollection([face], **kwargs))

def draw_iarrays_3d (ax, xx, xi=0, yi=1, zi=2, color='tab:blue') :
    for x in xx :
        draw_iarray_3d(ax, x, xi, yi, zi, color)

def plot_interval_t (ax, tt, x, **kwargs) :
    xl, xu = i2lu(x)
    alpha = kwargs.pop('alpha', 0.25)
    label = kwargs.pop('label', None)
    ax.fill_between(tt, xl, xu, alpha=alpha, label=label, **kwargs)
    ax.plot(tt, xl, **kwargs)
    ax.plot(tt, xu, **kwargs)


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

def gen_ics (x0, N, key=jax.random.key(0)) :
    # X = np.empty((N, len(x0)))
    X = []
    keys = jax.random.split(key, len(x0))
    for i in range(len(x0)) :
        # X[:,i] = uniform_disjoint(range, N)
        X.append(jax.random.uniform(key=keys[i],shape=(N,),minval=x0.lower[i], maxval=x0.upper[i]))
    return jnp.array(X).T

def set_columns_from_corner(corner:Corner, A:Interval):
    _Jx = jnp.where(jnp.asarray(corner) == 0, A.lower, A.upper)
    J_x = jnp.where(jnp.asarray(corner) == 0, A.upper, A.lower)
    return _Jx, J_x

def get_corners (x:Interval, corners:Tuple[Corner]|None=None) :
    corners = all_corners(len(x)) if corners is None else corners
    xut = i2ut(x)
    return jnp.array([jnp.array([x.lower[i] if c[i] == 0 else x.upper[i] for i in range(len(x))]) for c in corners])

def I_refine (A:jax.Array) -> Callable[[Interval], Interval]:
    def vec_refine(null_vector: jax.Array, var_index: int, y:Interval):
        ret = icopy(y)

        # Set up linear algebra computations for the refinement
        bounding_vars = interval(null_vector.at[var_index].set(0))
        ref_var = interval(null_vector[var_index])
        b1 = lambda: ((-bounding_vars @ null_vector) / ref_var) & ret[var_index]
        b2 = lambda: ret[var_index]

        # Compute refinement based on null vector, if possible 
        ndb0 = (jnp.abs(null_vector[var_index]) > 1e-10)
        ret = jax.lax.cond(ndb0, b1, b2) 

        # fix fpe problem with upper < lower
        retu = jnp.where(ret.upper >= ret.lower, ret.upper, ret.lower)
        return interval(ret.lower, retu)

    mat_refine = jax.vmap(vec_refine, in_axes=(0, None, None), out_axes=0)
    mat_refine_all = jax.vmap(mat_refine, in_axes=(None, 0, None), out_axes=1)

    def best_refinement(y:Interval):
        refinements = mat_refine_all(A, jnp.arange(len(y)), y)
        return interval(jnp.max(refinements.lower, axis=0), jnp.min(refinements.upper, axis=0))
    
    return best_refinement

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
