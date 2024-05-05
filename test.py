import jax
import jax.numpy as jnp
import immrax as irx
from functools import partial
from itertools import accumulate

n = 4; p = 2; q = 3
cumsum = tuple(accumulate((n, p, q)))
print(cumsum)

def arg2z (*args) :
    return jnp.concatenate(args)

def z2arg (z) :
    return jnp.split(z, cumsum[:-1])

x = jnp.arange(n)
u = jnp.arange(p)
w = jnp.arange(q)

ix = irx.icentpert(x, 0.1)
iu = irx.icentpert(u, 0.1)
iw = irx.icentpert(w, 0.1)

# print(irx.natif(arg2z)(ix, iu, iw))

z = arg2z(x,u,w)
iz = irx.icentpert(z, 0.1)
print(z2arg(z))
iix = irx.natif(z2arg)(iz)

print(irx.natif(z2arg)(iz))

# print(iix)

"""

def f (path) :
    A, x, w = path
    return A @ x

paths = [
    (jnp.array([[1., 2.], [3., 4.]]), jnp.array([1.,2.]), jnp.array(0.)),
    (jnp.array([[5., 6.], [7., 8.]]), jnp.array([1.,2.]), jnp.array(0.))
]

inner_structure = jax.tree_util.tree_structure(('*', '*', '*'))
outer_structure = jax.tree_util.tree_structure(('*', '*'))
paths = jax.tree_util.tree_transpose(outer_structure, inner_structure, paths)
paths = [jnp.array(v) for v in paths]
print(paths)

# paths = [
#     jnp.array([jnp.array([[1., 2.], [3., 4.]]), jnp.array([[5., 6.], [7., 8.]])]),
#     jnp.array([jnp.array([1.,2.]), jnp.array([1.,2.])]),
# ]
# print(paths)

print(jax.tree_util.tree_structure(paths))
print(jax.vmap(f, [[0, 0, 0]])(paths))
"""
