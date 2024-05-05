import jax
import jax.numpy as jnp
import immrax as irx
from functools import partial

A = jnp.array([[1., 2.], [4., 5.]])
x = jnp.array([1.,2.])

def f (A, x) :
    return A @ x
fA = partial(f, A)
F = irx.natif(f)

class Dum (irx.System) :
    def __init__ (self) :
        self.xlen = 2
        self.evolution = 'continuous'
        self.fi = [
            lambda t, x, *args, **kwargs : x[0] + x[1],
            lambda t, x, *args, **kwargs : x[0] - x[1]
        ]

    def f (self, t, x, *args, **kwargs) :
        return jnp.array([self.fi[i](t, x) for i in range(sys.xlen)])

sys = Dum()

@jax.jit
def f1 (x) :
    # return sys.fi[0](0.,x)
    return x[0] + x[1]

f1(x)

print(f1.__dir__())
# print(jax.make_jaxpr(f1)(x))

FA = irx.natif(sys.f)
# EA = irx.ifemb(sys, FA)
embsys = irx.natemb(sys)

iA = irx.icentpert(A, 0.1)
ix = irx.icentpert(x, 0.1)

print(jax.make_jaxpr(sys.f)(0., x))
# print(jax.make_jaxpr(f1)(x))
# print(jax.make_jaxpr(embsys.F)(0., ix))
print(jax.make_jaxpr(embsys.Fi[0])(0., ix))
# print(jax.make_jaxpr(embsys.E)(0., irx.i2ut(ix)))

# print(jax.make_jaxpr(f)(A, x))
# print(jax.make_jaxpr(jax.jit(F))(iA, ix))
# print(jax.make_jaxpr(jax.jit(EA.E))(0., irx.i2ut(ix)))

"""

ix = irx.icentpert(jnp.array([1.,2.]), 0.1)
print(ix)

F = irx.natif(f)
F(iA, ix)
print('JIT compiled')

print(F(iA, ix))
"""
