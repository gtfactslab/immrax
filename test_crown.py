import jax
import jax.lax as lax
import jax.numpy as jnp
import immrax as irx
import equinox as eqx
import equinox.nn as nn
from collections import namedtuple
from typing import Union
from immrax import Interval, interval

"""
Use scan to implement the linear bound propogation on the neural network.
"""

class CROWNResult (namedtuple('CROWNResult', ['lC', 'uC', 'ld', 'ud'])) :
    def __call__(self, x:Union[jax.Array, Interval]) -> Interval:
        if isinstance(x, Interval) :
            lCp = jnp.clip(self.lC, 0, jnp.inf)
            lCn = jnp.clip(self.lC, -jnp.inf, 0)
            uCp = jnp.clip(self.uC, 0, jnp.inf)
            uCn = jnp.clip(self.uC, -jnp.inf, 0)
            return interval(
                lCp@x.lower + lCn@x.upper + self.ld,
                uCn@x.lower + uCp@x.upper + self.ud
            )
        elif isinstance(x, jax.Array) :
            return interval(self.lC@x + self.ld, self.uC@x + self.ud)


# net = lambda x : jax.nn.sigmoid(W @ x + b)
net = irx.NeuralNetwork('examples/vehicle/100r100r2')
# net = irx.NeuralNetwork('testnet', False)

ix = irx.icentpert(jnp.ones(4), 0.5)

def _relu_linprop (l, u) :
    # branch = jnp.array(l < 0, "int32") + 2*jnp.array(u > 0, "int32")
    # case0 = lambda : (0., 0., -jnp.inf, jnp.inf) # l > 0, u < 0, Impossible
    # case1 = lambda : (0., 0., 0., 0.) # l < 0, u < 0, ReLU is inactive
    # case2 = lambda : (1., 1., 0., 0.) # l < 0, u > 0, ReLU is active
    # ola = u/(u - l)
    # case3 = lambda : (ola, 0., -ola*l, 0.) # l < 0, u > 0, ReLU is active. 
    # return lax.switch (branch, (case0, case1, case2, case3))
    on = l >= 0.
    active = jnp.logical_and(l < 0., u >= 0.)
    ua = on.astype(jnp.float32) + active.astype(jnp.float32) * (u / (u - l))
    ub = active.astype(jnp.float32) * (-ua * l)
    la = (ua >= 0.5).astype(jnp.float32)
    lb = jnp.zeros_like(ub)
    return la, ua, lb, ub

print(_relu_linprop(jnp.array([-1., -1., 0.5]), jnp.array([1., -0.5, 1.])))

@jax.jit
def newcrown (ix) :
    lx = [ix.lower]
    ux = [ix.upper]
    
    # # IBP forward pass

    # for i, layer in enumerate(net.seq) :
    #     if isinstance (layer, nn.Linear) :
    #         # lux = irx.natif(layer)(interval(lx[i], ux[i]))
    #         # lx.append(lux.lower)
    #         # ux.append(lux.upper)
    #         Wp = jnp.clip(layer.weight, 0, jnp.inf); Wn = jnp.clip(layer.weight, -jnp.inf, 0)
    #         lx.append(Wp @ lx[i] + Wn @ ux[i] + layer.bias)
    #         ux.append(Wp @ ux[i] + Wn @ lx[i] + layer.bias)

    #     elif isinstance (layer, nn.Lambda) :
    #         if layer.fn == jax.nn.relu :
    #             lx.append(jnp.maximum(lx[i], 0))
    #             ux.append(jnp.maximum(ux[i], 0))

    # CROWN forward pass

    lC = jnp.eye(len(lx[0]))
    uC = jnp.eye(len(ux[0]))
    ld = jnp.zeros(len(lx[0]))
    ud = jnp.zeros(len(ux[0]))

    for i, layer in enumerate(net.seq) :
        if isinstance (layer, nn.Linear) :
            W = layer.weight
            b = layer.bias
            Wp, Wn = jnp.clip(W, 0, jnp.inf), jnp.clip(W, -jnp.inf, 0)
            _lC = Wp @ lC + Wn @ uC
            _uC = Wp @ uC + Wn @ lC
            _ld = Wp @ ld + Wn @ ud + b
            _ud = Wp @ ud + Wn @ ld + b
            _lCp = jnp.clip(_lC, 0, jnp.inf); _lCn = jnp.clip(_lC, -jnp.inf, 0)
            _uCp = jnp.clip(_uC, 0, jnp.inf); _uCn = jnp.clip(_uC, -jnp.inf, 0)
            # lx.append(jnp.maximum(_lCp @ ix.lower + _lCn @ ix.upper + _ld, Wp @ lx[i] + Wn @ ux[i] + b))
            # ux.append(jnp.minimum(_uCp @ ix.upper + _uCn @ ix.lower + _ud, Wp @ ux[i] + Wn @ lx[i] + b))
            lx.append(_lCp @ ix.lower + _lCn @ ix.upper + _ld)
            ux.append(_uCp @ ix.upper + _uCn @ ix.lower + _ud)
            lC, uC, ld, ud = _lC, _uC, _ld, _ud
        elif isinstance (layer, nn.Lambda) :
            if layer.fn == jax.nn.relu :
                lCp = jnp.clip(lC, 0, jnp.inf); lCn = jnp.clip(lC, -jnp.inf, 0)
                l_con = jnp.maximum(lCp @ ix.lower + lCn @ ix.upper + ld, lx[i])
                uCp = jnp.clip(uC, 0, jnp.inf); uCn = jnp.clip(uC, -jnp.inf, 0)
                u_con = jnp.minimum(uCp @ ix.upper + uCn @ ix.lower + ud, ux[i])
                la, ua, lb, ub = _relu_linprop(l_con, u_con)
                lC = (lC.T * la).T 
                uC = (uC.T * ua).T 
                ld = lb + la*ld
                ud = ub + ua*ud
                lx.append(jnp.minimum(lx[i], 0))
                ux.append(jnp.maximum(ux[i], 0))

    # CROWN backward pass

    lC = jnp.eye(len(lx[-1]))
    uC = jnp.eye(len(ux[-1]))
    ld = jnp.zeros(len(lx[-1]))
    ud = jnp.zeros(len(ux[-1]))

    for i in range(len(net.seq)-1, -1, -1) :
        layer = net.seq[i]
        if isinstance (layer, nn.Linear) :
            W = layer.weight
            b = layer.bias
            _lC = lC @ W
            _uC = uC @ W
            _ld = lC @ b + ld
            _ud = uC @ b + ud
            lC, uC, ld, ud = _lC, _uC, _ld, _ud
        
        elif isinstance (layer, nn.Lambda) :
            if layer.fn == jax.nn.relu :
                la, ua, lb, ub = _relu_linprop(lx[i], ux[i])
                lCp = jnp.clip(lC, 0, jnp.inf); lCn = jnp.clip(lC, -jnp.inf, 0)
                uCp = jnp.clip(uC, 0, jnp.inf); uCn = jnp.clip(uC, -jnp.inf, 0)
                _lC = lCp * la + lCn * ua
                _uC = uCp * ua + uCn * la
                _ld = lCp @ lb + lCn @ ub + ld
                _ud = uCp @ ub + uCn @ lb + ud
                lC, uC, ld, ud = _lC, _uC, _ld, _ud

    return CROWNResult(lC, uC, ld, ud)

oldcrown = jax.jit(irx.crown(net))

from time import time

t0 = time()
oldres = oldcrown(ix)
tf = time()
print(f'Finished jit for oldcrown in {tf - t0}')

t0 = time()
newres = newcrown(ix)
tf = time()
print(f'Finished jit for newcrown in {tf - t0}')


oldres, oldtimes = irx.utils.run_times(1000, oldcrown, ix)
newres, newtimes = irx.utils.run_times(1000, newcrown, ix)
print(f'old: {oldtimes.mean()} +/- {oldtimes.std()}')
print(f'new: {newtimes.mean()} +/- {newtimes.std()}')

print('old', oldres)
print('new', newres)

print(net(ix.upper))
print('old', oldres(ix.upper))
print('new', newres(ix.upper))
