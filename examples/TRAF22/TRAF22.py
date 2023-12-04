from typing import Union
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float, Integer
import immrax as irx
import csv
from functools import partial

class TRAF22 (irx.OpenLoopSystem) :
    lwb:float
    def __init__(self, lwb:float = 2.578) -> None :
        self.evolution = 'continuous'
        self.xlen = 5
        self.lwb = lwb
    
    def f(self, t:int|float, x:ArrayLike, u:ArrayLike, w:ArrayLike) -> jax.Array:
        δ, ψ, v, sx, sy = x.ravel()
        u1, u2 = u.ravel()
        w1, w2 = w.ravel()
        return jnp.array([
            u1 + w1,
            (v/self.lwb)*jnp.tan(δ),
            u2 + w2,
            v*jnp.cos(ψ),
            v*jnp.sin(ψ),
        ])

def trafemb (sys:TRAF22) -> irx.EmbeddingSystem :
    sys_mjacM = jax.jit(irx.mjacM(sys.f))

    @partial(jax.jit, static_argnums=(0,), static_argnames=['orderings'])
    def F(self, t:irx.Interval, x:irx.Interval, u:irx.Interval, w:irx.Interval, z:irx.Interval,
          K:irx.Interval, reference, orderings=None) -> jax.Array:
        Mt, Mx, Mu, Mw = sys_mjacM(t, x, u, w, orderings=orderings, centers=(reference,))
        tc, xc, uc, wc = reference
        return (
            # ([Jx] + [Ju]K)([\ulx,\olx] - x_nom)
            (Mx + Mu @ K) @ (x - xc) + K @ z
            # + [Ju](u - u_nom)
            + Mu @ (u - uc)
            # + [Jw]([\ulw,\olw] - w_nom)
            + Mw @ (w - wc)
            # + f(xc, uc, wc)
            + self.sys.f(tc, xc, uc, wc)
        )
    return irx.InclusionEmbedding(sys, F)

def load_scenario (file) :
    """Loads a TRAF22 scenario from a csv file.

    Returns:
        tt (jnp.ndarray): Array of time steps.
        uu (jnp.ndarray): Array of control inputs.
        KK (jnp.ndarray): Array of control gains.
    """
    with open(file) as csvfile :
        reader = csv.reader(csvfile, delimiter=';')
        tt = []
        uu = []
        KK = []
        for i, row in enumerate(reader) :
            if i == 0 :
                tt.append(float(row[0]))
                x0 = jnp.array([float(row[j]) for j in range(1,6)])
            else :
                tt.append(float(row[0]))
                uu.append([float(row[j]) for j in range(1,3)])
                KK.append(jnp.array([float(row[j]) for j in range(3,13)]).reshape(2,5))
        return jnp.asarray(tt), jnp.asarray(uu), jnp.asarray(KK)
    
class TRAFClosedSystem (irx.System) :
    sys:TRAF22
    embsys:irx.InclusionEmbedding
    tt:jnp.ndarray
    uu:jnp.ndarray
    KK:jnp.ndarray
    def __init__ (self, sys:TRAF22, embsys:irx.InclusionEmbedding, scenario) :
        self.sys = sys
        self.embsys = embsys
        self.xlen = sys + embsys
        self.evolution = sys.evolution
        self.tt, self.KK, self.uu = load_scenario(scenario)

    @partial(jax.jit, static_argnums=(0,))
    def f(self, t:int|float, x:ArrayLike, u:irx.Interavl, w:irx.Interval, z:irx.Interval) -> jax.Array :
        i = jnp.searchsorted(self.tt, t) - 1
        return super().f(t, x, *args, **kwargs)
