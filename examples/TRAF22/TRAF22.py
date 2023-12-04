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
        # return jnp.array([
        #     u[0] + w[0],
        #     (x[2]/self.lwb)*jnp.tan(x[0]),
        #     u[1] + w[1],
        #     x[2]*jnp.cos(x[1]),
        #     x[2]*jnp.sin(x[1]),
        # ])

def trafemb (sys:TRAF22) -> irx.EmbeddingSystem :
    sys_mjacM = jax.jit(irx.mjacM(sys.f))

    @partial(jax.jit, static_argnums=(0,), static_argnames=['orderings'])
    def F(t:irx.Interval, x:irx.Interval, u:irx.Interval, w:irx.Interval, z:irx.Interval,
          K:irx.Interval, reference, orderings=None) -> jax.Array:
        # K = K * 10.
        Mt, Mx, Mu, Mw = sys_mjacM(t, x, u, w, orderings=orderings, centers=(reference,))[0]
        tc, xc, uc, wc = reference
        return (
            # ([Jx] + [Ju]K)([\ulx,\olx] - x_nom)
            (Mx + Mu @ K) @ (x - xc) + Mu @ K @ z
            # + [Ju](u - u_nom)
            + Mu @ (u - uc)
            # + [Jw]([\ulw,\olw] - w_nom)
            + Mw @ (w - wc)
            # + f(xc, uc, wc)
            + sys.f(tc, xc, uc, wc)
        )
    return irx.InclusionEmbedding(sys, F)

def _load_scenario (file) :
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
        return x0, jnp.asarray(tt), jnp.asarray(uu), jnp.asarray(KK)
    
class TRAFClosedEmbedding (irx.System) :
    sys:TRAF22
    embsys:irx.InclusionEmbedding
    x0:jnp.ndarray
    tt:jnp.ndarray
    uu:jnp.ndarray
    KK:jnp.ndarray
    def __init__ (self, sys:TRAF22, scenario='BEL_Putte-4_2_T-1_controls.csv') :
        self.sys = sys
        self.embsys = trafemb(sys)
        self.xlen = sys.xlen + self.embsys.xlen
        self.evolution = sys.evolution
        self.load_scenario(scenario)

    def load_scenario (self, scenario='BEL_Putte-4_2_T-1_controls.csv') :
        self.x0, self.tt, self.uu, self.KK = _load_scenario(scenario)

    @partial(jax.jit, static_argnums=(0,))
    def f(self, t:int|float, x:ArrayLike, w:irx.Interval, z:irx.Interval) -> jax.Array :
        i = jnp.searchsorted(self.tt, t) - 1
        K = self.KK[i]
        u_ref = self.uu[i]

        x1, x2 = x[:self.embsys.xlen], x[self.embsys.xlen:]

        return jnp.concatenate((
            self.embsys.E(irx.interval(jnp.array([t])), x1, irx.interval(u_ref), 
                          w, z, irx.interval(K), 
                          reference=(jnp.array([t]), x2, u_ref, jnp.zeros(2))),
            self.sys.f(t, x2, u_ref, jnp.zeros(2)),
        ))

    def jit_compile (self, wmap, zmap, solver='tsit5') :
        return self.compute_trajectory(
            self.tt[0], self.tt[-1], jnp.zeros(self.xlen), 
            (wmap, zmap),
            0.01)

    def run_scenario (self, wmap, zmap, dt=0.01, solver='tsit5') :
        return self.compute_trajectory(
            self.tt[0], self.tt[-1], 
            jnp.concatenate((self.x0, irx.i2ut(zmap(0, self.x0) + self.x0))), 
            (wmap, zmap),
            dt)
