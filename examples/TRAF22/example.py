import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float, Integer
import immrax as irx
from functools import partial
from TRAF22 import *
import matplotlib.pyplot as plt

sys = TRAF22()
clembsys = TRAFClosedEmbedding(sys, 'BEL_Putte-4_2_T-1_controls.csv')
print(clembsys.KK)

# w = irx.icentpert(jnp.zeros(2), jnp.array([0.02, 0.3]))
# w = irx.icentpert(jnp.zeros(2), jnp.array([0.00001, 0.00001]))
w = irx.icentpert(jnp.zeros(2), jnp.array([0.00001, 0.00001]))
# w = irx.izeros(2)
# z = irx.icentpert(jnp.zeros(5), jnp.array([0.0004, 0.0004, 0.006, 0.002, 0.002]))
z = irx.izeros(5)

def wmap (t, x) :
    return w
def zmap (t, x) :
    return z

clembsys.jit_compile(wmap, zmap)
print('Finished setup and compilation.')

traj = clembsys.run_scenario(wmap, zmap)
tfinite = jnp.isfinite(traj.ts)
print(traj.ys[tfinite][0])
print(traj.ys[tfinite][1])
print(traj.ys[tfinite][20])
print(traj.ys[tfinite][100])

tt = traj.ts[tfinite]
xuts = traj.ys[tfinite][:,:10]
xint = irx.interval(xuts[:,:5], xuts[:,5:])
print(xint.shape)
print(xint[:,0].shape)
xnom = traj.ys[tfinite][:,10:]

fig, ax = plt.subplots(1,3,figsize=(10,3), dpi=100)
ax[0].plot(xnom[:,3], xnom[:,4], 'k-', label='Nominal')
ax[1].plot(tt, xnom[:,3], 'k-', label='Nominal')
ax[2].plot(tt, xnom[:,4], 'k-', label='Nominal')
irx.utils.plot_interval_t(ax[1], tt, xint[:,3])
irx.utils.plot_interval_t(ax[2], tt, xint[:,4])

plt.show()