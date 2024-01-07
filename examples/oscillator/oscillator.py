import sympy as sp
import immrax as irx
import jax.numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.patches import Wedge

def draw_angint (ax:plt.Axes, r:irx.Interval, th:irx.Interval, **kwargs) :
    thdeg = th * (180 / jnp.pi)
    return ax.add_patch(Wedge((0,0), r.upper, thdeg.lower, thdeg.upper, width=(r.upper - r.lower), **kwargs))

class oscillator (irx.System) :
    """A system of """
    def __init__ (self) :
        self.evolution = 'continuous'
        self.xlen = 2

    def f(self, t, x) :
        return jnp.array([
            -(x[0] - 1),
            1.
        ]) 

sys = oscillator ()
x0 = jnp.array([2.,0.]) 
traj = sys.compute_trajectory(0., 100., x0, dt=0.05)

embsys = irx.natemb(sys)
embx0 = irx.i2ut(irx.icentpert(x0, 0.01))
embtraj = embsys.compute_trajectory(0., 100., embx0, dt=0.05)

fig, ax = plt.subplots(1,1)

tfinite = jnp.isfinite(traj.ts)
ax.plot(traj.ys[tfinite,0]*jnp.cos(traj.ys[tfinite,1]), traj.ys[tfinite,0]*jnp.sin(traj.ys[tfinite,1]))

embtfinite = jnp.isfinite(embtraj.ts)
for yy in embtraj.ys[embtfinite] :
    print(irx.ut2i(yy)[0])
    draw_angint(ax, irx.ut2i(yy)[0], irx.ut2i(yy)[1])

plt.show()

