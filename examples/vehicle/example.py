import jax
import jax.numpy as jnp
from immrax import Control, NeuralNetwork
from immrax import OpenLoopSystem, NNCSystem, NNCEmbeddingSystem
from immrax import TransformEmbedding
from immrax import interval, icentpert, i2ut, ut2i
from immrax.system import OpenLoopSystem
from immrax import Corner, bot_corner, top_corner, two_corners, all_corners, standard_ordering
from immrax.utils import run_times, draw_iarray, draw_iarrays, get_partitions_ut, gen_ics
# from immrax import *
from functools import partial
import equinox as eqx
import sympy as sp
import jax_verify as jv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from immutabledict import immutabledict

method = 'euler'
device = 'gpu'

def jit (*args, **kwargs) :
    kwargs.setdefault('backend', device)
    return jax.jit(*args, **kwargs)

# t, px, py, psi, v, u1, u2, w = sp.symbols('t p_x p_y psi v u1 u2 w')

class Vehicle (OpenLoopSystem) :
    def __init__(self) -> None:
        self.evolution = 'continuous'
        self.xlen = 4
    def f(self, t:jnp.ndarray, x:jnp.ndarray, u:jnp.ndarray, w:jnp.ndarray) -> jnp.ndarray :
        px, py, psi, v = x.ravel()
        u1, u2 = u.ravel()
        beta = jnp.arctan(jnp.tan(u2)/2)
        return jnp.array([
            v*jnp.cos(psi + beta),
            v*jnp.sin(psi + beta),
            v*jnp.sin(beta),
            u1
        ])

# print(sys.f())
# print(sys.f_eqn)
sys = Vehicle()
net = NeuralNetwork('100r100r2')
clsys = NNCSystem(sys, net)

cent = jnp.array([8,7,-2*jnp.pi/3,2])
x0 = icentpert([8,7,-2*jnp.pi/3,2], [0.1,0.1,0.01,0.01])
# x0 = icentpert([8,7,-2*jnp.pi/3,2], [0.05,0.05,0.01,0.01])
# x0 = icentpert([8,7,-2*jnp.pi/3,2], [0.1,0.1,0.01,0.01])
# x0 = icentpert([8,7,-2*jnp.pi/3,2], [0.05,0.05,0.01,0.01])
# x0 = icentpert([8,7,-2*jnp.pi/3,2], [0.05,0.05,0.01,0.01])
# x0 = icentpert([8.,7.,-2*jnp.pi/3,2.], [0.01,0.01,0.01,0.01])
# x0 = icentpert([8,7,-2*jnp.pi/3,2], [0.0,0.0,0.0,0.0])
w = icentpert([0.], 0.)
clembsys = NNCEmbeddingSystem(clsys, 'crown', 'local', 'local')
# clembsys.E(interval(0.), i2ut(x0), w)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 14
})
fig, axs = plt.subplots(2,3,dpi=100,figsize=[10,7])
fig.subplots_adjust(left=0.05, right=0.975, bottom=0.075, top=0.975, wspace=0.15, hspace=0.25)
axs = axs.reshape(-1)
# NN = [(1, method, True),]
NN = [(1, method, True), (2**4, method, True), (3**4, method, True),
      (4**4, method, True), (5**4, method, True), (6**4, method, True)]

orderings = standard_ordering (1+4+2+1)
# corners = bot_corner(1+4+2+1)
# corners = top_corner(1+4+2+1)
corners = two_corners(1+4+2+1)

for ax, (N, solver, to_plot) in zip(axs, NN) :
    x0s = get_partitions_ut(i2ut(x0), N)
    print(f'Using {len(x0s)} partitions')

    def w_map (t, x) :
        return w
    # clembsys.compute_trajectory(0.,1.5,jnp.zeros(8),(w_map,),0.125)


    # @jax.jit
    def compute_traj (x0, t_end) :
        return clembsys.compute_trajectory(0.,t_end,x0,(w_map,),0.125, 
                                           f_kwargs=immutabledict({'corners': corners,
                                                                   'orderings': orderings}), 
                                           solver=solver)

    t_end = 1.5
    z = jnp.zeros(8)

    # compute_traj(z, 0.125)

    vmapped = jit(jax.vmap(compute_traj, (0, None)))
    vmapped(jnp.zeros_like(x0s), 0.125)
    # print(jax.make_jaxpr(vmapped)(jnp.zeros((1,8))))
    print('Finished setup and compilation.')

    trajs, times = run_times(10,vmapped,x0s, t_end)
    avg_runtime, std_runtime = jnp.mean(times), jnp.std(times)
    print(f'partitions: ${len(x0s)}$, {solver}, {avg_runtime} ± {std_runtime}')
    # print(trajs.ts[0,jnp.isfinite(trajs.ts[0])])
    # print(trajs.ys[0,jnp.isfinite(trajs.ts[0])])

    tfinite = jnp.where(jnp.isfinite(trajs.ts[0]))[0]
    print(tfinite)
    print(trajs.ys[:,tfinite,:])

    if to_plot :
        for t in tfinite :
        # for t in range(trajs.ys.shape[1]) :
        # for t in range(Tf) :
            ut_t = trajs.ys[:,t,:]
            boxes_t = [ut2i(box) for box in ut_t]
            draw_iarrays(ax, boxes_t, zorder=2)

        def mc_wmap (t, x) :
            return jnp.array([0.])
        for mc_x0 in gen_ics(x0, 100) :
            mc_traj = clsys.compute_trajectory(0.,t_end,mc_x0,(mc_wmap,),0.125,solver=solver)
            ax.plot(mc_traj.ys[:,0], mc_traj.ys[:,1], color='tab:red', zorder=0)


        # # for yi in traj.ys :
        # #     draw_iarrays(plt,ut2i(yi))

        ax.add_patch(Circle((4,4),3/1.25,lw=0,fc='salmon',zorder=0))

        ax.set_xlim([-0.5,8.5])
        ax.set_ylim([-0.5,8.5])
        ax.set_xlabel('$p_x$',labelpad=3); ax.set_ylabel('$p_y$',labelpad=3, rotation='horizontal')

        # ax.text(0,8,f'runtime: ${avg_runtime:.3f} \\pm {std_runtime:.3f}$',fontsize=12,verticalalignment='top')
        # ax.text(0,7.5,f'partitions: ${len(x0s)}$',fontsize=12,verticalalignment='top')
        ax.text(0,8,f'partitions: ${len(x0s)}$, {solver}',fontsize=16,verticalalignment='top')

fig.savefig('vehicle.pdf')
plt.show()
