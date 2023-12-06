import jax
from jax import jit, grad, jacfwd, jacrev, jvp, vjp
import jax.numpy as jnp
from jaxtyping import Float, Integer
import numpy as np
from cyipopt import minimize_ipopt
from immrax import OpenLoopSystem, ControlledSystem, EmbeddingSystem
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Wedge
from immrax.inclusion import *
from immrax.utils import plot_interval_t, get_partitions_ut
from immrax.embedding import natemb, ifemb

# Enable 64 bit floating point precision
jax.config.update("jax_enable_x64", True)
# # We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
# config.update('jax_platform_name', 'cpu')

g = 9.81

class Pendulum (OpenLoopSystem) :
    m:float
    l:float
    b:float

    def __init__(self, m:float = 0.1, l:float = 0.5, b:float = 0.1) -> None:
        super().__init__()
        self.evolution = 'continuous'
        self.xlen = 2
        self.m = m
        self.l = l
        self.b = b

    def f(self, t: float, x: jax.Array, u: jax.Array, w: jax.Array) -> jax.Array:
        return jnp.array([
            x[1],
            (((1 + w[0])*u[0] - self.b*x[1]) / ((self.m) * self.l**2)) - (g/self.l)*jnp.sin(x[0])
        ])
    
    def get_xy (self, x:jax.Array) :
        return self.l*jnp.sin(x[0]), -self.l*jnp.cos(x[0])



# class CLEmbeddingPendulum (EmbeddingSystem) :
#     sys:Pendulum
#     def __init__(self, sys:Pendulum) -> None:
#         self.sys = sys
#         self.evolution = self.sys.evolution
#         self.xlen = self.sys.xlen * 2
#         self.sys_mjacM = mjacM (sys.f)
    
#     @partial(jit, static_argnums=(0,), static_argnames=['orderings'])
#     def E (self, t:Interval, x:jax.Array, u:Interval, w:Interval,
#            K:jax.Array, nominal:Sequence[jax.Array], orderings:Tuple[Ordering] = None) :
#         t = interval(t)
#         ix = ut2i(x)
#         K = interval(K)

#         Jt, Jx, Ju, Jw = self.sys_mjacM(t, ix, u, w, orderings=orderings, centers=(nominal,))[0]
#         # Jt, Jx, Ju, Jw = [Interval.from_jittable(Mi) for Mi in Mpre[0]]
#         tc, xc, uc, wc = nominal

#         def F (t:Interval, x:Interval, u:Interval, w:Interval) :
#             return (
#                 # ([Jx] + [Ju]K)([\ulx,\olx] - x_nom)
#                 (Jx + Ju @ K) @ (x - xc)
#                 # + [Ju](u - u_nom)
#                 + Ju @ (u - uc)
#                 # + [Jw]([\ulw,\olw] - w_nom)
#                 + Jw @ (w - wc)
#                 # + f(xc, uc, wc)
#                 + self.sys.f(tc, xc, uc, wc)
#             )

#         n = self.sys.xlen
#         ret = jnp.empty(self.xlen)
#         for i in range(n) :
#             _xi = jnp.copy(x).at[i+n].set(x[i])
#             ret = ret.at[i].set(F(interval(t), ut2i(_xi), u, w).lower[i])
#             x_i = jnp.copy(x).at[i].set(x[i+n])
#             ret = ret.at[i+n].set(F(interval(t), ut2i(x_i), u, w).upper[i])
#         return ret

sys = Pendulum(m=0.15)
sys_mjacM = mjacM(sys.f)
def F (t:Interval, x:Interval, u:Interval, w:Interval, K:jnp.ndarray, nominal:jnp.ndarray) :
    Mt, Mx, Mu, Mw = sys_mjacM(t, x, u, w, orderings=None, centers=(jnp.array([0.]),))[0]
    Jt, Jx, Ju, Jw = self.sys_mjacM(t, ix, u, w, orderings=orderings, centers=(nominal,))[0]
    # Jt, Jx, Ju, Jw = [Interval.from_jittable(Mi) for Mi in Mpre[0]]
    tc, xc, uc, wc = nominal
    return (
        # ([Jx] + [Ju]K)([\ulx,\olx] - x_nom)
        (Jx + Ju @ K) @ (x - xc)
        # + [Ju](u - u_nom)
        + Ju @ (u - uc)
        # + [Jw]([\ulw,\olw] - w_nom)
        + Jw @ (w - wc)
        # + f(xc, uc, wc)
        + self.sys.f(tc, xc, uc, wc)
    )

# embsys = CLEmbeddingPendulum(sys)
natembsys = natemb(sys)

# t0, tf = 0, 4.
# t0, te, tf, dt = 0, 4., 4., 0.05
t0, te, tf, dt = 0, 4., 4.25, 0.05
# t0, te, tf, dt = 0, 4.25, 4.5, 0.05
x0 = interval(jnp.array([0.,0.]))
x0ut = i2ut(x0); x0cent, x0pert = i2centpert(x0)
xf = icentpert(jnp.array([jnp.pi,0.]), jnp.array([10.*(jnp.pi/360),0.1]))
# xf = icentpert(jnp.array([jnp.pi,0.]), jnp.array([10.*(jnp.pi/360),0.05]))
xfl, xfu = i2lu(xf); xfut = i2ut(xf); xfcent, xfpert = i2centpert(xf)
# w = icentpert(0., 0.03)
w = icentpert(0., 0.02)
# w = icentpert(0., 0.01)
Ne = round((te - t0)/dt) 
N = round((tf - t0)/dt) + 1
tt = jnp.arange(t0, tf + dt, dt)
# K = jnp.array([[-10*sys.m, -5*sys.m]])
# K = jnp.array([[-5*sys.m, -5*sys.m]])
# K = jnp.array([[-0.1, -1.]])
# K = jnp.array([[-.1, -.5]])
K = jnp.array([[-1.,-1.]])

# initial guess
u0 = jnp.concatenate((jnp.zeros(N), K.reshape(-1)))
# u0 = jnp.zeros(N)

def split_u (u:jax.Array) -> jax.Array :
    return u[:-2], u[-2:].reshape(1,2)
    # return u, K

@jit
def rollout_ol_sys_undisturbed (u:jax.Array) -> jax.Array :
    u, K = split_u(u)
    def f_euler (xt, ut) :
        xtp1 = xt + dt*sys.f(0., xt, jnp.array([ut]), jnp.array([0.]))
        return (xtp1, xtp1)
    _, x = jax.lax.scan(f_euler, x0cent, u)
    return x

@jit
def rollout_cl_embsys (u:jax.Array) -> jax.Array :
    u, K = split_u(u)
    def f_euler (xt, ut) :
        xtut, xnomt = xt
        xtutp1 = xtut + dt*embsys.E(0., xtut, interval(jnp.array([ut])), w, K, 
                                (jnp.array([0.]), xnomt, jnp.array([ut]), jnp.array([0.])))
        xnomtp1 = xnomt + dt*sys.f(0., xnomt, jnp.array([ut]), jnp.array([0.]))
        return ((xtutp1,xnomtp1), xtutp1)
    _, x = jax.lax.scan(f_euler, (x0ut,x0cent), u)
    return x

@jit
def rollout_natembsys (u:jax.Array) -> jax.Array :
    u, K = split_u(u)
    def f_euler (xt, ut) :
        xtp1 = xt + dt*natembsys.E(0., xt, interval(jnp.array([ut])), w)
        return (xtp1, xtp1)
    _, x = jax.lax.scan(f_euler, x0ut, u)
    return x

# OBJECTIVE
@jit
def obj (u:jax.Array) -> jax.Array :
    x = rollout_cl_embsys(u)
    return jnp.sum(u**2) + jnp.sum((x[:,2:] - x[:,:2])**2)
    # return 0.1*jnp.sum(u**2) + 0.01*jnp.sum((x[:,2:] - x[:,:2])**2)
    # x_un = rollout_ol_sys_undisturbed(u)
    # + 0.1*(jnp.sum(((x[-1,2:] + x[-1,:2])/2 - xfcent)**2))
    # return 0.1*jnp.sum(u[:-2]**2) + 1*jnp.sum(jnp.sum((x[:,2:] - x[:,:2])**2)) + 10*jnp.sum(u[-2:]**2)
    # return 0.1*jnp.sum(u[:-2]**2) + 0.1*jnp.sum(jnp.sum((x[:,2:] - x[:,:2])**2)) + 0.1*jnp.sum((x_un[-1] - xfcent)**2) + 10*jnp.sum(u[-2:]**2)
    # return jnp.sum(0.1*u**2) + jnp.sum((x[-1] - xfcent)**2)

# CONSTRAINTS 
# Two inequality constraints: 
# xf.lower <= x(tf).lower  ==>  x(tf).lower - xf.lower >= 0
# x(tf).upper <= xf.upper  ==>  xf.upper - x(tf).upper >= 0
@jit
def con_ineq (u:jax.Array) :
    x = rollout_cl_embsys(u)
    return jnp.concatenate(((x[Ne:,:2] - xfl).reshape(-1), (xfu - x[Ne:,2:]).reshape(-1)))
    # return jnp.concatenate((x[-1,:2] - xfl, xfu - x[-1,2:]))

obj_grad = jit(grad(obj))  # objective gradient
obj_hess = jit(jacfwd(jacrev(obj)))  # objective hessian
# con_eq_jac = jit(jacfwd(con_eq))  # jacobian
# con_eq_hess = jit(jacrev(jacfwd(con_eq))) # hessian
# con_eq_hessvp = jit(lambda x, v: con_eq_hess(x).T @ v) # hessian vector-product
con_ineq_jac = jit(jacfwd(con_ineq))  # jacobian
con_ineq_hess = jit(jacfwd(jacrev(con_ineq))) # hessian

@jit
def con_ineq_hessvp (u, v) :
    def hessvp (u) :
        _, hvp = jax.vjp(con_ineq, u)
        return hvp(v)[0] # One tangent, one output. u^T dc_v
    return jacrev(hessvp)(u) 
    
print('JIT Compiling')
obj_grad(u0)
obj_hess(u0)
con_ineq_jac(u0)
con_ineq_hessvp(u0, jnp.ones(4*(N - Ne)))

# con_ineq_hess(u0)

# constraints
# Note that 'hess' is the hessian-vector-product
cons = [
    # {'type': 'eq', 'fun': con_eq, 'jac': con_eq_jac, 'hess': con_eq_hessvp},
    {'type': 'ineq', 'fun': con_ineq, 'jac': con_ineq_jac, 'hess': con_ineq_hessvp},
]

# # variable bounds: 1 <= x[i] <= 5
bnds = [(-100.,100.) for _ in range(u0.size)]

ipopt_opts = {
    b'disp': 5, 
    b'linear_solver': 'ma57', 
    b'hsllib': 'libcoinhsl.so', 
    b'tol': 1e-5, 
    b'max_iter': 10000,
}

print('Finished setup and compilation.')

# res = minimize_ipopt(obj, jac=obj_grad, hess=obj_hess, x0=u0, bounds=bnds,
#                      constraints=cons, options=ipopt_opts)
res = minimize_ipopt(obj, jac=obj_grad, hess=obj_hess, x0=u0,
                     constraints=cons, options=ipopt_opts)

uu = res.x
# uu = 1.0*jnp.exp(-tt)
print(uu)

unrolledout = rollout_ol_sys_undisturbed(uu)
clrolledout = rollout_cl_embsys(uu)
narolledout = rollout_natembsys(uu)

if jnp.all(con_ineq(uu) >= 0) :
    print('Constraints satisfied!')
else :
    print('Constraints not satisfied!')
    print(con_ineq(uu))


# print(unrolledout)
# print(clrolledout)
# print(narolledout)

names = ['Undisturbed', 'Closed-loop (M-Jac)', 'Open-loop (Natural)']

fig1, axs = plt.subplots(1,2,dpi=100,figsize=[8,4])

axs[0].plot(tt, unrolledout[:,0], color='k')
axs[1].plot(tt, unrolledout[:,1], color='k')
plot_interval_t(axs[0], tt, xf[0]*jnp.ones(N), color='k')
plot_interval_t(axs[1], tt, xf[1]*jnp.ones(N), color='k')
plot_interval_t(axs[0], tt, interval(clrolledout[:,0], clrolledout[:,2]), color='tab:blue')
plot_interval_t(axs[1], tt, interval(clrolledout[:,1], clrolledout[:,3]), color='tab:blue')
# plot_interval_t(axs[0], tt, interval(narolledout[:,0], narolledout[:,2]), color='tab:red')
# plot_interval_t(axs[1], tt, interval(narolledout[:,1], narolledout[:,3]), color='tab:red')

fig1.savefig('figures/compared.png')

# to_plot = [unrolledout, clrolledout, narolledout]
to_plot = [unrolledout, clrolledout]
th2deg = lambda th : (th * 180) / jnp.pi - 90
fig2, anim = plt.subplots(1,len(to_plot),dpi=100,figsize=[8,4])
wedges = []
for i, xx in enumerate(to_plot) :
    if i == 0:
        wedges.append(anim[i].add_patch(Wedge((0,0),sys.l,th2deg(xx[0,0]),th2deg(xx[0,0]), lw=2, ec='k')))
    else:
        wedges.append(anim[i].add_patch(Wedge((0,0),sys.l,th2deg(xx[0,0]),th2deg(xx[0,2]), lw=2, ec='k')))
    anim[i].add_patch(Wedge((0,0), sys.l, th2deg(xfl[0]), th2deg(xfu[0]), color='k', alpha=0.25))
    anim[i].set_xlim(-sys.l*1.2, sys.l*1.2)
    anim[i].set_ylim(-sys.l*1.2, sys.l*1.2)
def animate(t) :
    for i, xx in enumerate(to_plot) :
        if i == 0 :
            wedges[i].set( theta1 = th2deg(xx[t,0]), theta2 = th2deg(xx[t,0]) )
        else :
            wedges[i].set( theta1 = th2deg(xx[t,0]), theta2 = th2deg(xx[t,2]) )

interval = dt*1000
ani = animation.FuncAnimation(fig2, animate, frames=N, repeat=True, interval=interval)
FFwriter = animation.FFMpegWriter(fps=(1/dt))
ani.save('figures/compared.mp4', writer=FFwriter)

plt.show()
