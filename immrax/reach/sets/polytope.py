from ..dual_star import DualStar
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
from ...utils import null_space
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Annulus
from matplotlib.axes import Axes
from jax.tree_util import register_pytree_node_class
from ...inclusion import Interval, interval, icentpert, i2centpert, natif
import numpy as onp
from pypoman import plot_polygon, compute_polytope_vertices, project_polytope

@register_pytree_node_class
class Polytope (DualStar) :
    def __init__ (self, ox, H, ly, uy) :
        super().__init__(ox, [H], [ly], [uy])
        # m_ks = [H.shape[0], len(ly), len(uy)]
        # if len(set(m_ks)) != 1 :
        #     raise Exception(f"Dimension mismatch: {m_ks}")
        # self.mk = m_ks[0]
        # if len(ox) != H.shape[1] :
        #     raise Exception(f"Dimension mismatch: {ox.shape=} {H.shape=}")
    
    @classmethod
    def from_ds (cls, ds:DualStar) :
        # print(ds.H)
        return cls(ds.ox, ds.H[0], ds.ly[0], ds.uy[0])
    
    def g (self, i:int, a:ArrayLike) :
        # Identity for polytopes. ly <= Hx <= uy
        if i != 0 :
            raise Exception(f"Polytope has only one H, got {i=}")

        return a
    
    def ginv (self, i:int, iy:Interval) :
        # Inverse image is also identity
        if i != 0 :
            raise Exception(f"Polytope has only one H, got {i=}")

        return iy
    
    def plot_projection (self, ax, xi=0, yi=1, rescale=False, **kwargs) :
        Hi = onp.vstack((-self.H[0], self.H[0]))
        bi = onp.hstack((-self.ly[0], self.uy[0]))
        if Hi.shape[1] == 2 :
            V = compute_polytope_vertices(Hi, bi)
        elif Hi.shape[1] > 2 :
            E = onp.zeros((2, len(self.H[0])))
            E[0, xi] = 1; E[1, yi] = 1
            Hi = onp.vstack((-self.H[0], self.H[0]))
            bi = onp.hstack((-self.ly[0], self.uy[0]))
            V = project_polytope((E, jnp.zeros(2)), (Hi, bi))
        plt.sca(ax)
        kwargs.setdefault('alpha', 1.)
        kwargs.setdefault('fill', False)
        plot_polygon([v + self.ox[(xi,yi),] for v in V], **kwargs)
        # plot_polygon(V, **kwargs)
    
    def one_d_proj (self, yi=0, rescale=False, **kwargs) :
        # 1D projection onto xi, time. Plotted as a tube
        Hi = onp.vstack((-self.H[0], self.H[0]))
        bi = onp.hstack((-self.ly[0], self.uy[0]))
        # V = compute_polytope_vertices(Hi, bi)
        E = onp.zeros((1, len(self.H[0])))
        E[0, yi] = 1
        return project_polytope((E, jnp.zeros(1)), (Hi, bi))


    @classmethod
    def from_Hpolytope (H, uy, ox=jnp.zeros(2)) :
        return Polytope(ox, H, -jnp.inf*jnp.ones_like(uy), uy)

    @classmethod
    def from_interval (cls, *args) :
        cent, pert = i2centpert(interval(*args))
        return Polytope(cent, jnp.eye(len(cent)), -pert, pert)

@register_pytree_node_class
class IntervalDualStar (Polytope) :
    def __init__ (self, I:Interval, ox=None) :
        ox = (I.upper + I.lower) / 2 if ox is None else ox
        super().__init__(ox, jnp.eye(len(ox)), I.lower - ox, I.upper - ox)

def ds_add_interval (ds:DualStar) :
    """Add an interval to the DualStar

    Parameters
    ----------
    ds : DualStar
        The DualStar to add the interval to
    iy : Interval
        The interval to add
    """
    ids = IntervalDualStar(ds.iover(), ds.ox)
    # List addition here
    return DualStar(ds.ox, ds.H + ids.H, ds.ly + ids.ly, ds.uy + ids.uy)
    