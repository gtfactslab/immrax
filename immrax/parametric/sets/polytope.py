from ..parametope import hParametope
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.tree_util import register_pytree_node_class
from ...inclusion import interval, i2centpert
import numpy as onp
from pypoman import plot_polygon, compute_polytope_vertices, project_polytope


def _lu2y(l, u):
    return jnp.concatenate((-l, u))


def _y2lu(y):
    print(y)
    return -y[: len(y) // 2], y[len(y) // 2 :]


@register_pytree_node_class
class Polytope(hParametope):
    # def __init__ (self, ox, H, ) :
    #     super().__init__(ox, H, _lu2y(ly, uy))
    #     # m_ks = [H.shape[0], len(ly), len(uy)]
    #     # if len(set(m_ks)) != 1 :
    #     #     raise Exception(f"Dimension mismatch: {m_ks}")
    #     # self.mk = m_ks[0]
    #     # if len(ox) != H.shape[1] :
    #     #     raise Exception(f"Dimension mismatch: {ox.shape=} {H.shape=}")

    # @classmethod
    # def from_parametope (cls, pt:hParametope) :
    #     # print(ds.H)
    #     return cls(pt.ox, pt.alpha, *_y2lu(pt.y))

    def h(self, z):
        # Identity nonlinearity
        return jnp.concatenate((-z, z))

    def hinv(self, y):
        # Inverse image is also identity
        return interval(-y[: len(y) // 2], y[len(y) // 2 :])

    @property
    def H(self):
        return self.alpha

    @property
    def ly(self):
        return -self.y[: len(self.y) // 2]

    @property
    def uy(self):
        return self.y[len(self.y) // 2 :]

    @property
    def iy(self):
        return interval(self.ly, self.uy)

    def get_vertices(self):
        Hi = jnp.vstack((-self.H, self.H))
        bi = jnp.hstack((-self.ly, self.uy))
        return jnp.asarray(compute_polytope_vertices(Hi, bi)) + self.ox

    def plot_projection(self, ax, xi=0, yi=1, rescale=False, **kwargs):
        Hi = onp.vstack((-self.H, self.H))
        bi = onp.hstack((-self.ly, self.uy))
        if Hi.shape[1] == 2:
            V = compute_polytope_vertices(Hi, bi)
        elif Hi.shape[1] > 2:
            E = onp.zeros((2, self.H.shape[1]))
            E[0, xi] = 1
            E[1, yi] = 1
            Hi = onp.vstack((-self.H, self.H))
            bi = onp.hstack((-self.ly, self.uy))
            # print(Hi.shape, bi.shape)
            V = project_polytope((E, jnp.zeros(2)), (Hi, bi))
        plt.sca(ax)
        kwargs.setdefault("alpha", 1.0)
        kwargs.setdefault("fill", False)
        plot_polygon([v + self.ox[(xi, yi),] for v in V], **kwargs)
        # plot_polygon(V, **kwargs)

    def one_d_proj(self, yi=0, rescale=False, **kwargs):
        # 1D projection onto xi, time. Plotted as a tube
        Hi = onp.vstack((-self.H, self.H))
        bi = onp.hstack((-self.ly, self.uy))
        # V = compute_polytope_vertices(Hi, bi)
        E = onp.zeros((1, len(self.H)))
        E[0, yi] = 1
        return project_polytope((E, jnp.zeros(1)), (Hi, bi))

    # @classmethod
    # def from_Hpolytope (H, uy, ox=jnp.zeros(2)) :
    #     return Polytope(ox, H, -jnp.inf*jnp.ones_like(uy), uy)

    @classmethod
    def from_interval(cls, *args):
        cent, pert = i2centpert(interval(*args))
        return Polytope(cent, jnp.eye(len(cent)), jnp.concatenate((pert, pert)))

    def add_rows(self, Haug, Hp):
        yaug = interval(Haug @ Hp) @ self.hinv(self.y)
        return Polytope(
            self.ox,
            jnp.vstack((self.H, Haug)),
            _lu2y(
                jnp.concatenate((self.ly, yaug.lower)),
                jnp.concatenate((self.uy, yaug.upper)),
            ),
        )

    # Override in subclasses to unpack the flattened data
    @classmethod
    def from_parametope(cls, pt: "hParametope"):
        return Polytope(pt.ox, pt.alpha, pt.y)

    # @classmethod
    # def tree_unflatten (cls, aux_data, children) :
    #     return cls.from_parametope(hParametope(*children))


# @register_pytree_node_class
# class IntervalDualStar (Polytope) :
#     def __init__ (self, I:Interval, ox=None) :
#         ox = (I.upper + I.lower) / 2 if ox is None else ox
#         super().__init__(ox, jnp.eye(len(ox)), I.lower - ox, I.upper - ox)

# def ds_add_interval (ds:DualStar) :
#     """Add an interval to the DualStar

#     Parameters
#     ----------
#     ds : DualStar
#         The DualStar to add the interval to
#     iy : Interval
#         The interval to add
#     """
#     ids = IntervalDualStar(ds.iover(), ds.ox)
#     # List addition here
#     return DualStar(ds.ox, ds.H + ids.H, ds.ly + ids.ly, ds.uy + ids.uy)
