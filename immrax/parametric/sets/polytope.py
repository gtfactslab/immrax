from .affine import hParametope
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.tree_util import register_pytree_node_class
from ...inclusion import interval, i2centpert
import numpy as onp
from pypoman import plot_polygon, compute_polytope_vertices, project_polytope


def _lu2y(l, u):
    return jnp.concatenate((-l, u))


def _y2lu(y):
    return -y[: len(y) // 2], y[len(y) // 2 :]


@register_pytree_node_class
class Polytope(hParametope):
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

    @classmethod
    def from_Hpolytope(cls, H, uy, ox=jnp.zeros(2)):
        return Polytope(ox, H, jnp.hstack((jnp.inf * jnp.ones_like(uy), uy)))

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

    def to_Hpolytope(self):
        Hox = self.alpha @ self.ox
        return (
            jnp.vstack((-self.H, self.H)),
            jnp.hstack((-self.ly - Hox, self.uy + Hox)),
        )

    # Override in subclasses to unpack the flattened data
    @classmethod
    def from_parametope(cls, pt):
        return Polytope(pt.ox, pt.alpha, pt.y)
