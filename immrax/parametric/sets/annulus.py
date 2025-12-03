from ..parametope import hParametope
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
from ...utils import null_space
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Annulus, Patch
from matplotlib.axes import Axes
import numpy as onp
from matplotlib.path import Path

# import transforms from matplotlib
from matplotlib import transforms
from jax.tree_util import register_pytree_node_class
from ...inclusion import Interval, interval, icentpert


@register_pytree_node_class
class LpAnnulus(hParametope):
    p: float

    def __init__(self, ox, H, ly, uy, p=2.0):
        # ly = jnp.zeros_like(uy) if uy is not None else None
        super().__init__(ox, [H], [ly], [uy])
        self.p = p

    @classmethod
    def from_parametope(cls, pt: hParametope):
        return LpAnnulus(pt.ox, pt.alpha, pt.y)

    def g(self, i: int, a: ArrayLike):
        if i != 0:
            raise Exception(f"Ellipsoid has only one constraint, got {i=}")
        return jnp.sum(jnp.abs(a) ** self.p) ** (1 / self.p)

    def ginv(self, i: int, iy: Interval):
        # Returns a box containing the preimage of the constraint over iy

        if i != 0:
            raise Exception(f"Annulus has only one constraint, got {i=}")

        n = len(self.ox)

        # |x|_inf \leq |x|_p \leq n^{1/p} |x|_inf
        return icentpert(jnp.zeros(n), iy.upper * jnp.ones(n))

    def iover(self):
        return self.ginv(self.H @ interval(self.ly, self.uy))

    @property
    def P(self):
        return self.H[0].T @ self.H[0]

    def V(self, x: ArrayLike):
        return self.g(0, self.H[0] @ (x - self.ox))

    # def plot_projection (self, ax, xi=0, yi=1, rescale=False, **kwargs) :

    def __repr__(self):
        return f"Ellipsoid(ox={self.ox}, H={self.H}, uy={self.uy})"

    def __str__(self):
        return f"Ellipsoid(ox={self.ox}, H={self.H}, uy={self.uy})"
