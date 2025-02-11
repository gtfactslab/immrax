from ..dual_star import DualStar
import jax.numpy as jnp
from jaxtyping import ArrayLike
from ...utils import null_space
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Annulus
from matplotlib.axes import Axes

class Polytope (DualStar) :
    def __init__ (self, ox, H, ly, uy) :
        super().__init__(ox, [lambda x : x], [H], [ly], [uy])
    
def H_polytope (H, uy, ox=jnp.zeros(2)) :
    return Polytope(ox, H, -jnp.inf*jnp.ones_like(uy), uy)