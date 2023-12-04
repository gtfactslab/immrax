import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Float, Integer
import immrax as irx

class SPRE22 (irx.System) :
    def __init__(self) -> None:
        self.evolution = 'continuous'
        self.xlen = 4
    def f(self, t:ArrayLike, x:ArrayLike, u:ArrayLike, **kwargs) -> jax.Array :
        px, py, vx, vy = x.ravel()
        ux, uy = u.ravel()
        μ = 3.986 * 10**14 * 60**2
        r = 42164 * 10**3
        mc = 500
        n = jnp.sqrt(μ / r**3)
        rc = jnp.sqrt((r+px)**2 + py**2)
        return jnp.array([
            vx,
            vy,
            n**2 * px + 2*n*vy + μ/(r**2) - μ*(r+px)/(rc**3) + ux/mc,
            n**2 * py - 2*n*vx - μ*py/(rc**3) + uy/mc,
        ])
