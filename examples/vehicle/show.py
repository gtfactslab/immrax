import jax.numpy as jnp
import immrax as irx

net = irx.NeuralNetwork("100r100r2")


class Vehicle(irx.OpenLoopSystem):
    def __init__(self) -> None:
        self.evolution = "continuous"
        self.xlen = 4

    def f(
        self, t: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray, w: jnp.ndarray
    ) -> jnp.ndarray:
        px, py, psi, v = x.ravel()
        u1, u2 = u.ravel()
        beta = jnp.arctan(jnp.tan(u2) / 2)
        return jnp.array(
            [v * jnp.cos(psi + beta), v * jnp.sin(psi + beta), v * jnp.sin(beta), u1]
        )


olsys = Vehicle()
net = irx.NeuralNetwork("100r100r2")
clsys = irx.ControlledSystem(olsys, net)


print(net(jnp.zeros(4)))

crown_net = irx.crown(net)
fastlin_net = irx.fastlin(net)
ix = irx.icentpert(jnp.zeros(4), 1.0)
print(f"{ix=}")
print(crown_net(ix))
res = fastlin_net(ix)
print(res.C)
