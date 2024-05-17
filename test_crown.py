import jax
import jax.lax as lax
import jax.numpy as jnp
import immrax as irx
import equinox as eqx
import equinox.nn as nn

"""
Use scan to implement the linear bound propogation on the neural network.
"""

W = jax.random.normal(jax.random.PRNGKey(0), (10, 10))
b = jax.random.normal(jax.random.PRNGKey(0), (10,))

net = lambda x : jax.nn.sigmoid(W @ x + b)

print(net(jnp.ones((10,))))

