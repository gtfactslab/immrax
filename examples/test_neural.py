from jax import jit
from immrax.inclusion import *
from immrax.neural import *

net = NeuralNetwork('vehicle/100r100r2')
net_crown = fastlin(net)

x0 = icentpert(jnp.zeros(4), 0.01)
res = net_crown(x0)
print(res)
print(res(x0))

res = net_crown(x0)
print(res(x0))
