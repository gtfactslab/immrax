import jax.numpy as jnp 
import immrax as irx

x = irx.interval(-1.,1.)

print(irx.natif(jnp.arctan)(x))