import jax 
import jax.numpy as jnp
import immrax

f = lambda x : jnp.array([
    (x[0] + x[1])**2, x[0] + x[1] + 2*x[1]*x[2]])
Fnat = jax.jit(immrax.natif(f))
Fjac = jax.jit(immrax.jacif(f))
Fmix = jax.jit(immrax.mjacif(f))
x0 = immrax.icentpert(jnp.zeros(2), 0.1)
for F in [Fnat, Fjac, Fmix] :
    F(x0) # JIT Compile
    ret, times = immrax.utils.run_times(10000, F, x0)
    print(ret)
    print(f'{times.mean():.3e} \u00B1 {times.std():.3e}')