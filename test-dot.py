import jax
import jax.lax as lax
import jax.numpy as jnp
import immrax as irx

A = jnp.ones((100,))
B = jnp.ones((100, 100))

iA = irx.icentpert(A, 0.1)
iB = irx.icentpert(B, 0.1)

res1, t1 = irx.utils.run_times(1000, jnp.matmul, A, B)
res2, t2 = irx.utils.run_times(1000, jax.jit(irx.natif(jnp.matmul)), iA, iB)

# print(res1)
# print(res2)

# print(jnp.allclose(res1, res2))

print(f'{jnp.mean(t1)} +/- {jnp.std(t1)}')
print(f'{jnp.mean(t2)} +/- {jnp.std(t2)}')

# print(lax.dot_general(A, B, (((1, 2), (0, 2)), ((), ()))).shape)

# print(irx.natif(jnp.matmul)(A, B))
# print(irx.natif(lax.dot_general)(iA, iB, (((0,), (0,)), ())))

print((iA@iB).shape)

A = jnp.ones((10,100))
iA = irx.icentpert(A, 0.1)

print((iA@iB).shape)


