import jax
import jax.numpy as jnp
import immrax as irx

x = 0.5*jnp.array([1., 1.])
status = lambda b : "inside" if b else "outside"

# Testing Normotopes

for NT in [irx.L1Normotope, irx.LinfNormotope, irx.L2Normotope] :
    nt = NT(jnp.array([0., 0.]), jnp.eye(2), 0.9)
    print(f"{x} is {status(nt.contains(x))} {nt}.\n")

# Testing Polytopes
A = jnp.array([[1., 1.], [-1., 2.], [2., -1.]])
b = jnp.array([1.0, 1.0, 1.0])
# polytope = irx.Polytope(jnp.array([0., 0.]), A, jnp.hstack((b,b)))
polytope = irx.Polytope.from_Hpolytope(A, b)
print(polytope.g(x))
print(f"{x} is {status(polytope.contains(x))} {polytope}.\n")

# Example usage of g_parametope, creates a halfspace
def g (alpha, x) :
    return jnp.dot(alpha, x)

Halfspace = irx.g_parametope(g, 'Halfspace')

halfspace = Halfspace(jnp.array([0., 0.]), jnp.array([1., 2.]), jnp.array([3.]))
print(halfspace)


print(f"{x} is {status(halfspace.contains(x))} {halfspace}.\n")


