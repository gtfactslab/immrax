import functools
import inspect

import jax
import jax.numpy as jnp
from jax.interpreters import ad, batching, mlir

from immrax.inclusion import nif


def register_inclusion_primitive(inclusion_func, *, abstract_eval_func=None):
    """A decorator to register a JAX traceable function as a primitive with a custom inclusion function.

    This annotation will:
    1. Create a custom primitive that is bound to the implementation of the given function.
    2. Associate the primitive with default batching, lowering, and jvp rules.
    3. Derive the correct shape for abstract evaluation, and associate this shape with the new primitive's abstract eval.
    4. Associate the primitive with the given inclusion function in `nif.inclusion_registry`.

    For example, to create a primitive for `jnp.polyval` with a custom inclusion function:

    .. code-block:: python

        from immrax.inclusion.custom_if import register_inclusion_primitive
        import jax.numpy as jnp

        def polyval_inclusion(a, x):
            # custom inclusion logic
            ...

        polyval_primitive_func = register_inclusion_primitive(polyval_inclusion)(jnp.polyval)

    Now `polyval_primitive_func` can be used in computations, and `nif.natif` will dispatch
    to `polyval_inclusion` when it encounters this primitive.

    Args:
        inclusion_func: The inclusion function to be associated with the primitive.
        abstract_eval_func: (Optional) A custom abstract evaluation function for shape inference.
            If not provided, a default will be used which may fail for complex functions.
    """

    def decorator(func):
        primitive_name = f"{func.__name__}_p"
        primitive = jax.extend.core.Primitive(primitive_name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if kwargs:
                raise TypeError(
                    f"Primitive '{primitive_name}' does not support keyword arguments. "
                    "Consider using functools.partial to bind keyword arguments."
                )
            return primitive.bind(*args)

        # 1. Implementation
        primitive.def_impl(func)

        # 2. Abstract evaluation (shape inference)
        if abstract_eval_func is not None:
            primitive.def_abstract_eval(abstract_eval_func)
        else:

            def default_abstract_eval(*args_aval):
                try:
                    return jax.eval_shape(func, *args_aval)
                except Exception as e:
                    raise TypeError(
                        f"Automatic shape inference for '{func.__name__}' failed. "
                        "Please provide a custom `abstract_eval_func` to `register_inclusion_primitive` "
                        "to specify the output shape of your function."
                    ) from e

            primitive.def_abstract_eval(default_abstract_eval)

        # 3. JIT lowering
        # Assuming the wrapped function returns a single result, as in polynomial.py.
        lowering = mlir.lower_fun(func, multiple_results=False)
        mlir.register_lowering(primitive, lowering)

        # 4. Batching rule
        def batching_rule(vector_arg_values, batch_axes):
            res = jax.vmap(func, in_axes=batch_axes)(*vector_arg_values)

            if isinstance(res, (list, tuple)):
                return res, tuple([0] * len(res))
            else:
                return res, 0

        batching.primitive_batchers[primitive] = batching_rule

        # 5. JVP rules for autodiff
        try:
            sig = inspect.signature(func)
            num_args = sum(
                1
                for param in sig.parameters.values()
                if param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
        except (ValueError, TypeError):
            # Fallback for C functions or other callables without a clear signature
            # This part might need to be adjusted if those functions are used.
            # For now, we won't define JVP rules if we can't inspect the signature.
            num_args = 0

        jvprules = []
        for i in range(num_args):

            def make_jvp_rule(arg_num):
                def jvp_rule(tangent, *primals):
                    # Create zero tangents for all other arguments
                    tangents = [
                        tangent
                        if i == arg_num
                        else jax.tree_util.tree_map(jnp.zeros_like, primal)
                        for i, primal in enumerate(primals)
                    ]
                    _primals_out, tangents_out = jax.jvp(func, primals, tuple(tangents))
                    return tangents_out

                return jvp_rule

            jvprules.append(make_jvp_rule(i))

        if jvprules:
            ad.defjvp(primitive, *jvprules)

        # 6. Register inclusion function
        nif.inclusion_registry[primitive] = inclusion_func

        # The returned wrapper is the new function that uses the primitive.
        # We can attach the primitive to it for inspection.
        wrapper.primitive = primitive
        return wrapper

    return decorator
