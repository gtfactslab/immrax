import functools
import inspect

import jax
import jax.numpy as jnp
from jax.interpreters import ad, batching, mlir

from immrax.inclusion import nif


class custom_if:
    """A decorator to define a custom inclusion function for a JAX-traceable function.

    This is analogous to `jax.custom_jvp`.

    This annotation will:
    1. Create a custom primitive that is bound to the implementation of the given function.
    2. Associate the primitive with default batching, lowering, and jvp rules.
    3. Derive the correct shape for abstract evaluation, and associate this shape with the new primitive's abstract eval.
    4. Associate the primitive with a custom inclusion function, which can be defined using the `@f.defif` decorator.

    For example, to create a primitive for `jnp.polyval` with a custom inclusion function:

    .. code-block:: python

        from immrax.inclusion.custom_if import custom_if
        import jax.numpy as jnp

        @custom_if
        def polyval(a, x):
            return jnp.polyval(a, x)

        @polyval.defif
        def polyval_inclusion(a, x):
            # custom inclusion logic
            ...

    Now `polyval` can be used in computations, and `nif.natif` will dispatch
    to `polyval_inclusion` when it encounters this primitive.
    """

    def __init__(self, fun):
        self.fun = fun
        self._if = None
        self.primitive = self._create_primitive()
        functools.update_wrapper(self, fun)

    def _create_primitive(self):
        primitive_name = f"{self.fun.__name__}_p"
        primitive = jax.extend.core.Primitive(primitive_name)

        # 1. Implementation
        primitive.def_impl(self.fun)

        # 2. Abstract evaluation (shape inference)
        def default_abstract_eval(*args_aval):
            try:
                shape_dtype = jax.eval_shape(self.fun, *args_aval)
                # TODO: I am not entirely sure if this will respect the device / ref counting behavior of the wrapped function
                # Should look here first if those types of problems come up
                return jax.core.ShapedArray(shape_dtype.shape, shape_dtype.dtype)
            except Exception as e:
                raise TypeError(
                    f"Automatic shape inference for '{self.fun.__name__}' failed. "
                ) from e

        primitive.def_abstract_eval(default_abstract_eval)

        # 3. JIT lowering
        # Assuming the wrapped function returns a single result, as in polynomial.py.
        lowering = mlir.lower_fun(self.fun, multiple_results=False)
        mlir.register_lowering(primitive, lowering)

        # 4. Batching rule
        def batching_rule(vector_arg_values, batch_axes):
            res = jax.vmap(self.fun, in_axes=batch_axes)(*vector_arg_values)

            if isinstance(res, (list, tuple)):
                return res, tuple([0] * len(res))
            else:
                return res, 0

        batching.primitive_batchers[primitive] = batching_rule

        # 5. JVP rules for autodiff
        try:
            sig = inspect.signature(self.fun)
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
                    _primals_out, tangents_out = jax.jvp(
                        self.fun, primals, tuple(tangents)
                    )
                    return tangents_out

                return jvp_rule

            jvprules.append(make_jvp_rule(i))

        if jvprules:
            ad.defjvp(primitive, *jvprules)

        # 6. Register inclusion function
        def inclusion_dispatcher(*args, **kwargs):
            if self._if is None:
                raise NotImplementedError(
                    f"No inclusion function defined for '{self.fun.__name__}'. "
                    f"Use '@{self.fun.__name__}.defif' to define it."
                )
            return self._if(*args, **kwargs)

        nif.inclusion_registry[primitive] = inclusion_dispatcher

        return primitive

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise TypeError(
                f"Primitive '{self.primitive.name}' does not support keyword arguments. "
                "Consider using functools.partial to bind keyword arguments."
            )
        return self.primitive.bind(*args)

    def defif(self, if_fun):
        """Decorator to define the inclusion function for a custom_if function."""
        self._if = if_fun
        return if_fun
