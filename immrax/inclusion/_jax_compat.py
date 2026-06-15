"""Compatibility helpers smoothing over jax version differences.

immrax interprets jaxprs directly, so it depends on a few internal jax APIs that
have changed across releases. Centralizing the version handling here keeps the
interpreters (``nif.py``, ``linbp.py``) free of duplicated shims.

* ``Primitive.get_bind_params`` returned a ``(subfuns, params)`` tuple before
  jax 0.10. From jax 0.10 on it returns just the ``params`` dict, folding any
  subfuns into that dict under the ``"subfuns"`` key.
* The jit primitive was named ``pjit_p`` before jax 0.9 and ``jit_p`` from 0.9 on.
"""

import jax

__all__ = ["split_bind_params", "jit_primitive"]


def split_bind_params(primitive, params):
    """Return ``(subfuns, bind_params)`` for a primitive across jax versions.

    ``subfuns`` are the positional sub-function arguments expected before the
    primitive's operands; ``bind_params`` are the remaining keyword parameters.
    """
    bind_params = primitive.get_bind_params(params)
    if isinstance(bind_params, tuple):
        # jax < 0.10
        subfuns, bind_params = bind_params
    else:
        # jax >= 0.10: subfuns, if any, live inside the params dict
        subfuns = bind_params.pop("subfuns", ())
    return subfuns, bind_params


# jax >= 0.9 renamed pjit_p to jit_p
jit_primitive = getattr(jax._src.pjit, "jit_p", getattr(jax._src.pjit, "pjit_p", None))
