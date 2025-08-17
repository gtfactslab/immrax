from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp
from jaxtyping import ArrayLike
from ..inclusion import Interval


@register_pytree_node_class
class Parametope:
    r"""Parametope. Defines the set

    .. math::
        {x : g(\alpha, x - \mathring{x}) <= y}

    """

    ox: ArrayLike  # Center
    alpha: ArrayLike  # Parameters
    y: ArrayLike  # Offset

    def __init__(self, ox, alpha, y):
        self.ox = ox
        self.alpha = alpha
        self.y = y

    def g(self, x: ArrayLike):
        r"""Evaluates the nonlinearity :math:`g(\alpha, x - \mathring{x})` at x

        Parameters
        ----------
        alpha : ArrayLike
            _description_
        x : ArrayLike
            _description_
        """
        raise NotImplementedError("Subclasses must implement the g method.")

    # Always flatten parametope data into (ox, alpha, y)
    def tree_flatten(self):
        return ((self.ox, self.alpha, self.y), type(self).__name__)

    # Override in subclasses to unpack the flattened data
    @classmethod
    def from_parametope(cls, pt: "Parametope"):
        return pt

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls.from_parametope(Parametope(*children))

    @property
    def dtype(self) -> jnp.dtype:
        return self.ox.dtype

    def __str__(self):
        return f"Parametope(ox={self.ox}, alpha={self.alpha}, y={self.y})"


@register_pytree_node_class
class hParametope(Parametope):
    r"""Defines a parametope with the particular structured nonlinearity

    .. math::
        g(\alpha, x - \mathring{x}) = (-h(\alpha (x - \mathring{x})), h(\alpha (x - \mathring{x})))

    and y split into lower and upper bounds y = (ly, uy).
    """

    def h(self, z: ArrayLike):
        """Evaluates the nonlinearity h at z

        Parameters
        ----------
        z : ArrayLike
            Input to the nonlinearity
        """
        pass

    def g(self, x: ArrayLike):
        """Evaluates the nonlinearity g at alpha, x

        Parameters
        ----------
        z : ArrayLike
            Input to the nonlinearity
        """
        return (
            -self.h(jnp.dot(self.alpha, x - self.ox)),
            self.h(jnp.dot(self.alpha, x - self.ox)),
        )

    def hinv(self, iy: Interval):
        """Overapproximating inverse image of the nonlinearity h

        Parameters
        ----------
        iy : ArrayLike
            _description_
        """
        pass

    def k_face(self, k: int) -> Interval:
        """Overapproximate the k-face of the hParametope"""
        pass

    # Override in subclasses to unpack the flattened data
    @classmethod
    def from_parametope(cls, pt: "hParametope"):
        return pt

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls.from_parametope(hParametope(*children))
