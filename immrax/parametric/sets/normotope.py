import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, ArrayLike, Float
from matplotlib.axes import Axes

from ...inclusion import Interval, interval, icentpert, i2centpert, mjacM
from ...system import System
from ..parametope import Parametope
from ..embedding import ParametricEmbedding
from .polytope import Polytope
from .ellipsoid import Ellipsoid
from ...utils import get_corners, get_sparse_corners

from functools import partial
from math import sqrt
from itertools import product


@register_pytree_node_class
class Normotope(Parametope):
    r"""Defines the set

    .. math::
        {x : \|H(x - \ox)\| \leq y}

    where :math:`\|\cdot\|` is a norm, :math:`\ox` is the center, :math:`H` is a shaping matrix, and :math:`y` is the offset.

    Define :math:`h` as the norm in subclasses, and :math:`\mu` as the logarithmic norm associated to :math:`h`.
    """

    def __init__(
        self, ox: ArrayLike, alpha: ArrayLike, y: Float, alpha_inv: Array | None = None
    ):
        super().__init__(ox, alpha, y)
        # self.alpha_inv = alpha_inv if alpha_inv is not None else jnp.linalg.inv(alpha)
        self._alpha_inv = alpha_inv

    def g(self, x: Array) -> Float:
        return self.norm(jnp.dot(self.alpha, x - self.ox))

    @property
    def alpha_inv(self) -> Array:
        if self._alpha_inv is None:
            self._alpha_inv = jnp.linalg.inv(self.alpha)
        return self._alpha_inv

    @classmethod
    def norm(self, z: Array) -> Float:
        """The norm defining the normotope."""
        raise NotImplementedError("Subclasses must implement the norm method.")

    @classmethod
    def induced_norm(cls, A: Array) -> Float:
        """Computes the induced norm of A."""
        raise NotImplementedError("Subclasses must implement the induced_norm method.")

    @classmethod
    def logarithmic_norm(cls, A: Array) -> Float:
        """The logarithmic norm associated to h."""
        raise NotImplementedError(
            "Subclasses must implement the logarithmic_norm method."
        )

    @classmethod
    def mu(cls, A: Array) -> Float:
        """Alias for the logarithmic norm."""
        return cls.logarithmic_norm(A)

    @classmethod
    def norm_ball_iover(cls, n: int) -> Interval:
        """An interval overapproximation of the norm ball of radius 1 in R^n."""
        raise NotImplementedError(
            "Subclasses must implement the norm_ball_iover method."
        )

    def iover(self) -> Interval:
        """An interval overapproximation of the normotope, defaults to interval analysis"""
        return (
            interval(self.alpha_inv) @ (self.norm_ball_iover(self.ox.shape[0]) * self.y)
            + self.ox
        )

    def plot_projection(
        self, ax: Axes, xi: int = 0, yi: int = 1, rescale: bool = False, **kwargs
    ) -> None:
        """Plot the projection of the normotope onto the xi-yi plane."""
        raise NotImplementedError(
            "Subclasses must implement the plot_projection method."
        )

    @classmethod
    def from_parametope(cls, pt: Parametope) -> "Normotope":
        return Normotope(pt.ox, pt.alpha, pt.y)

    # def __getitem__ (self, item):
    #     """Allows indexing into the normotope's parameters."""
    #     return self.__class__.from_parametope (Parametope(self.ox[item], self.alpha[item], self.y[item]))

    def vec(self) -> Array:
        """Vectorizes the normotope into a one dimensional array."""
        return jnp.concatenate(
            (self.ox, self.alpha.reshape(-1), jnp.atleast_1d(self.y))
        )

    @classmethod
    def unvec(cls, vec: Array, n: int | None = None) -> "Normotope":
        """Unvectorizes a vector into a normotope."""
        y = vec[-1]
        N = len(vec) - 1
        if n is None:
            # Assume alpha is nxn, so N = n*n + n = n*(n+1)
            # QF: n^2 + n - N = 0 ==> n = (-1 + sqrt(1 + 4*N)) / 2
            n = int((sqrt(1 + 4 * N) - 1) // 2)
        # if alpha is mxn, N = m*n + n = m*(n+1)

        ox = vec[:n]
        alpha = vec[n:N].reshape(-1, n)
        return cls(ox, alpha, y)

    # def sample_boundary (self, key:jax.random.PRNGKey, num_samples:int) -> Array:
    #     """Samples points uniformly from the boundary of the normotope."""
    #     raise NotImplementedError("Subclasses must implement the sample_boundary method.")


class NormotopeEmbedding(ParametricEmbedding):
    def __init__(self, sys: System):
        super().__init__(sys)
        self.Df_x = jax.jacfwd(sys.f, 1)
        self.Mf = mjacM(sys.f)
        self.NT = None
        self.gsc = None

    def _initialize(self, nt0: Normotope, *, no_gsc=False) -> ArrayLike:
        if not isinstance(nt0, Normotope):
            raise ValueError(f"{nt0=} is not a Normotope needed for NormotopeEmbedding")

        self.NT = nt0.__class__

        if not no_gsc:
            ix0 = nt0.iover()
            M = self.Mf(0.0, ix0, centers=((jnp.zeros(1), nt0.ox),))[0][1]
            self.gsc = get_sparse_corners(interval(M))
        else:
            self.gsc = get_corners

    @partial(jax.jit, static_argnames=("perm", "adjoint"))
    def _dynamics(self, t, state, U, *, perm=None, adjoint=True):
        Ut = U.reshape(self.Ush)

        nt = self.NT.unvec(state)
        H = nt.alpha
        Hp = nt.alpha_inv
        y = nt.y

        A = self.Df_x(0.0, nt.ox)

        if adjoint:
            H_dot = -H @ A + Ut
        else:
            H_dot = Ut

        MM = self.Mf(
            t, nt.iover(), centers=((jnp.zeros(1), nt.ox),), permutations=perm
        )[0]
        Mx = MM[1]

        mus = [nt.mu(H_dot @ Hp + H @ M @ Hp) for M in self.gsc(interval(Mx))]
        c = jnp.max(jnp.asarray(mus))

        return jnp.concatenate(
            (self.sys.f(0.0, nt.ox), H_dot.reshape(-1), jnp.atleast_1d(c * y))
        )


@register_pytree_node_class
class LinfNormotope(Normotope):
    r"""Defines the set

    .. math::
        {x : \|H(x - \ox)\|_\infty \leq y}

    """

    def norm(self, z: Array) -> Float:
        """The infinity norm"""
        return jnp.max(jnp.abs(z))

    @classmethod
    def induced_norm(cls, A: Array) -> Float:
        r"""Computes the induced :math:`\ell_\infty` norm of A"""
        # Maximum row sum of |A|
        return jnp.max(jnp.sum(jnp.abs(A), axis=1))

    @classmethod
    def logarithmic_norm(cls, A: Array) -> Float:
        r"""Computes the logarithmic :math:`\ell_\infty` norm of A"""
        # Maximum row sum of A_M (Metzlerized)
        A_M = jnp.where(jnp.eye(A.shape[0], dtype=bool), A, jnp.abs(A))
        return jnp.max(jnp.sum(A_M, axis=1))

    @classmethod
    def norm_ball_iover(cls, n: int) -> Interval:
        return icentpert(jnp.zeros(n), jnp.ones(n))

    def to_polytope(self) -> Polytope:
        n = self.alpha.shape[0]
        return Polytope(self.ox, self.alpha, jnp.ones(2 * n) * self.y)

    def plot_projection(self, ax: Axes, xi=0, yi=1, rescale=False, **kwargs) -> None:
        self.to_polytope().plot_projection(ax, xi, yi, rescale, **kwargs)

    @classmethod
    def from_interval(cls, *args) -> "LinfNormotope":
        cent, pert = i2centpert(interval(*args))
        return LinfNormotope(cent, jnp.diag(1 / pert), 1.0)

    @classmethod
    def from_normotope(cls, nt: Normotope) -> "LinfNormotope":
        return LinfNormotope(nt.ox, nt.alpha, nt.y)


@register_pytree_node_class
class L1Normotope(Normotope):
    r"""Defines the set

    .. math::
        {x : \|H(x - \ox)\|_1 \leq y}

    """

    def norm(self, z: Array) -> Float:
        """The L1 norm"""
        return jnp.sum(jnp.abs(z))

    @classmethod
    def induced_norm(cls, A: Array) -> Float:
        r"""Computes the induced :math:`\ell_1` norm of A"""
        # Maximum row sum of |A|
        return jnp.max(jnp.sum(jnp.abs(A), axis=0))

    @classmethod
    def logarithmic_norm(cls, A: Array) -> Float:
        r"""Computes the logarithmic :math:`\ell_1` norm of A"""
        # Maximum row sum of A_M (Metzlerized)
        A_M = jnp.where(jnp.eye(A.shape[0], dtype=bool), A, jnp.abs(A))
        return jnp.max(jnp.sum(A_M, axis=0))

    @classmethod
    def norm_ball_iover(cls, n: int) -> Array:
        return icentpert(jnp.zeros(n), jnp.ones(n))

    def to_polytope(self) -> Polytope:
        # n = self.alpha.shape[0]
        # return Polytope (self.ox, self.alpha, jnp.ones(2*n)*self.y)
        # S is the matrix whose rows are all sign combinations of length n
        n = self.alpha.shape[0]
        S = jnp.array(list(product(*[[1, -1]] * n)))
        return Polytope(self.ox, S @ self.alpha, jnp.ones(2 * 2**n) * self.y)

    def plot_projection(self, ax: Axes, xi=0, yi=1, rescale=False, **kwargs) -> None:
        self.to_polytope().plot_projection(ax, xi, yi, rescale, **kwargs)

    @classmethod
    def from_interval(cls, *args) -> "L1Normotope":
        cent, pert = i2centpert(interval(*args))
        return L1Normotope(cent, jnp.diag(1 / pert), 1.0)

    @classmethod
    def from_normotope(cls, nt: Normotope) -> "L1Normotope":
        return L1Normotope(nt.ox, nt.alpha, nt.y)


@register_pytree_node_class
class L2Normotope(Normotope):
    r"""Defines the set

    .. math::
        {x : \|H(x - \ox)\|_2 \leq y}

    """

    def norm(self, z: Array) -> Float:
        """The L_2 norm"""
        return jnp.sum(z**2) ** 0.5

    @classmethod
    def norm_ball_iover(cls, n: int) -> Array:
        return icentpert(jnp.zeros(n), jnp.ones(n))

    def iover(self) -> Interval:
        """Tightest interval overapproximation of an L2 normotope."""
        n = self.alpha.shape[0]
        # Pinv = jnp.linalg.inv(self.alpha.T@self.alpha/self.y**2)
        # Pinv = self.alpha.T @ self.alpha
        Pinv = self.alpha_inv @ self.alpha_inv.T * self.y**2
        return icentpert(self.ox, jnp.sqrt(jnp.diag(Pinv)))

    @classmethod
    def induced_norm(cls, A: Array) -> Float:
        r"""Computes the induced :math:`\ell_2` norm of A"""
        return jnp.linalg.norm(A, ord=2)

    @classmethod
    def logarithmic_norm(cls, A: Array) -> Float:
        r"""Computes the :math:`\ell_2` logarithmic norm of A"""
        return jnp.max(jnp.linalg.eigvalsh((A + A.T) / 2))

    # def to_polytope (self) -> Polytope :
    #     n = self.alpha.shape[0]
    #     return Polytope (self.ox, self.alpha, jnp.ones(2*n)*self.y)

    def plot_projection(
        self, ax: Axes, xi: int = 0, yi: int = 1, rescale: bool = False, **kwargs
    ) -> None:
        # self.to_polytope().plot_projection(ax, xi, yi, rescale, **kwargs)
        Ellipsoid(self.ox, self.alpha / self.y, jnp.array([0.0, 1.0])).plot_projection(
            ax, xi, yi, rescale, **kwargs
        )

    @classmethod
    def from_interval(cls, *args) -> "L2Normotope":
        cent, pert = i2centpert(interval(*args))
        rn = jnp.sqrt(len(cent))
        # rn = 1.
        return L2Normotope(cent, jnp.diag(1 / (rn * pert)), 1.0)

    @classmethod
    def from_normotope(cls, nt: Normotope) -> "L2Normotope":
        return L2Normotope(nt.ox, nt.alpha, nt.y)

    def sample_boundary(self, key: jax.random.PRNGKey, num_samples: int) -> Array:
        """Samples points uniformly along the boundary of the L2Normotope."""
        gauss = jax.random.multivariate_normal(
            key, jnp.zeros_like(self.ox), jnp.eye(len(self.ox)), shape=(num_samples,)
        )
        unif_Sn = gauss / jnp.linalg.norm(gauss, axis=-1, keepdims=True)
        alpha_inv = jnp.linalg.inv(self.alpha / self.y)
        unif_nt = jax.vmap(lambda x: alpha_inv @ x + self.ox)(unif_Sn)
        return unif_nt
