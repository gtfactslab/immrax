import jax
import jax.numpy as jnp
from ..parametope import Parametope
from .polytope import Polytope
from ...inclusion import Interval, interval, icentpert, i2centpert
from jax.tree_util import register_pytree_node_class
from .ellipsoid import Ellipsoid
from functools import partial
from math import sqrt
from itertools import product

@register_pytree_node_class
class Normotope (Parametope) :
    r""" Defines the set

    .. math::
        {x : \|H(x - \ox)\| \leq y}

    where :math:`\|\cdot\|` is a norm, :math:`\ox` is the center, :math:`H` is a shaping matrix, and :math:`y` is the offset.

    Define :math:`h` as the norm in subclasses, and :math:`\mu` as the logarithmic norm associated to :math:`h`.
    """

    def g (self, x) :
        return self.h(jnp.dot(self.alpha, x - self.ox))

    def h (self, z) :
        """The norm associated to the normotope."""
        raise NotImplementedError("Subclasses must implement the h method.")
    
    def hinv (self, y) :
        """An interval overapproximation of the inverse image of y under h."""
        raise NotImplementedError("Subclasses must implement the hinv method.")

    @classmethod
    def induced_norm (cls, A) :
        """Computes the induced norm of A."""
        raise NotImplementedError("Subclasses must implement the induced_norm method.")

    @classmethod
    def logarithmic_norm (cls, A) :
        """The logarithmic norm associated to h."""
        raise NotImplementedError("Subclasses must implement the logarithmic_norm method.")

    @classmethod
    def mu (cls, A) :
        """Alias for the logarithmic norm."""
        return cls.logarithmic_norm(A)

    def plot_projection (self, ax, xi=0, yi=1, rescale=False, **kwargs) :
        """Plot the projection of the normotope onto the xi-yi plane."""
        raise NotImplementedError("Will implement a sampling based thing in the future")

    @property 
    def H (self) :
        return self.alpha

    @classmethod
    def from_parametope(cls, pt:Parametope):
        return Normotope(pt.ox, pt.alpha, pt.y)
    
    def __getitem__ (self, item):
        """Allows indexing into the normotope's parameters."""
        return self.__class__.from_parametope (Parametope(self.ox[item], self.alpha[item], self.y[item]))

    def vec (self) :
        """Vectorizes the normotope into a vector."""
        return jnp.concatenate((self.ox, self.alpha.reshape(-1), jnp.atleast_1d(self.y)))

    @classmethod
    # @partial(jax.jit, static_argnames=('n',))
    def unvec (cls, vec, n=None) :
        """Unvectorizes a vector into a normotope."""
        y = vec[-1]
        N = len(vec) - 1
        if n is None :
            # Assume alpha is nxn, so N = n*n + n = n*(n+1)
            # QF: n^2 + n - N = 0 ==> n = (-1 + sqrt(1 + 4*N)) / 2
            n = int((sqrt(1 + 4*N) - 1) // 2)
        # if alpha is mxn, N = m*n + n = m*(n+1)

        ox = vec[:n]
        alpha = vec[n:N].reshape(-1, n)
        return cls(ox, alpha, y)

    def sample_boundary (self, key:jax.random.PRNGKey, num_samples:int) -> jnp.ndarray:
        """Samples points uniformly from the boundary of the normotope."""
        raise NotImplementedError("Subclasses must implement the sample_boundary method.")

@register_pytree_node_class
class LinfNormotope (Normotope) :
    r""" Defines the set 
    
    .. math::
        {x : \|H(x - \ox)\|_\infty \leq y}

    """
    def h (self, z) :
        """The infinity norm"""
        return jnp.max(jnp.abs(z))

    def hinv (self, y) :
        n = self.alpha.shape[0]
        return icentpert(jnp.zeros(n), y*jnp.ones(n))

    @classmethod 
    def induced_norm (cls, A) :
        r"""Computes the induced :math:`\ell_\infty` norm of A"""
        # Maximum row sum of |A|
        return jnp.max(jnp.sum(jnp.abs(A), axis=1))
    
    @classmethod
    def logarithmic_norm (cls, A) :
        r"""Computes the logarithmic :math:`\ell_\infty` norm of A"""
        # Maximum row sum of A_M (Metzlerized)
        A_M = jnp.where(jnp.eye(A.shape[0], dtype=bool), A, jnp.abs(A))
        return jnp.max(jnp.sum(A_M, axis=1))
    
    def to_polytope (self) -> Polytope :
        n = self.alpha.shape[0]
        return Polytope (self.ox, self.alpha, jnp.ones(2*n)*self.y)
    
    def plot_projection (self, ax, xi=0, yi=1, rescale=False, **kwargs) :
        self.to_polytope().plot_projection(ax, xi, yi, rescale, **kwargs)
    
    @classmethod
    def from_interval (cls, *args) :
        cent, pert = i2centpert(interval(*args))
        return LinfNormotope(cent, jnp.diag(1/pert), 1.)
    
    @classmethod
    def from_normotope (cls, nt:Normotope):
        return LinfNormotope(nt.ox, nt.alpha, nt.y)

@register_pytree_node_class
class L1Normotope (Normotope) :
    r""" Defines the set 
    
    .. math::
        {x : \|H(x - \ox)\|_1 \leq y}

    """
    def h (self, z) :
        """The L1 norm"""
        return jnp.sum(jnp.abs(z))

    def hinv (self, y) :
        n = self.alpha.shape[0]
        return icentpert(jnp.zeros(n), y*jnp.ones(n))

    @classmethod 
    def induced_norm (cls, A) :
        r"""Computes the induced :math:`\ell_1` norm of A"""
        # Maximum row sum of |A|
        return jnp.max(jnp.sum(jnp.abs(A), axis=0))
    
    @classmethod
    def logarithmic_norm (cls, A) :
        r"""Computes the logarithmic :math:`\ell_1` norm of A"""
        # Maximum row sum of A_M (Metzlerized)
        A_M = jnp.where(jnp.eye(A.shape[0], dtype=bool), A, jnp.abs(A))
        return jnp.max(jnp.sum(A_M, axis=0))
    
    def to_polytope (self) -> Polytope :
        # n = self.alpha.shape[0]
        # return Polytope (self.ox, self.alpha, jnp.ones(2*n)*self.y)
        # S is the matrix whose rows are all sign combinations of length n
        n = self.alpha.shape[0]
        S = jnp.array(list(product(*[[1, -1]]*n)))
        return Polytope (self.ox, S@self.alpha, jnp.ones(2*2**n)*self.y)

    def plot_projection (self, ax, xi=0, yi=1, rescale=False, **kwargs) :
        self.to_polytope().plot_projection(ax, xi, yi, rescale, **kwargs)
    
    @classmethod
    def from_interval (cls, *args) :
        cent, pert = i2centpert(interval(*args))
        return LinfNormotope(cent, jnp.diag(1/pert), 1.)
    
    @classmethod
    def from_normotope (cls, nt:Normotope):
        return LinfNormotope(nt.ox, nt.alpha, nt.y)

@register_pytree_node_class
class L2Normotope (Normotope) :
    r""" Defines the set 
    
    .. math::
        {x : \|H(x - \ox)\|_2 \leq y}

    """
    def h (self, z) :
        """The L_2 norm"""
        return jnp.sum(z**2)**0.5

    def hinv (self, y) :
        n = self.alpha.shape[0]
        return icentpert(jnp.zeros(n), y*jnp.ones(n))
        
    def iover (self) :
        n = self.alpha.shape[0]
        Pinv = jnp.linalg.inv(self.alpha.T@self.alpha/self.y**2)
        # Pinv = self.alpha.T @ self.alpha
        return icentpert(jnp.zeros(n), jnp.sqrt(jnp.diag(Pinv))) + self.ox

    @classmethod 
    def induced_norm (cls, A) :
        r"""Computes the induced :math:`\ell_2` norm of A"""
        return jnp.linalg.norm(A, ord=2)
    
    @classmethod
    def logarithmic_norm (cls, A) :
        r"""Computes the :math:`\ell_2` logarithmic norm of A"""
        return jnp.max(jnp.linalg.eigvalsh((A + A.T) / 2))

    
    # def to_polytope (self) -> Polytope :
    #     n = self.alpha.shape[0]
    #     return Polytope (self.ox, self.alpha, jnp.ones(2*n)*self.y)
    
    def plot_projection (self, ax, xi=0, yi=1, rescale=False, **kwargs) :
        # self.to_polytope().plot_projection(ax, xi, yi, rescale, **kwargs)
        Ellipsoid(self.ox, self.alpha, jnp.array([0., self.y**2])).plot_projection(ax, xi, yi, rescale, **kwargs)
    
    @classmethod
    def from_interval (cls, *args) :
        cent, pert = i2centpert(interval(*args))
        rn = jnp.sqrt(len(cent))
        # rn = 1.
        return L2Normotope(cent, jnp.diag(1/(rn*pert)), 1.)
    
    @classmethod
    def from_normotope (cls, nt:Normotope):
        return L2Normotope(nt.ox, nt.alpha, nt.y)

    def sample_boundary (self, key:jax.random.PRNGKey, num_samples:int) -> jnp.ndarray:
        """Samples points uniformly along the boundary of the L2Normotope."""
        gauss = jax.random.multivariate_normal(key, jnp.zeros_like(self.ox), jnp.eye(len(self.ox)), shape=(num_samples,))
        unif_Sn = (gauss / jnp.linalg.norm(gauss, axis=-1, keepdims=True))
        alpha_inv = jnp.linalg.inv(self.alpha/self.y)
        unif_nt = jax.vmap(lambda x : alpha_inv@x + self.ox)(unif_Sn)
        return unif_nt

