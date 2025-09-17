import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Integer, Float
from .param_reach import ParametopeEmbedding
from .sets.normotope import Normotope
from ..inclusion import mjacM, interval
from ..utils import get_corners

class NormotopeEmbedding (ParametopeEmbedding) :
    def __init__ (self, sys) :
        super().__init__(sys)
        self.Df_x = jax.jacfwd(sys.f, 1)
        self.Mf = mjacM(sys.f)
        self.NT = None
    
    def _initialize (self, nt0:Normotope) -> ArrayLike :
        if not isinstance(nt0, Normotope) :
            raise ValueError(f"{nt0=} is not a Normotope needed for NormotopeEmbedding")
        self.NT = nt0.__class__
        self.Ush = nt0.alpha.shape

    def _dynamics (self, t, state, U, *, perm=None, closed=None, adjoint=True, mask=True) : 
        Ut = U.reshape(self.Ush)
        
        nt = self.NT.unvec(state)
        H = nt.alpha
        Hp = jnp.linalg.inv(H)
        y = nt.y

        A = self.Df_x(0., nt.ox)
        H_dot = -H@A + Ut

        MM = self.Mf (0., interval(Hp)@nt.hinv(nt.y) + nt.ox, 
                        centers=((jnp.zeros(1),nt.ox),), permutations=perm)[0]
        Mx = MM[1]

        mus = [nt.mu(H_dot@Hp + H@M@Hp) for M in get_corners(interval(Mx))]
        c = jnp.max(jnp.asarray(mus))

        return jnp.concatenate((self.sys.f(0., nt.ox), H_dot.reshape(-1), jnp.atleast_1d(c*y)))


    # def optimizer_iterate (self, )
