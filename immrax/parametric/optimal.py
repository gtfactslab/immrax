import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Integer, Float
from .param_reach import ParametopeEmbedding
from .sets.normotope import Normotope
from ..inclusion import mjacM, interval
from ..utils import get_corners
