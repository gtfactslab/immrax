import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float, Integer
import immrax as irx
import csv
from functools import partial
from TRAF22 import *

sys = TRAF22()
embsys = trafemb(sys)

print(sys, embsys)
