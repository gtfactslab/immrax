from . import inclusion as inclusion
from .inclusion import *

from . import system as system
from .system import *

from . import control as control
from .control import *

import sys
import os

jax_verify_path = os.path.join(
    os.path.dirname(__file__),
    "_vendor",
    "jax_verify",
)
try:
    sys.path.insert(0, jax_verify_path)
    import jax_verify

    from . import neural as neural
    from .neural import *

    from . import parametric as parametric
    from .parametric import *
except ImportError:
    print(
        "WARN (immrax): Failed to import jax_verify. Some neural and parametric features may not be available."
    )
    print("WARN (immrax): Did you remember to initialize all git submodules?")
finally:
    sys.path.remove(jax_verify_path)

from . import embedding as embedding
from .embedding import *


from . import refinement as refinement
from . import utils as utils
