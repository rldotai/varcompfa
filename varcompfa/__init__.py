"""
The main `__init__.py` file, specifies what is imported and to what namespace
in the `varcompfa` package.
"""
__version__ = '0.1.1'

from . import algos
from . import callbacks
from . import engine
from . import envs
from . import features
from . import misc
from . import utils

# Frequently used items should appear in global namespace
from .algos import DiscreteQ, TD, TabularQ
from .engine import Agent, PolicyEvaluation
from .engine.parameters import Constant
from .features import BiasUnit, BinaryVector, UniformTiling, Union
from .policies import DiscreteGreedy, DiscreteSoftmax, DiscreteRandomControl



# TODO: Check that `__all__` really includes everything we wish to import
__all__ = ["algos", "envs", "features", "policies", "misc"]

