"""
The main `__init__.py` file, specifies what is imported and to what namespace
in the `varcompfa` package.
"""
from . import callbacks
from . import envs
from . import misc
from .algos import DiscreteQ, TD, TabularQ
from .engine import experiment
from .features import BiasUnit, BinaryVector, UniformTiling, Union
from .policies import DiscreteGreedy, DiscreteSoftmax, DiscreteRandomControl
from .misc import serialize_features



# TODO: Check that `__all__` really includes everything we wish to import
__all__ = ["algos", "envs", "features", "policies", "misc"]

# TODO: Set up automatic version consistency
__version__ = '0.0.1'
