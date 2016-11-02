"""
The main `__init__.py` file, specifies what is imported and to what namespace
in the `varcompfa` package.
"""
from . import envs
from . import misc
from .algos import DiscreteQ, TD, TabularQ
from .features import BinaryVector, UniformTiling, Union
from .policies import DiscreteGreedy, DiscreteSoftmax, DiscreteRandomAgent
from .misc import serialize_features


# TODO: Check that `__all__` really includes everything we wish to import
__all__ = ["algos", "envs", "features", "policies", "misc"]
