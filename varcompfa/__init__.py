"""
The main `__init__.py` file, specifies what is imported and to what namespace
in the `varcompfa` package.
"""
__version__ = '0.2.0'

# Get the initial logger
import logging
logger = logging.getLogger(__name__)

# Set up logging from configuration module
from . import configuration
configuration.logger_setup()


from . import algos
from . import analysis
from . import callbacks
from . import engine
from . import envs
from . import features
from . import features as ft
from . import misc
from . import utils

# Frequently used items should appear in global namespace
from .algos import DiscreteQ, TD
from .engine import Agent, LiveExperiment, ReplayExperiment
from .engine import parameters
from .engine.parameters import Constant
from .features import BiasUnit, BinaryVector, UniformTiling, Union
from .policies import DiscreteGreedy, DiscreteSoftmax, DiscreteRandomControl



# TODO: Check that `__all__` really includes everything we wish to import
__all__ = ["algos", "envs", "features", "policies", "misc"]

