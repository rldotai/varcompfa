"""
The main `__init__.py` file, specifies what is imported and to what namespace
in the `varcompfa` package.
"""
__version__ = "0.7.2"

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
from . import policies
from . import utils

# Frequently used items should appear in global namespace
from .algos import DiscreteQ, TD
from .engine import Agent, LiveExperiment, ReplayExperiment
from .engine import parameters
from .engine.parameters import Constant
from .features import BiasUnit, BinaryVector, UniformTiling, Union
from .policies import DiscreteRandomControl
from .utils import dump_pickle, load_pickle


__all__ = ["algos", "envs", "features", "policies", "misc"]

# Make the current git hash available
__commit__ = utils.current_git_hash()
