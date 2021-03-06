from .feature_base import Feature
from .binary_vector import BinaryVector
from .generic_features import BiasUnit, Map, Union
from .rbf import RBF, NRBF
from .tile_coding import BinaryTiling, UniformTiling

# Currently grouping ad-hoc/one-off features together
from .special import BoyanFeatures
