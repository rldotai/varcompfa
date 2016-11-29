from .feature_base import Feature
from .binary_vector import BinaryVector
from .generic_features import BiasUnit, Union
from .tile_coding import UniformTiling


# def serialize_features(feat):
#     """Serializing the features and their implicit dependency graph."""
#     def expand_children(ff):
#         _params = ff.params
#         if 'children' in _params:
#             _params['children'] = [expand_children(i) for i in _params['children'] if i]
#         return _params
#     return expand_children(feat)

