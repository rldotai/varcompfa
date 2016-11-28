from .binary_vector import BinaryVector
from .tile_coding import UniformTiling


# TODO: Move this into its own file? Or a collection of similar miscellaneous features?
import numpy as np
from .base import Feature

class BiasUnit(Feature):
    """A feature vector of length one and value one, that acts as a bias."""
    NAME = "Bias"
    def __init__(self):
        pass

    def __call__(self, obs):
        return np.array([1])

    def __len__(self):
        return 1

    @property
    def params(self):
        return {
            "name" : self.NAME,
            "children": list(),
        }


class Union(Feature):
    """A feature vector created from appending two or more feature vectors together."""
    NAME = "Union"
    def __init__(self, *children):
        self.children = children
        self._length = sum(len(child) for child in children)

    def __call__(self, obs):
        """Get the features and concatenate them."""
        return np.hstack((child(obs) for child in self.children))

    def __len__(self):
        return self._length

    @property
    def params(self):
        return {
            "name" : self.NAME,
            "children": [child for child in self.children],
        }

class Identity(Feature):
    # TODO: Implement this or rework the whole thing
    pass
