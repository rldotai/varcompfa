from .binary_vector import BinaryVector
from .tile_coding import UniformTiling


# TODO: Move this into its own file? Or a collection of similar miscellaneous features?
import numpy as np
class Union:
    """A feature vector created from appending two or more feature vectors together."""
    def __init__(self, *children):
        self.children = children
        self._length = sum(len(child) for child in children)

    def __call__(self, obs):
        """Get the features and concatenate them."""
        return np.hstack((child(obs) for child in self.children))

    @property
    def size(self):
        return self._length

    def __len__(self):
        return self._length