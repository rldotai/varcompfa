"""Represent features as binary vectors"""
import numpy as np
from .feature_base import Feature
from .generic_features import Identity


class BinaryVector(Feature):
    """Represent features as a binary vector of a given length.

    e.g., for length 5, the input `[0, 3]` produces the vector `[1, 0, 0, 1, 0]`.
    """
    def __init__(self, length, child=Identity()):
        """
        Parameters
        ----------
        length : int
            The length of the feature vector.
        child : callable
            A callable that acts as a preprocessing step for the feature vector function.
        """
        self._length = length
        self.child = child

    def get_config(self):
        ret = {'length': self._length, 'child': self.child}

    @classmethod
    def from_config(cls, config):
        return cls(config['length'], config['child'])

    def get_config(self):
        return {'length': self.length, 'child': self.child}

    def __call__(self, x):
        indices = self.child(x)
        ret = np.zeros(self._length, dtype=int)
        ret[indices] = 1
        return ret

    def __len__(self):
        return self._length

