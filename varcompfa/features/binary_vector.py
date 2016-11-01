"""Represent features as binary vectors"""
import numpy as np 
from .base import Feature


class BinaryVector(Feature):
    """Represent features as a binary vector of a given length.

    e.g., for length 5, the input `[0, 3]` produces the vector `[1, 0, 0, 1, 0]`.
    """
    NAME = "BinaryVector"
    def __init__(self, length, child=None):
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

    def __call__(self, indices):
        if self.child:
            indices = self.child(indices)
        ret = np.zeros(self._length, dtype=int)
        ret[indices] = 1
        return ret

    @property 
    def params(self):
        """The parameters necessary to fully specify the feature."""
        return {
            'name' : self.NAME,
            'length': self._length,
            'children': [self.child],
        } 

    def __len__(self):
        return self._length
