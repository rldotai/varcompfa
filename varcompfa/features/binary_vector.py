"""Represent features as binary vectors"""
import numpy as np 


class BinaryVector:
    """Represent features as a binary vector of a given length.

    e.g., for length 5, the input `[0, 3]` produces the vector `[1, 0, 0, 1, 0]`.
    """
    def __init__(self, length, preprocess=None):
        if preprocess is None:
            self.preprocess = lambda x: np.array(x).astype(int)
        else:
            self.preprocess = preprocess
        self._length = length

    def __call__(self, indices):
        indices = self.preprocess(indices)
        ret = np.zeros(self._length, dtype=int)
        ret[indices] = 1
        return ret

    @property 
    def size(self):
        return self._length

    def __len__(self):
        return self._length