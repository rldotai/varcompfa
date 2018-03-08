"""Generic features, ones that are relatively simple and common."""
import numpy as np
from .feature_base import Feature


class BiasUnit(Feature):
    """A feature vector of length one and value one, that acts as a bias."""
    def __init__(self):
        pass

    def __call__(self, obs):
        return np.array([1])

    def __len__(self):
        return 1

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls()


class Union(Feature):
    """A feature vector created from appending two or more feature vectors together."""
    def __init__(self, *children):
        self.children = children
        self._length = sum(len(child) for child in children)

    def __call__(self, obs):
        """Get the features and concatenate them."""
        return np.hstack([child(obs) for child in self.children])

    def __len__(self):
        return self._length

    def get_config(self):
        ret = {'children' : [child for child in self.children]}
        return ret

    def from_config(cls, config):
        return cls(**config['children'])


class Identity(Feature):
    """A feature that returns whatever input it is given."""
    def __init__(self):
        pass

    def __call__(self, x):
        return x

    def __len__(self):
        # TODO: Not sure about using `None` for unspecified lengths
        return (None,)

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls()


class Noise(Feature):
    """A feature that adds Gaussian noise to the feature it wraps."""
    def __init__(self, child, scale):
        self.child = child
        self.scale = scale
        self._length = len(child)

    def __len__(self):
        return self._length

    def __call__(self, obs):
        return self.child(obs) + np.random.normal(loc=0.0, scale=self.scale, size=self._length)

class Map(Feature):
    """A dictionary mapping feature, initialized with a `dict` where the keys
    are states and the values are arrays corresponding to the feature vectors.
    """
    __mapping = {
    }
    def __init__(self, dct):
        self.mapping = {k: np.array(v) for k, v in dct.items()}
        lengths = [len(v) for v in self.mapping.values()]
        self._length = lengths[0]
        assert(all([i == self._length for i in lengths]))
        
    def __call__(self, obs):
        """Return the feature vector associated with `obs`."""
        return self.mapping[obs]

    def __len__(self):
        return self._length
    

# class WrappedFunction(Feature):
#     """A feature created by wrapping a function, with additional provided
#     information to ensure that the resulting feature can be combined with other
#     features in a sensible way.
#     """
