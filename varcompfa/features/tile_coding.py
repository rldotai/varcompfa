"""
Tile coding implementations, for discretizing a continuous space.
"""
import numpy as np
from .feature_base import Feature
from .generic_features import Identity


class UniformTiling(Feature):
    """
    Simple uniform tile coding.

    Maps a point in an n-dimensional space to an integer representing the coordinates of a `tile`
    where that point would lie.
    """
    NAME = "UniformTiling"
    def __init__(self, space, num_tiles, child=Identity()):
        """
        Parameters
        ----------
        space: box_like
            A space (as defined by, e.g., gym.spaces.box.Box in OpenAI gym).
        num_tiles: int or array_like
            The number of tiles to use per-dimension.
            If specified as an integer, that number of tiles is used for each dimension.
        child: callable
            A callable that acts as a preprocessing step for the feature vector function.
        """
        self.space = space
        self._high = space.high
        self._low = space.low
        self.intervals = space.high - space.low
        self.dimensions = np.ndim(self.intervals)

        num_tiles = np.array(num_tiles)
        if num_tiles.ndim == 0:
            # set number of tiles per dimension to the same value
            self.num_tiles = (np.ones(space.shape)*num_tiles).astype(int)
        else:
            assert(num_tiles.shape == space.shape)
            self.num_tiles = num_tiles.astype(int)

        # set preprocessing step
        self.child = child

        # maximum value of feature vector
        self._max = np.prod(self.num_tiles)

    def get_config(self):
        pass

    @classmethod
    def from_config(cls, config):
        pass

    def __call__(self, obs):
        """Compute the coordinates of the tile for the supplied observation."""
        # get the coordinates of the tile
        # essentially it is the same as the below code, but vectorized (for broadcasting)
        # [int(i//j) for i, j in zip(self.num_tiles*(obs-self._low), self.intervals)]
        coords = np.floor_divide((self.num_tiles-1)*(obs-self._low), self.intervals).astype(int)

        # get the tile's index as a flat vector
        index = np.ravel_multi_index(coords.T, self.num_tiles)
        return index

    @property
    def high(self):
        return self._max

    @property
    def low(self):
        return 0

    @property
    def params(self):
        """The parameters necessary to fully specify the feature."""
        return {
            'name': self.NAME,
            'high': self._high,
            'low' : self._low,
            'num_tiles' : self.num_tiles,
            'children' : [self.child],
        }

    def __len__(self):
        return 1


# class LayeredTiling:
#     """Tile coding with multiple layers. Currently unimplemented."""
#     pass


# class HashedTiling:
#     """Tile coding with multiple layers and hashing. Currently unimplemented."""
#     pass
