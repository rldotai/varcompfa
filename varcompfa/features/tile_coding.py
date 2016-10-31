"""
Tile coding implementations, for discretizing a continuous space.
"""
import numpy as np 


class UniformTiling:
    """
    Simple uniform tile coding.

    Maps a point in an n-dimensional space to an integer representing the coordinates of a `tile`
    where that point would lie.
    """
    def __init__(self, space, num_tiles):
        """
        Parameters
        ----------
        space: box_like
            A space (as defined by, e.g., gym.spaces.box.Box in OpenAI gym).
        num_tiles: int or array_like
            The number of tiles to use per-dimension.
            If specified as an integer, that number of tiles is used for each dimension.
        """
        self.space = space
        self._high = space.high
        self._low = space.low
        self.intervals = space.high - space.low
        self.dimensions = np.ndim(self.intervals)

        num_tiles = np.array(num_tiles)
        if num_tiles.ndim == 0:
            # number of tiles the same per dimension
            self.num_tiles = (np.ones(space.shape)*num_tiles).astype(int)
        else:
            assert(num_tiles.shape == space.shape)
            self.num_tiles = num_tiles.astype(int)

        # maximum value of feature vector
        self._max = np.prod(self.num_tiles)

    def __call__(self, obs):
        """Compute the coordinates of the tile for the supplied observation."""
        # get the coordinates of the tile
        # essentially it is the same as the below code, but vectorized (for broadcasting)
        # [int(i//j) for i, j in zip(self.num_tiles*(obs-self._low), self.intervals)]
        coords = np.floor_divide(self.num_tiles*(obs-self._low), self.intervals).astype(int)
        
        # get the tile's index as a flat vector
        index = np.ravel_multi_index(coords.T, self.num_tiles)
        return index

    @property
    def high(self):
        return self._max

    @property
    def low(self):
        return 0


class LayeredTiling:
    """Tile coding with multiple layers. Currently unimplemented."""
    pass


class HashedTiling:
    """Tile coding with multiple layers and hashing. Currently unimplemented."""
    pass