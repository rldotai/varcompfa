"""
Utility functions, e.g. plotting and serialization.

Likely contains code that ought to be located elsewhere, but that is probably
best done once the codebase is more mature, if ever.
"""
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

from varcompfa.misc import window_avg


##############################################################################
# Data handling
##############################################################################
def serialize_features(feat):
    """Serializing the features and their implicit dependency graph."""
    def expand_children(ff):
        _params = ff.params
        if 'children' in _params:
            _params['children'] = [expand_children(i) for i in _params['children'] if i]
        return _params
    return expand_children(feat)


##############################################################################
# Plotting
##############################################################################
def plot_phase(trajectory, smoothing=5):
    """Plotting the phase space of a trajectory."""
    trajectory = np.apply_along_axis(window_avg, 0, trajectory, n=smoothing)
    _x, _y = trajectory.T
    plt.plot(_x, _y)
    plt.show()


##############################################################################
# Array manipulation
##############################################################################
def unit(length, ix):
    """A unit vector of size `length` with index `ix` set to 1, other entries 0."""
    ret = np.zeros(length)
    ret[ix] = 1
    return ret

def bitvector(length, *ixs):
    """A binary vector of size `length` with indices `ixs` set to 1, other entries 0."""
    ret = np.zeros(length)
    ret[[ixs]] = 1
    return ret