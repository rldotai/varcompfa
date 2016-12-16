import numpy as np
import itertools
import toolz


def grid_space(space, n=50):
    """Return a sequence of linear interval arrays for a given `space` (of the
    kind provided by OpenAI gym.
    """
    assert(isinstance(space, gym.core.Space))
    return [np.linspace(lo, hi, num=n) for lo, hi in zip(space.low, space.high)]

def map2d(xdata, ydata, func):
    """Given two sequences `xdata` and `ydata`, and a function `func`, return a
    2D-array `grid` whose i-jth entry is `func((xdata[i], ydata[j]))`.

    NOTE:
    -----
    We pass the value to `func` as a single argument.
    """
    xdata = np.squeeze(xdata)
    ydata = np.squeeze(ydata)
    assert(xdata.ndim == ydata.ndim == 1)
    nx = len(xdata)
    ny = len(ydata)
    indices = np.ndindex((nx, ny))
    # Appy function to data and reshape array
    grid = np.reshape([func((xdata[i], ydata[j])) for i, j in indices], (nx, ny))
    return grid

# Define a way of computing mappings over the array
def array_map(arr, func, shape=None):
    """Apply a function to each entry of `arr`, or optionally just over the
    indices provided by `shape`.
    """
    if shape is None:
        shape = np.shape(arr)
    ret = np.empty(shape)
    for ix in np.ndindex(shape):
        ret[ix] = func(arr[ix])
    return ret

def index_map(func, shape):
    """Maps a function over the indices provided in `shape` returning an array
    of that same shape.
    """
    ret = np.empty(shape)
    for ix in np.ndindex(shape):
        ret[ix] = func(ix)
    return ret

##############################################################################
# Helper functions for manipulating data into more workable forms
##############################################################################
def lpluck(ind, seqs, default='__no_default__'):
    """Like `toolz.pluck`, but it returns a list instead of an iterator."""
    return list(toolz.pluck(ind, seqs, default=default))

def lconcat(seqs):
    """Like `toolz.concat`, but it returns a list instead of an iterator."""
    return list(toolz.concat(seqs))

def apluck(ind, seqs, default='__no_default__'):
    """Like `toolz.pluck`, but it returns an array instead of an iterator."""
    return np.array(list(toolz.pluck(ind, seqs, default=default)))

def aconcat(seqs):
    """Like `toolz.concat`, but it returns a array instead of an iterator."""
    return np.array(list(toolz.concat(seqs)))

def window_avg(seq, n=5):
    """Compute the average within a window of size `n` for each entry of `seq`."""
    kernel = np.ones(n)/n
    return np.convolve(seq, kernel, mode='valid')

def subsample(seq, n):
    """Subsample the array-like `seq`, return `n` evenly spaced values from it."""
    arr = np.array(seq)
    indices = np.rint((len(arr)-1)*np.linspace(0, 1, n)).astype(int)
    return arr[indices]

def columnize_dict(dct):
    """Convert a dictionary into a string of columnar values.

    NB: This is a really simple function and probably not as useful in
    general as `json.dumps(dct, indent=2)` or `pprint.pprint(dct)`.
    """
    ret = []
    longest = max([len(str(x)) for x in dct.keys()])
    for key in sorted(dct.keys(), key=lambda x: str(x)):
        format_string = '{0:%d}: {1}'%longest
        ret.append(format_string.format(key, dct[key]))
    return '\n'.join(ret)

def print_dict(dct):
    """Print dictionaries with nicely aligned columns.
    NB: depending on use-case, you might be better off using JSON instead.
    """
    longest = max([len(str(x)) for x in dct.keys()])
    for key in sorted(dct.keys(), key=lambda x: str(x)):
        format_string = '{0:%d}: {1}'%longest
        print(format_string.format(key, dct[key]))
