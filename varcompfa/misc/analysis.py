"""
Functions for analyzing and working with data generated by experiments.
"""
import numpy as np 

import itertools
import toolz

##############################################################################
# Data handling
##############################################################################
def calculate_return(rewards, gamma):
    """Calculate return from a list of rewards and gamma, which may be a sequence or a constant."""
    ret = []
    g = 0
    # Allow for gamma to be specified as a sequence or a constant
    if not hasattr(gamma, '__iter__'):
        gamma = itertools.repeat(gamma)
    # Work backwards through the 
    for r, gm in reversed(list(zip(rewards, gamma))):
        g += r 
        g *= gm
        ret.append(g)
    # inverse of reverse
    ret.reverse()
    return np.array(ret)

def episode_return(epi):
    """Calculate the return from an episode.

    Assumes episode is an iterable of `dict` objects which at the very least
    contain entries for `reward` and `gm` (the current discount parameter).
    """
    ret = []
    g = 0
    # work backwards from the end of the episode
    for step in reversed(epi):
        g *= step['gm']
        g += step['reward']
        ret.append(g)
    # inverse of reverse 
    ret.reverse()
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