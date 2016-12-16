"""
Functions for analyzing and working with data generated by experiments.
"""
import numpy as np
import itertools


##############################################################################
# Data handling
##############################################################################
def calculate_return(rewards, gamma):
    """Calculate return from a list of rewards and a list of gammas.

    Notes
    -----
    The discount parameter `gamma` should be the discount for the *next* state,
    if you are using general value functions.
    This is because (in the episodic setting) the terminal state has a discount
    factor of zero, but the state preceding it has a normal discount factor,
    as does the state following.

    So as we compute G_{t} = R_{t+1} + γ_{t+1}*G_{t+1}
    """
    ret = []
    g = 0
    # Allow for gamma to be specified as a sequence or a constant
    if not hasattr(gamma, '__iter__'):
        gamma = itertools.repeat(gamma)
    # Work backwards through the
    for r, gm in reversed(list(zip(rewards, gamma))):
        g *= gm
        g += r
        ret.append(g)
    # inverse of reverse
    ret.reverse()
    return np.array(ret)

def context_return(ctxlst):
    """Calculate return from a list of contexts."""
    ret = []
    g = 0
    for ctx in reversed(ctxlst):
        reward      = ctx['r']
        discount    = ctx.get('gm_p', 1)
        if ctx['done']:
            discount = 0
        g *= discount
        g += reward
        ret.append(g)
    ret.reverse()
    return ret


def squared_error(a, b):
    """Return the squared difference between sequences `a` and `b`."""
    a = np.array(a)
    b = np.array(b)
    return np.sum((a - b)**2)

def mse(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.mean((a - b)**2)
