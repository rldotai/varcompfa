"""
Random agents, useful for various sanity checks.
"""
import numpy as np


class DiscreteRandomAgent:
    """An agent that chooses randomly from (discrete) actions."""
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, *args):
        """Select an action."""
        return np.random.randint(self.num_actions)


class ContinuousRandomAgent:
    """An agent that chooses randomly from the (uniform) distribution of the 
    space enclosed by `bounds`.

    NB: UNTESTED.
    """
    def __init__(self, bounds):
        self.bounds = bounds
        self.ndim = len(bounds)
        self.low, self.high = self.bounds

    def act(self, *args):
        """Select an action."""
        return np.random.sample(self.ndim)*(self.high - self.low) + self.low