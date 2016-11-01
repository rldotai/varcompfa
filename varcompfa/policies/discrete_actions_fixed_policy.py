"""
Fixed policies for environments with discrete actions.
"""
import numpy as np 


class DiscreteGreedy:
    def __init__(self, weights):
        """Initialize the policy with the given weights."""
        self._weights = np.copy(weights)

    @classmethod
    def from_file(cls, fname):
        """Initialize the policy from a file containing the weights."""
        weights = np.load(fname)
        return cls(weights) 

    def act(self, x):
        """Select an a action according to the policy.

        Parameters
        ----------
        x : array_like
            The feature vector for the state in which to act.
        """
        return np.argmax(np.dot(self.weights, x))

class DiscreteSoftmax:
    pass
