"""
A policy that chooses the same action for every state, useful for evaluating
Markov processes in the same way as MDPs.
"""
import numpy as np


class ConstantPolicy:
    def __init__(self, action=0):
        """Initialize the policy with the given weights."""
        self.action = action

    def act(self, x):
        """Select an a action according to the policy.

        Parameters
        ----------
        x : array_like
            The feature vector for the state in which to act.
        """
        return self.action
