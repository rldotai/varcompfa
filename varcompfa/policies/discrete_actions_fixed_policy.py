"""
Fixed policies for environments with discrete actions.
"""
import numpy as np
from gym.utils import seeding


class DiscreteFixed:
    """A (possibly stochastic) policy that chooses actions via table-lookup.

    Policies are specified in terms of dictionaries with states/observations as
    keys.
    The values are specified in terms of a sequence of tuples, with each tuple
    having the form `(action, weight)`, such that the weights are nonnegative
    and proportional to the frequency that `action` should be selected in the
    given state.

    Notes
    -----
    Since we're using dicts to specify the policy, it requires the keys to be
    hashable and to completely cover the visitable state-space.
    """
    def __init__(self, policy):
        if not self._validate(policy):
            raise Exception("Invalid policy supplied: %s"%policy)

        self.policy = {}
        for state, values in policy.items():
            choices, probs = zip(*values)
            probs = probs/np.sum(probs)
            self.policy[state] = {}
            self.policy[state]['probs'] = np.array(probs)
            self.policy[state]['choices'] = np.array(choices)

        # Initialize random number generation
        self._seed()

    def act(self, x):
        choices = self.policy[x]['choices']
        probs   = self.policy[x]['probs']
        return self.np_random.choice(choices, p=probs)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def _validate(policy):
        """Determine if the supplied policy is valid."""
        for key, val in policy.items():
            # Check that policy is (action, weight) pairs
            if not all([len(x) == 2 for x in val]):
                return False
            # Check that no weight is negative
            if any([p < 0 for a, p in val]):
                return False
        return True


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

# TODO: Implement this class
class DiscreteSoftmax:
    pass
