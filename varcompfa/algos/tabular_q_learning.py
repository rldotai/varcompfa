"""
Tabular Q-Learning (i.e., with table lookup), an off-policy control algorithm.
"""
import numpy as np
from .algo_base import LearningAlgorithm


class TabularQ(LearningAlgorithm):
    """Q-Learning with table lookup"""
    def __init__(self, num_states, num_actions, epsilon=5e-2):
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon

        # Create the Q-value table
        self.qv = np.random.randn(self.num_states, self.num_actions)
        # Eligibility traces
        self.z  = np.zeros((self.num_states, self.num_actions))

    def get_config(self):
        """Return the parameters needed to specify the algorithm's state."""
        # ret = {
        #     'num_features' : self.num_features,
        #     'weights' : self.w.copy(),
        #     'traces': self.z.copy(),
        # }
        return ret

    def start_episode(self):
        """Get ready to start a new episode."""
        self.z *= 0

    def act(self, s):
        """Select an action following the ε-greedy policy."""
        # Explore with probability epsilon, otherwise go with most valued action
        if np.random.random() <= self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.qv[s])
        return action

    def learn(self, s, a, r, sp, alpha, gm, lm):
        """Update value function approximation from new experience.

        Parameters
        ----------
        x  : int
        a  : int
        r  : float
        xp : int
        alpha  : float
        gm  : float
        lm  : float
        """
        # Get current state-action value and maximum value of next state
        v  = self.qv[s, a]
        vp = np.max(self.qv[sp])

        # Compute TD-error
        δ = r + gm*vp - v

        # Update eligibility trace
        self.z *= gm*lm
        self.z[s,a] += 1

        # Update Q-values
        self.qv += alpha * δ * self.z

        # Return the TD-error, for lack of something more informative
        return {'delta': δ}

    def get_value(self, s, a=None):
        if a is None:
            return np.max(self.qv[s])
        else:
            return self.qv[s,a]

    def greedy_action(self, s):
        return np.argmax(self.qv[s])

    @property
    def trace(self):
        """Return a copy of the current eligibility trace values."""
        return np.copy(self.z)

    def save_weights(self, fname):
        """Save the weights to a file."""
        np.save(fname, self.qv)

    def load_weights(self, fname):
        """Load the weights from a file."""
        self.qv = np.load(fname)
