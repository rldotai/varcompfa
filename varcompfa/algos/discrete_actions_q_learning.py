"""
Q-Learning using LFA with discrete actions.
"""
import numpy as np
from .algo_base import LearningAlgorithm


class DiscreteQ(LearningAlgorithm):
    """Q-Learning with linear function approximation.

    Actions are assumed to be discrete, while states are represented via a
    feature vector. That is, `Q(s,a) = [〈w, x〉]_a`
    Exploration occurs via an ε-greedy policy.
    """
    def __init__(self, num_features, num_actions, epsilon=5e-2):
        self.num_features = num_features
        self.num_actions = num_actions
        self.epsilon = epsilon

        # Create the weight matrix
        self.w = np.random.randn(self.num_actions, self.num_features)
        # Eligibility traces
        self.z  = np.zeros((self.num_actions, self.num_features))

    def start_episode(self):
        """Get ready to start a new episode."""
        self.z *= 0

    def get_config(self):
        ret = {
            'num_features': self.num_features,
            'num_actions' : self.num_actions,
            'epsilon' : self.epsilon,
            'weights' : self.w.copy(),
            'traces' : self.z.copy(),
        }
        return ret

    @classmethod
    def from_config(cls, config):
        num_features = config['num_features']
        num_actions = config['num_actions']
        epsilon = config['epsilon']
        weights = np.array(config['weights'])
        traces = np.array(config['traces'])
        obj = cls(num_features, num_actions, epsilon)

        # Do some checks to avoid loading obviously wrong configurations
        if (weights.shape != traces.shape):
            raise Exception("Shape of `weights` and `traces` incompatible.")
        obj.w = weights.copy()
        obj.z = traces.copy()
        return obj

    @property
    def trace(self):
        """Return a copy of the current eligibility trace values."""
        return np.copy(self.z)

    def act(self, x):
        """Select an action following the ε-greedy policy.

        Parameters
        ----------
        x  : Vector[float]
        """
        if np.random.random() <= self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(np.dot(self.w, x))
        return action

    def learn(self, x, a, r, xp, alpha, gm, gm_p, lm):
        """Update value function approximation from new experience.

        Parameters
        ----------
        x  : Vector[float]
        a  : int
        r  : float
        xp : Vector[float]
        alpha  : float
        gm  : float
        gm_p : float
        lm  : float
        """
        v = np.dot(self.w[a], x)
        vp = np.max(np.dot(self.w, xp))

        # Compute TD-error
        δ = r + gm_p*vp - v

        # Update eligibility trace
        self.z *= gm*lm
        self.z[a] += x

        # Update Q-values
        self.w += alpha * δ * self.z

        # Return the TD-error, for lack of something more informative
        return δ

    def get_value(self, x, a=None):
        """Get the value for a given state and action, or if action is left unspecified, just the
        value for the best action in the given state.

        Parameters
        ----------
        x  : Vector(float)
        a  : int
        """
        if a is None:
            return np.max(np.dot(self.w, x), axis=0)
        else:
            return np.dot(self.w[a], x)

    def greedy_action(self, x):
        """Return the action that would be taken following the greedy (w/r/t Q-values) policy."""
        return np.argmax(np.dot(self.w, x), axis=0)

    def save_weights(self, fname):
        """Save the weights to a file."""
        np.save(fname, self.w)

    def load_weights(self, fname):
        """Load the weights from a file."""
        self.w = np.load(fname)
