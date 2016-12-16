"""
Least-squares temporal difference learning, also known as LSTD(λ).
"""
import numpy as np
from .algo_base import LearningAlgorithm


class LSTD(LearningAlgorithm):
    """Least-squares temporal difference learning.

    Attributes
    ----------
    n : int
        The number of features (and therefore the length of the weight vector).
    z : Vector[float]
        The eligibility trace vector.
    A : Matrix[float]
        A matrix with shape `(n, n)` that acts like a potential matrix.
    b : Vector[float]
        A vector of length `n` that accumulates the trace multiplied by the
        reward over a trajectory.
    """
    def __init__(self, n, epsilon=0):
        """Initialize the learning algorithm.

        Parameters
        -----------
        n : int
            The number of features
        epsilon : float
            To avoid having the `A` matrix be singular, it is sometimes helpful
            to initialize it with the identity matrix multiplied by `epsilon`.
        """
        self.n = n
        self.reset(epsilon)

    def get_config(self):
        """Return the parameters needed to specify the algorithm's state."""
        # ret = {
        #     'num_features' : self.num_features,
        #     'weights' : self.w.copy(),
        #     'traces': self.z.copy(),
        # }
        return ret

    def reset(self, epsilon=0):
        """Reset weights, traces, and other parameters."""
        self.z = np.zeros(self.n)
        self.A = np.eye(self.n) * epsilon
        self.b = np.zeros(self.n)

    def start_episode(self):
        """Get ready to start a new episode."""
        self.z *= 0

    @property
    def trace(self):
        """Return a copy of the current eligibility trace values."""
        return np.copy(self.z)

    @property
    def theta(self):
        """Compute the weight vector via `A^{-1} b`."""
        _theta = np.dot(np.linalg.pinv(self.A), self.b)
        return _theta

    def learn(self, x, r, xp, alpha, gm, gm_p, lm):
        """Update from new experience, i.e. from a transition `(x,r,xp)`.

        Parameters
        ----------
        x : Vector[float]
            The observation/features from the current timestep.
        r : float
            The reward from the transition.
        xp : Vector[float]
            The observation/features from the next timestep.
        gm : float
            Gamma, abbreviated `gm`, the discount factor for the current state.
        gm_p : float
            The discount factor for the next state.
        lm : float
            Lambda, abbreviated `lm`, is the bootstrapping parameter for the
            current timestep.
        """
        self.z = (gm * lm * self.z + x)
        self.A += np.outer(self.z, (x - gm_p*xp))
        self.b += self.z * reward
        return {}
