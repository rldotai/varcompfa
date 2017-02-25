"""
An attempt to create a learning algorithm that learns the *quantiles* of the
return, as opposed to its expected value.

# In pseudo-LaTeX, the update equations look like:

#     δ_{t}   = R_{t+1} + γ_{t+1} w_{t}^T x_{t+1} - w_{t}^{T} x_{t}
#     e_{t}   = ρ_{t} (λ_{t} γ_{t} e_{t-1} + x_{t})
#     w_{t+1} = w_{t} + α δ_{t} e_{t}

Where:
    - δ refers to the temporal difference error
    - γ is the discount parameter
    - λ is the bootstrapping parameter
    - α is the stepsize parameter
    - w is the weight vector
    - e is the eligibility trace
    - x and r are feature vectors and rewards respectively

"""
import numpy as np
from .algo_base import LearningAlgorithm


class QuantileTD(LearningAlgorithm):
    """
    Attributes
    ----------
    num_features : int
        The number of features (and therefore the length of the weight vector).
    z : Vector[float]
        The eligibility trace vector.
    w : Vector[float]
        The weight vector.
    """
    def __init__(self, num_features, p):
        """Initialize the learning algorithm.

        Parameters
        -----------
        num_features : int
            The number of features, i.e. expected length of the feature vector.

        p: float
            A float in [0, 1], specifying the quantile to estimate.
        """
        self.num_features = num_features
        self.p = np.float(p)
        self.h = np.zeros(self.num_features)
        self.w = np.zeros(self.num_features)
        self.z = np.zeros(self.num_features)

    def get_value(self, x):
        """Get the approximate value for feature vector `x`."""
        return np.dot(self.w, x)

    def learn(self, x, r, xp, alpha, gm, gm_p, lm):
        """Update from new experience, i.e., a transition `(x, r, xp)`.

        Parameters
        ----------
        x : Vector[float]
            The observation/features from the current timestep.
        r : float
            The reward from the transition.
        xp : Vector[float]
            The observation/features from the next timestep.
        alpha : float
            The step-size parameter for updating the weight vector.
        gm : float
            Gamma, abbreviated `gm`, the discount factor for the current state.
        gm_p : float
            The discount factor for the next state.
        lm : float
            Lambda, abbreviated `lm`, is the bootstrapping parameter for the
            current timestep.

        Notes
        -----
        Features (`x` and `xp`) are assumed to be 1D arrays of length `self.n`.
        Other parameters are floats but are generally expected to be in the
        interval [0, 1]."""
        # Update traces
        self.z = x + gm*lm*self.z

        # Estimate quantile of current reward
        r_hat = np.dot(self.h, x)
        q_delta = np.sign(r - r_hat) + 2*self.p - 1
        self.h += alpha*x*q_delta
        # use traces? # should r_hat be updated?

        # Use it to update the value function
        delta = r_hat + gm_p*np.dot(self.w, xp) - np.dot(self.w, x)
        self.w += alpha*self.z*delta
        return {'delta': delta, 'q_delta': q_delta}

    def reset(self):
        """Reset weights, traces, and other parameters."""
        self.h = np.zeros(self.num_features)
        self.w = np.zeros(self.num_features)
        self.z = np.zeros(self.num_features)

    def start_episode(self):
        """Get ready to start a new episode."""
        self.z *= 0

    @property
    def traces(self):
        """Return a copy of the current eligibility trace values."""
        return np.copy(self.z)

    @property
    def weights(self):
        return self.w.copy()


class QuantileTD2(LearningAlgorithm):
    """
    Attributes
    ----------
    num_features : int
        The number of features (and therefore the length of the weight vector).
    z : Vector[float]
        The eligibility trace vector.
    w : Vector[float]
        The weight vector.
    """
    def __init__(self, num_features, q):
        """Initialize the learning algorithm.

        Parameters
        -----------
        num_features : int
            The number of features, i.e. expected length of the feature vector.

        q: float
            A float in [0, 1], specifying the quantile to estimate.
        """
        self.num_features = num_features
        self.q = np.float(q)
        self.w = np.zeros(self.num_features)
        self.z = np.zeros(self.num_features)

    def get_value(self, x):
        """Get the approximate value for feature vector `x`."""
        return np.dot(self.w, x)

    def learn(self, x, r, xp, alpha, gm, gm_p, lm):
        """Update from new experience, i.e., a transition `(x, r, xp)`.

        Parameters
        ----------
        x : Vector[float]
            The observation/features from the current timestep.
        r : float
            The reward from the transition.
        xp : Vector[float]
            The observation/features from the next timestep.
        alpha : float
            The step-size parameter for updating the weight vector.
        gm : float
            Gamma, abbreviated `gm`, the discount factor for the current state.
        gm_p : float
            The discount factor for the next state.
        lm : float
            Lambda, abbreviated `lm`, is the bootstrapping parameter for the
            current timestep.

        Notes
        -----
        Features (`x` and `xp`) are assumed to be 1D arrays of length `self.n`.
        Other parameters are floats but are generally expected to be in the
        interval [0, 1]."""
        delta = r + gm_p*np.dot(self.w, xp) - np.dot(self.w, x)
        self.z = x + gm*lm*self.z
        self.w += alpha*self.z*(np.sign(delta) + 2*self.q - 1)
        return {'delta': delta}

    def reset(self):
        """Reset weights, traces, and other parameters."""
        self.w = np.zeros(self.num_features)
        self.z = np.zeros(self.num_features)

    def start_episode(self):
        """Get ready to start a new episode."""
        self.z *= 0

    @property
    def traces(self):
        """Return a copy of the current eligibility trace values."""
        return np.copy(self.z)

    @property
    def weights(self):
        return self.w.copy()
