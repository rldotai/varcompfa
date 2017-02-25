"""
Temporal difference learning incorporating variance.

Implemented following work by Martha and Adam White in the paper[0] proposing
VTD(λ), an algorithm for adapting the bootstrapping parameter to make a greedy
bias-variance trade off in order to reduce the MSE.

[0]:    A Greedy Approach to Adapting the Trace Parameter; White & White; 2016;
        arxiv:1607.00446v2
"""
import numpy as np
from .algo_base import LearningAlgorithm



class VarianceTD(LearningAlgorithm):
    """TD(λ) with modifications for learning the variance of the return.

    Attributes
    ----------
    num_features : int
        The number of features (and therefore the length of the weight vector).
    w : Vector[float]
        The weight vector for the value function.
    z : Vector[float]
        The eligibility trace vector for the value function estimate.
    w_bar: Vector[float]
        The weight vector for the variance estimate's value functio.
    z_bar: Vector[float]
        The eligibility trace vector for the variance estimate.


    Notes
    -----
    Implemented based on:
        A Greedy Approach to Adapting the Trace Parameter
        White & White; 2016; arxiv:1607.00446v2

    In contrast to the VTD(λ) algorithm as defined in the paper, this algorithm
    does not attempt to adapt lambda, instead it just learns the variance of
    the λ-return.
    """
    def __init__(self, num_features):
        """Initialize the learning algorithm.

        Parameters
        -----------
        num_features : int
            The number of features, i.e. expected length of the feature vector.
        """
        self.num_features = num_features
        self.w      = np.zeros(self.num_features)
        self.z      = np.zeros(self.num_features)
        self.w_bar  = np.zeros(self.num_features)
        self.z_bar  = np.zeros(self.num_features)

    def get_value(self, x):
        """Get the approximate value for feature vector `x`."""
        return np.dot(self.w, x)

    def get_variance(self, x):
        """Get approximate variance of return for feature vector `x`.

        Notes
        -----
        The estimated variance has a floor of zero, as the true variance is
        always positive.
        """
        return np.clip(np.dot(self.w_bar, x) - self.get_value(x)**2, 0, None)

    def get_second_moment(self, x):
        """Get the return's approximate second moment for feature vector `x`."""
        return np.dot(self.w_bar, x)

    def learn(self, x, r, xp, alpha, gm, gm_p, lm, lm_p, lm_bar):
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
            The step-size parameter for updating the value estimate's weights.
        alpha_bar: float
            The step-size parameter for updating the variance estimate weights.
        gm : float
            Gamma, abbreviated `gm`, the discount factor for the current state.
        gm_p : float
            The discount factor for the next state.
        lm : float
            Lambda, abbreviated `lm`, is the value function's bootstrapping
            parameter for the current timestep.
        lm_p: float
            The boostrapping parameter for the next state.
        lm_bar: float
            The variance estimate's bootstrapping parameter for the current
            timestep.

        Notes
        -----
        Features (`x` and `xp`) are assumed to be 1D arrays of length `self.n`.
        Other parameters are floats but are generally expected to be in the
        interval [0, 1]."""
        # Expected return starting from next state
        g_bar = np.dot(self.w, xp)

        # Update value function
        delta = r + gm_p*g_bar - np.dot(self.w, x)
        self.z = x + gm*self.z
        self.w += alpha*delta*self.z

        # Update variance estimate
        r_bar = r**2 + 2*gm_p * g_bar * r
        delta_bar = r_bar + gm_bar*np.dot(xp, self.w_bar) - np.dot(x, self.w_bar)
        self.z_bar = x + (gm_bar*lm_bar)*self.z
        self.w_bar += alpha*delta_bar*self.z_bar
        return {'delta': delta, 'delta_bar': delta_bar}

    def reset(self):
        """Reset weights, traces, and other parameters."""
        self.w      = np.zeros(self.num_features)
        self.z      = np.zeros(self.num_features)
        self.w_bar  = np.zeros(self.num_features)
        self.z_bar  = np.zeros(self.num_features)

    def start_episode(self):
        """Get ready to start a new episode."""
        self.z      *= 0
        self.z_bar  *= 0

    @property
    def traces(self):
        """Return a copy of the current eligibility trace values."""
        return np.copy(self.z)

    def get_config(self):
        """Return the parameters needed to specify the algorithm's state."""
        ret = {
            'num_features' : self.num_features,
            'weights' : self.w.copy(),
            'traces': self.z.copy(),
        }
        return ret

    @classmethod
    def from_config(cls, config):
        """Initialize from a configuration dictionary."""
        pass

# TODO: Documentation
class RewardVarianceTD(LearningAlgorithm):
    """
    Attributes
    ----------
    num_features : int
        The number of features (and therefore the length of the weight vector).
    z : Vector[float]
        The eligibility trace vector.
    w : Vector[float]
        The weight vector.
    h_avg : Vector[float]
        The weight vector for estimating each state's average reward.
    h_var : Vector[float]
        The weight vector for estimating each state's variance in reward.
    """
    def __init__(self, num_features):
        """Initialize the learning algorithm.

        Parameters
        -----------
        num_features : int
            The number of features, i.e. expected length of the feature vector.
        """
        self.num_features = num_features
        self.p = np.float(p)
        self.h_avg = np.zeros(self.num_features)
        self.h_var = np.zeros(self.num_features)
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

        # Estimate average reward for observation and its variance
        r_delta_1 = (r - np.dot(self.h_avg, x))

        # Update reward
        self.h_avg += alpha * x * r_delta_1

        # Welford-style variance update
        r_delta_2 = (r - np.dot(self.h_avg, x))
        self.h_var += alpha * x * (r_delta_1)*(r_delta_2)

        # Compute variance of current reward, use it as meta-reward
        r_bar = np.dot(self.h_var, x)

        # Use it to update the value function
        delta = r_bar + gm_p*np.dot(self.w, xp) - np.dot(self.w, x)
        self.w += alpha*self.z*delta
        return {'delta': delta, 'r_delta_1': r_delta_1, 'r_delta_2': r_delta_2}

    def reset(self):
        """Reset weights, traces, and other parameters."""
        self.h_avg = np.zeros(self.num_features)
        self.h_var = np.zeros(self.num_features)
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


class OldVarianceTD(LearningAlgorithm):
    """TD(λ) with modifications for learning the variance of the return.

    A previous implemetation of the VarianceTD algorithm, which learned the
    value function according to TD(λ), but estimated the variance according
    to TD(1) following the Bellman equation from the White & WHite paper.

    Attributes
    ----------
    num_features : int
        The number of features (and therefore the length of the weight vector).
    w : Vector[float]
        The weight vector for the value function.
    z : Vector[float]
        The eligibility trace vector for the value function estimate.
    w_bar: Vector[float]
        The weight vector for the variance estimate's value functio.
    z_bar: Vector[float]
        The eligibility trace vector for the variance estimate.


    Notes
    -----
    Implemented based on:
        A Greedy Approach to Adapting the Trace Parameter
        White & White; 2016; arxiv:1607.00446v2

    In contrast to the VTD(λ) algorithm as defined in the paper, this algorithm
    does not attempt to adapt lambda, instead it just learns the variance of
    the λ-return.
    """
    def __init__(self, num_features):
        """Initialize the learning algorithm.

        Parameters
        -----------
        num_features : int
            The number of features, i.e. expected length of the feature vector.
        """
        self.num_features = num_features
        self.w      = np.zeros(self.num_features)
        self.z      = np.zeros(self.num_features)
        self.w_bar  = np.zeros(self.num_features)
        self.z_bar  = np.zeros(self.num_features)

    def get_value(self, x):
        """Get the approximate value for feature vector `x`."""
        return np.dot(self.w, x)

    def get_variance(self, x):
        """Get approximate variance of return for feature vector `x`.

        Notes
        -----
        The estimated variance has a floor of zero, as the true variance is
        always positive.
        """
        return np.clip(np.dot(self.w_bar, x) - self.get_value(x)**2, 0, None)

    def get_second_moment(self, x):
        """Get the return's approximate second moment for feature vector `x`."""
        return np.dot(self.w_bar, x)

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
        # Expected return starting from next state
        g_bar = np.dot(self.w, xp)

        # Update value function
        delta = r + gm_p*g_bar - np.dot(self.w, x)
        self.z = x + gm*self.z
        self.w += alpha*delta*self.z

        # Update variance estimate
        r_bar = r**2 + 2*gm_p * g_bar * r
        delta_bar = r_bar + (gm_p**2)*np.dot(xp, self.w_bar) - np.dot(x, self.w_bar)
        self.z_bar = x + (gm**2)*self.z
        self.w_bar += alpha*delta_bar*self.z_bar
        return {'delta': delta, 'delta_bar': delta_bar}

    def reset(self):
        """Reset weights, traces, and other parameters."""
        self.w      = np.zeros(self.num_features)
        self.z      = np.zeros(self.num_features)
        self.w_bar  = np.zeros(self.num_features)
        self.z_bar  = np.zeros(self.num_features)

    def start_episode(self):
        """Get ready to start a new episode."""
        self.z      *= 0
        self.z_bar  *= 0

    @property
    def traces(self):
        """Return a copy of the current eligibility trace values."""
        return np.copy(self.z)

    def get_config(self):
        """Return the parameters needed to specify the algorithm's state."""
        ret = {
            'num_features' : self.num_features,
            'weights' : self.w.copy(),
            'traces': self.z.copy(),
        }
        return ret

    @classmethod
    def from_config(cls, config):
        """Initialize from a configuration dictionary."""
        pass

# TODO: Finish implementing this
class VTD(LearningAlgorithm):
    """VTD(λ) as described in:
        A Greedy Approach to Adapting the Trace Parameter
        White & White; 2016; arxiv:1607.00446v2

    Attributes
    ----------
    num_features : int
        The number of features (and therefore the length of the weight vector).
    z : Vector[float]
        The eligibility trace vector.
    w : Vector[float]
        The weight vector.
    """
    def __init__(self, num_features):
        """Initialize the learning algorithm.

        Parameters
        -----------
        num_features : int
            The number of features, i.e. expected length of the feature vector.
        """
        self.num_features = num_features
        self.w      = np.zeros(self.num_features)
        self.z      = np.zeros(self.num_features)
        self.w_err  = np.zeros(self.num_features)
        self.z_err  = np.zeros(self.num_features)
        self.w_bar  = np.zeros(self.num_features)
        self.z_bar  = np.zeros(self.num_features)

    def get_value(self, x):
        """Get the approximate value for feature vector `x`."""
        return np.dot(self.w, x)

    def get_variance(self, x):
        """Get approximate variance of return for feature vector `x`."""
        return np.max(np.dot(self.w_bar, x) - self.get_value(x)**2, 0)

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
        self.w += alpha*delta*self.z
        return delta

    def reset(self):
        """Reset weights, traces, and other parameters."""
        self.w      = np.zeros(self.num_features)
        self.z      = np.zeros(self.num_features)
        self.w_bar  = np.zeros(self.num_features)
        self.z_bar  = np.zeros(self.num_features)

    def start_episode(self):
        """Get ready to start a new episode."""
        self.z      *= 0
        self.z_bar  *= 0

    @property
    def traces(self):
        """Return a copy of the current eligibility trace values."""
        return np.copy(self.z)

    def get_config(self):
        """Return the parameters needed to specify the algorithm's state."""
        ret = {
            'num_features' : self.num_features,
            'weights' : self.w.copy(),
            'traces': self.z.copy(),
        }
        return ret

    @classmethod
    def from_config(cls, config):
        """Initialize from a configuration dictionary."""
        pass
