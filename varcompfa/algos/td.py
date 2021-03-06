"""
Temporal difference learning, AKA TD(λ), an on-policy linear-time online
learning algorithm

This is one of the foundational algorithms of reinforcement learning.
See the book "Reinforcement Learning: An Introduction" by Sutton and Barto for
a full introduction, in particular Chapter 7.

The algorithm is given in pseudocode on Rich Sutton's website[0].

It is known to converge in the on-policy setting under mild technical conditions,
although the fixed-point it converges to changes depending on the bootstrapping
parameter, λ.
For λ=0 we bootstrap the value of each state from the reward and the value of its
successor; this tends to converge quickly but its solution may be different from
the true value function (and its least-squares approximation).
With λ=1 we get effectively an online, every-visit Monte-Carlo method for
estimating state value which may be more accurate, but tends to have a higher
variance.

In pseudo-LaTeX, the update equations look like:

    δ_{t}   = R_{t+1} + γ_{t+1} w_{t}^T x_{t+1} - w_{t}^{T} x_{t}
    e_{t}   = ρ_{t} (λ_{t} γ_{t} e_{t-1} + x_{t})
    w_{t+1} = w_{t} + α δ_{t} e_{t}

Where:
    - δ refers to the temporal difference error
    - γ is the discount parameter
    - λ is the bootstrapping parameter
    - α is the stepsize parameter
    - w is the weight vector
    - e is the eligibility trace
    - x and r are feature vectors and rewards respectively

This version of TD(λ) is an on-policy algorithm, so it doesn't respond
well to updates from trajectories generated via policies other than the one
it is currently evaluating.
There are a slew of modifications that can allow for off-policy evaluation,
for example: GTD(λ), ETD(λ), and other importance sampling methods.
Here, we employ accumulating traces (vs. replacing traces or dutch traces),
although modifying the code for different traces is straightforward.

---

0: https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node75.html
"""
import numpy as np
from .algo_base import LearningAlgorithm


class TD(LearningAlgorithm):
    """Temporal Difference Learning or TD(λ) with accumulating traces.

    The version implemented here uses general value functions (GVFs), meaning that
    the discount factor, γ, and the bootstrapping factor, λ, may be functions
    of state.
    If that doesn't seem germane to your problem, just use a constant value for them.

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
        self.w = np.zeros(self.num_features)
        self.z = np.zeros(self.num_features)

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
        num_features = config['num_features']
        weights = np.ravel(config['weights'])
        traces = np.ravel(config['traces'])
        obj = cls(num_features)
        if num_features != len(weights) or num_features != len(traces):
            raise Exception("Invalid configuration, mismatched array lengths")
        obj.w = weights
        obj.z = traces
        return obj

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
        self.w += alpha*delta*self.z
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

