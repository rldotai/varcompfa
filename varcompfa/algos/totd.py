"""
True-online TD(λ), sometimes known as temporal difference learning with 'Dutch traces'.

---

0 : [van Seijen, Harm, and Richard S. Sutton. "True Online TD (lambda)."
ICML. Vol. 14. 2014.](http://www.jmlr.org/proceedings/papers/v32/seijen14.pdf)
"""
from .algo_base import LearningAlgorithm


# TODO: TEST THIS CODE
class TOTD(LearningAlgorithm):
    """True-online temporal difference learning with linear function approximation."""
    def __init__(self, n):
        """Initialize the learning algorithm.

        Parameters
        -----------
        n : int
            The number of features, i.e. expected length of the feature vector.

        Attributes
        ----------
        w : Vector[float]
            The current weight vector.
        w_old : Vector[float]
            The previous time-step's weight vector.
        z : Vector[float]
            The array of the eligibility traces.
        """
        self.n      = n
        self.w      = np.zeros(self.n)
        self.w_old  = np.zeros(self.n)
        self.z      = np.zeros(self.n)

    def get_config(self):
        """Return the parameters needed to specify the algorithm's state."""
        # ret = {
        #     'num_features' : self.num_features,
        #     'weights' : self.w.copy(),
        #     'traces': self.z.copy(),
        # }
        return ret

    def get_value(self, x):
        """Get the approximate value for feature vector `x`."""
        return np.dot(self.w, x)

    def learn(self, x, r, xp, alpha, gm, gm_p, lm):
        """Update from new experience, i.e. from a transition `(x,r,xp)`.


        Parameters
        ----------
        x : Vector(float)
            The observation/features from the current timestep.
        r : float
            The reward from the transition.
        xp : Vector(float)
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
        interval [0, 1].
        """
        delta = r + gm_p*np.dot(self.w, xp) - np.dot(self.w, x)
        self.z = gm*lm*self.z + alpha*x - alpha*gm*lm*np.dot(self.z, x)*x
        self.w += delta*self.z + alpha*(np.dot(self.w_old, x) - np.dot(self.w, x))*x
        return delta

    def reset(self):
        """Reset weights, traces, and other parameters."""
        self.w      = np.zeros(self.n)
        self.w_old  = np.zeros(self.n)
        self.z      = np.zeros(self.n)

    def start_episode(self):
        """Get ready to start a new episode."""
        self.z *= 0

    @property
    def trace(self):
        """Return a copy of the current eligibility trace values."""
        return np.copy(self.z)

    # TODO: REMOVE?
    def save_weights(self, fname):
        """Save the weights to a file."""
        np.save(fname, self.w)

    # TODO: REMOVE?
    def load_weights(self, fname):
        """Load the weights from a file."""
        self.w = np.load(fname)
