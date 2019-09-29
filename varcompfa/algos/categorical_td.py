"""
Categorical Temporal Difference Learning for distributional RL. 
"""
import numpy as np
from .algo_base import LearningAlgorithm


class CategoricalTD(LearningAlgorithm):
    """Categorical TD Learning for linear function approximation.
    Implemented with reference to [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
    
    States/observations are represented using a feature vector.
    Between `v_min` and `v_max` the possible values of the return are represented by 
    `n_atoms` linearly spaced values.
    
    NB: I am not aware of existing implementations for this particular algorithm, 
    but it is easy to come up with by modifying existing distributional RL implementations.
    
    Attributes
    ----------
    num_features : int
        The number of features (and therefore the length of the weight vector).
    traces : Vector[float]
        The eligibility trace vector.
    weights : Vector[float]
        The weight vector.
    """

    def __init__(self, n_features, n_atoms=51, vmin=-1.0, vmax=1.0):
        """Initialize the algorithm.
        
        Parameters
        ----------
        n_features : Vector[float]
            The length of the feature vector.
        n_atoms    : int
            Number of values to interpolate between `vmin` and `vmax`.
        vmin       : float
            Lower limit of the possible values.
        vmax       : float
            Upper limit of the possible values. 
        """
        # Record constants
        self.num_features = n_features
        self.num_atoms = n_atoms
        self.vmin = vmin
        self.vmax = vmax

        # Computed quantities
        self._width = (vmax - vmin) / (n_atoms - 1)
        self._bins = vmin + np.arange(n_atoms)*self._width
        self._support = (vmin, vmax)

        # Initialize weights and traces
        self.reset()

    def get_distribution(self, x):
        """Compute the distribution for a given state/observation.
        We first compute the unnormalized activations, then normalize them and return the result.
        
        Parameters
        ----------
        x  : Vector[float]
        """
        raw = np.exp(np.dot(self._weights, x))
        tot = np.sum(raw)
        ret = raw/tot
        return ret         
    
    def learn(self, x, r, xp, alpha, gm, gm_p, lm):
        """
        Update from experience.
        
        TODO: 
            - Compare with for-loop version

        Parameters
        ----------
        x  : Vector[float]
        r  : float
        xp : Vector[float]
        alpha  : float
        gm  : float
        gm_p : float
        lm  : float
        """
        # Update eligibility trace
        self._traces *= gm*lm
        self._traces += x
        
        # Compute current distribution
        p_cur = self.get_distribution(x)
        
        # Next distributions given `xp` and expected state-action values
        p_nxt = self.get_distribution(xp)

        # Next values before projection
        T_z = np.clip(r + gm_p*self._bins, self.vmin, self.vmax)

        # Compute projection using Eq. 7 in Distributional RL Paper
        m_proj = [
            np.sum([
                np.clip(1 - np.abs(T_z[j] - self._bins[i])/self._width, 0, 1) * p_nxt[j]
                for j in range(self.num_atoms)
            ])
            for i in range(self.num_atoms)
        ]
        
        # Cross entropy loss
        loss = -np.sum(m_proj * np.log(p_cur))
        
        # Distribution difference
        diff = p_cur - m_proj

        # Gradient of loss and update w/o traces
        grad = np.outer(p_cur - m_proj, self._traces)
        self._weights -= alpha*grad
        
        return {'loss': loss, 'grad': grad}

    def get_value(self, x, a=None):
        """Get the value for a given state and action, or if action is left unspecified, just the
        value for the best action in the given state.
        Parameters
        ----------
        x  : Vector[float]
        a  : int (optional)
        """
        dist = self.get_distribution(x)
        return np.dot(self._bins, dist)

    
    def compute_loss(self, x, r, xp, gm_p):
        """Compute the loss (a cross-entropy variant) for a given transition.
        
        Parameters
        ----------
        x  : Vector[float]
        r  : float
        xp : Vector[float]
        gm_p : float
        """
        pass 
    
    def compute_gradient(self, x, r, xp, gm_p):
        """Compute the gradient of the loss (a cross-entropy variant) for the given transition.
        
        Parameters
        ----------
        x  : Vector[float]
        r  : float
        xp : Vector[float]
        gm_p : float
        """
        pass 
    
    def start_episode(self):
        """Get ready to start a new episode."""
        self._traces *= 0

    def reset(self):
        """Reset (re-initialize) the weights and traces"""
        self._weights = np.ones((self.num_atoms, self.num_features))/self.num_atoms
        self._traces = np.zeros((self.num_features))

    def save_weights(self, fname):
        """Save weights to a file."""
        np.save(fname, self._weights)
    
    def load_weights(self, fname):
        """Load weights from a file."""
        self._weights = np.load(fname)

    @property
    def traces(self):
        """Return a copy of the current eligibility trace values."""
        return np.copy(self._traces)

    @property
    def weights(self):
        """Return a copy of the current weights"""
        return np.copy(self._weights)
