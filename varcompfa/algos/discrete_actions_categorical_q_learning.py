"""
Categorical Q-Learning with discrete actions for distributional RL.
"""
import numpy as np
from .algo_base import LearningAlgorithm


class DiscreteCategoricalQ(LearningAlgorithm):
    """Discrete Categorical Q Learning (AKA C51) for linear function approximation.
    Implemented following [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)

    Actions are discrete, while states/observations are represented using a feature vector.
    Between `v_min` and `v_max` the possible values of the return are represented by
    `n_atoms` linearly spaced values.
    For control, the algorithm normally selects the best action, but will occasionally
    select a random action (with probability `epsilon`) for exploration.
    """

    def __init__(
        self, n_features, n_actions, n_atoms=51, vmin=-1.0, vmax=1.0, epsilon=1e-2
    ):
        """Initialize the algorithm.
        
        Parameters
        ----------
        n_features : Vector[float]
            The length of the feature vector.
        n_actions : int
            The number of actions available in the environment.
        n_atoms    : int
            Number of values to interpolate between `vmin` and `vmax`.
        vmin       : float
            Lower limit of the possible values.
        vmax       : float
            Upper limit of the possible values. 
        epsilon    : float, optiona
            The probability of selecting a random action instead of 
            the greedy choice when used for control.
        """
        # Record constants
        self.num_features = n_features
        self.num_actions = n_actions
        self.num_atoms = n_atoms
        self.vmin = vmin
        self.vmax = vmax
        self.epsilon = epsilon

        # Computed quantities
        self._width = (vmax - vmin) / (n_atoms - 1)
        self._bins = vmin + np.arange(n_atoms) * self._width
        self._support = (vmin, vmax)

        # Initialize weights and traces
        self.reset()

    def get_distribution(self, x, a=None):
        """Compute the distribution, either over all actions or for a specific one if provided.
        We first compute the unnormalized activations, then normalize them and return the result.

        Parameters
        ----------
        x  : Vector[float]
        a  : int (optional)
        """
        if a is None:
            raw = np.exp(np.dot(self._weights, x))
            tot = np.sum(raw, axis=1)
            ret = np.divide(raw.T, tot)
        else:
            vec = self._weights[a]
            raw = np.exp(np.dot(vec, x))
            tot = np.sum(raw)
            ret = raw / tot
        return ret

    def learn(self, x, a, r, xp, alpha, gm, gm_p, lm):
        """
        Update from experience.

        TODO:
            - Compare with for-loop version

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
        # Update eligibility trace
        self._traces *= gm * lm
        self._traces[a] += x

        # Compute current distribution
        p_cur = self.get_distribution(x, a)

        # Next distributions given `xp` and expected state-action values
        dist_nxt = self.get_distribution(xp)
        qv_nxt = np.dot(self._bins, dist_nxt)

        # Greedy action
        a_nxt = np.argmax(qv_nxt)

        # Distribution for the `xp` and the selected action
        p_nxt = dist_nxt[:, a_nxt]

        # Next values before projection
        T_z = np.clip(r + gm_p * self._bins, self.vmin, self.vmax)

        # Compute projection using Eq. 7 in Distributional RL Paper
        # m_proj = [
        #     np.sum(
        #         [
        #             np.clip(1 - np.abs(T_z[j] - self._bins[i]) / self._width, 0, 1)
        #             * p_nxt[j]
        #             for j in range(self.num_atoms)
        #         ]
        #     )
        #     for i in range(self.num_atoms)
        # ]

        # The above can be vectorized for a significant speedup
        outer_diff = T_z[None, :] - self._bins[:, None]
        scaled = np.abs(outer_diff) / self._width
        clipped = np.clip(1 - scaled, 0, 1)
        weighted = clipped * p_nxt
        m_proj = np.sum(weighted, axis=1)
        
        # Cross entropy loss
        loss = -np.sum(m_proj * np.log(p_cur))

        # Distribution difference
        diff = p_cur - m_proj

        # Compute gradient (with traces) and update weights
        grad = np.einsum("ij,k->ikj", self._traces, diff)
        self._weights -= alpha * grad

        # Gradient of loss and update w/o traces
        # grad = np.outer(p_cur - m_proj, x)
        # self._weights[a] -= alpha*grad

        return {"loss": loss, "grad": grad}

    def get_value(self, x, a=None):
        """Get the value for a given state and action, or if action is left unspecified, just the
        value for the best action in the given state.
        Parameters
        ----------
        x  : Vector[float]
        a  : int (optional)
        """
        dist = self.get_distribution(x, a)
        if a is None:
            return np.max(np.dot(self._bins, dist))
        else:
            return np.dot(self._bins, dist)

    def act(self, x):
        """Select an action following the ε-greedy policy.

        Parameters
        ----------
        x  : Vector[float]
        """
        # Next distributions given `xp` and expected state-action values
        dist_nxt = self.get_distribution(xp)
        qv_nxt = np.dot(self._bins, dist_nxt)

        # Choose action using ε-greedy strategy
        if np.random.random() <= self.epsilon:
            ret = np.random.randint(self.num_actions)
        else:
            ret = np.argmax(qv_nxt)
        return ret

    def act_greedy(self, x):
        """Selects the greedy action given the supplied features.

        Parameters
        ----------
        x  : Vector[float]
        """
        # Next distributions given `xp` and expected state-action values
        dist_nxt = self.get_distribution(xp)
        qv_nxt = np.dot(self._bins, dist_nxt)
        return np.argmax(qv_nxt)

    def compute_loss(self, x, a, r, xp, gm_p):
        """Compute the loss (a cross-entropy variant) for a given transition.

        Parameters
        ----------
        x  : Vector[float]
        a  : int
        r  : float
        xp : Vector[float]
        gm_p : float
        """
        # Compute current distribution
        p_cur = self.get_distribution(x, a)

        # Next distributions given `xp` and expected state-action values
        dist_nxt = self.get_distribution(xp)
        qv_nxt = np.dot(self._bins, dist_nxt)

        # Greedy action
        a_nxt = np.argmax(qv_nxt)

        # Distribution for the `xp` and the selected action
        p_nxt = dist_nxt[:, a_nxt]

        # Next values before projection
        T_z = np.clip(r + gm_p * self._bins, self.vmin, self.vmax)

        # Compute projection using Eq. 7 in Distributional RL Paper
        m_proj = [
            np.sum(
                [
                    np.clip(1 - np.abs(T_z[j] - self._bins[i]) / self._width, 0, 1)
                    * p_nxt[j]
                    for j in range(self.num_atoms)
                ]
            )
            for i in range(self.num_atoms)
        ]

        # Cross entropy loss
        loss = -np.sum(m_proj * np.log(p_cur))
        return loss

    def compute_gradient(self, x, a, r, xp, gm_p):
        """Compute the gradient of the loss (a cross-entropy variant) for the given transition.

        Parameters
        ----------
        x  : Vector[float]
        a  : int
        r  : float
        xp : Vector[float]
        gm_p : float
        """
        # Compute current distribution
        p_cur = self.get_distribution(x, a)

        # Next distributions given `xp` and expected state-action values
        dist_nxt = self.get_distribution(xp)
        qv_nxt = np.dot(self._bins, dist_nxt)

        # Greedy action
        a_nxt = np.argmax(qv_nxt)

        # Distribution for the `xp` and the selected action
        p_nxt = dist_nxt[:, a_nxt]

        # Next values before projection
        T_z = np.clip(r + gm_p * self._bins, self.vmin, self.vmax)

        # Compute projection using Eq. 7 in Distributional RL Paper
        m_proj = [
            np.sum(
                [
                    np.clip(1 - np.abs(T_z[j] - self._bins[i]) / self._width, 0, 1)
                    * p_nxt[j]
                    for j in range(self.num_atoms)
                ]
            )
            for i in range(self.num_atoms)
        ]

        # Gradient of loss
        grad = np.outer(p_cur - m_proj, x)
        return grad

    def start_episode(self):
        """Get ready to start a new episode."""
        self._traces *= 0

    def reset(self):
        """Reset (re-initialize) the weights and traces"""
        self._weights = (
            np.ones((self.num_actions, self.num_atoms, self.num_features))
            / self.num_atoms
        )
        self._traces = np.zeros((self.num_actions, self.num_features))

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
