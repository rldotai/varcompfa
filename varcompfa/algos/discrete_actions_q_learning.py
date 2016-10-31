"""
Q-Learning using LFA with discrete actions.
"""
import numpy as np


class DiscreteQ:
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

    def act(self, x):
        """Select an action following the ε-greedy policy."""
        if np.random.random() <= self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(np.dot(self.w, x))
        return action

    def learn(self, x, a, r, xp, α, γ, λ):
        """Update value function approximation from new experience.

        Parameters
        ----------
        x  : array_like
        a  : int
        r  : float
        xp : array_like
        α  : float
        γ  : float 
        λ  : float 
        """
        v = np.dot(self.w[a], x)
        vp = np.max(np.dot(self.w, xp))

        # Compute TD-error
        δ = r + γ*vp - v

        # Update eligibility trace 
        self.z *= γ*λ
        self.z[a] += x

        # Update Q-values
        self.w += α * δ * self.z

        # Return the TD-error, for lack of something more informative
        return δ

    def get_value(self, x, a=None):
        """Get the value for a given state and action, or if action is left unspecified, just the 
        value for the best action in the given state.
        
        Parameters
        ----------
        x  : array_like
        a  : int
        """
        if a is None:
            return np.max(np.dot(self.w, x))
        else:
            return np.dot(self.w[a], x)

    def greedy_action(self, x):
        """Return the action that would be taken following the greedy (w/r/t Q-values) policy."""
        return np.argmax(np.dot(self.w, x))