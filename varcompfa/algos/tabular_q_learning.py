"""
Tabular Q-Learning (i.e., with table lookup), an off-policy control algorithm.
"""
import numpy as np


class TabularQ:
    """Q-Learning with table lookup"""
    def __init__(self, num_states, num_actions, epsilon=5e-2):
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon

        # Create the Q-value table
        self.qv = np.random.randn(self.num_states, self.num_actions)
        # Eligibility traces
        self.z  = np.zeros((self.num_states, self.num_actions))
    
    def act(self, s):
        """Select an action following the ε-greedy policy."""
        # Explore with probability epsilon, otherwise go with most valued action
        if np.random.random() <= self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.qv[s])
        return action

    def learn(self, s, a, r, sp, α, γ, λ):
        """Update value function approximation from new experience.

        Parameters
        ----------
        x  : int
        a  : int
        r  : float
        xp : int
        α  : float
        γ  : float 
        λ  : float 
        """
        # Get current state-action value and maximum value of next state
        v  = self.qv[s, a]
        vp = np.max(self.qv[sp])
        
        # Compute TD-error
        δ = r + γ*vp - v

        # Update eligibility trace 
        self.z *= γ*λ
        self.z[s,a] += 1

        # Update Q-values
        self.qv += α * δ * self.z

        # Return the TD-error, for lack of something more informative
        return δ

    def get_value(self, s, a=None):
        if a is None:
            return np.max(self.qv[s])
        else:
            return self.qv[s,a]

    def greedy_action(self, s):
        return np.argmax(self.qv[s])

    def save_weights(self, fname):
        """Save the weights to a file."""
        np.save(fname, self.qv)

    def load_weights(self, fname):
        """Load the weights from a file."""
        self.qv = np.load(fname)