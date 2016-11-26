"""
A simple Markov Decision Process environment.
"""
import numpy as np 
import gym
from gym import spaces



class ChainMDP(gym.Env):
    """
    An MDP that that acts like a corridor.
    """
    #TODO: Settle on a specification for this.

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(6)
        self.reward_range = (-1.0, 1.0)
        self._terminals = tuple([self.observation_space.n - 1])

    @property
    def state(self):
        """Ensure that state is always of type `numpy.ndarray`."""
        return np.array(self._state)

    def _reset(self):
        self._state = 0
        return self.state

    def _transition(self, s, a):
        if s in self._terminals:
            return s
        elif a == 0: 
            return np.max(s-1, 0)
        elif a == 1:
            return s+1
        else:
            raise Exception("Bad action passed {}".format(a))

    def _reward(self, s, a, sp):
        if s in self._terminals:
            return 0
        elif s in self._terminals:
            return 0
        else:
            return -1

    def _step(self, action):
        assert(self.action_space.contains(action))
        obs     = self.state
        obs_p   = self._transition(obs, action)
        reward  = self._reward(obs, action, obs_p)
        done    = obs_p in self._terminals
        info    = {}

        # Modify state and return the step tuple
        self._state = obs_p
        return (obs_p, reward, done, info)

    def _configure(self, *args, **kwargs):
        super()._configure(*args, **kwargs)

    def _close(self, *args, **kwargs):
        super()._close(*args, **kwargs)

    def _render(self, *args, **kwargs):
        super()._render(*args, **kwargs)

    def _seed(self, *args, **kwargs):
        super()._seed(*args, **kwargs)