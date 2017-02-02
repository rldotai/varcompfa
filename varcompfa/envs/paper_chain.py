"""
A simple Markov Decision Process environment.
"""
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class PaperChainMDP(gym.Env):
    """
    An MDP that that acts like a corridor.
    """
    def __init__(self):
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(5)
        self.reward_range = (-100.0, 100.0)
        self._terminals = tuple([self.observation_space.n - 1])
        self._state = 0
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

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
            return s+1
        else:
            raise Exception("Bad action passed {}".format(a))

    def _reward(self, s, a, sp):
        if s in self._terminals:
            return 0
        else:
            return self.np_random.normal(1.0, 1.0)

    def _step(self, action):
        assert(self.action_space.contains(action))
        obs     = self.state
        obs_p   = self._transition(obs, action)
        reward  = self._reward(obs, action, obs_p)
        done    = (obs_p in self._terminals)
        info    = {}

        # Modify state and return the step tuple
        self._state = obs_p
        return (obs_p, reward, done, info)

    # TODO: Override
    def _configure(self, *args, **kwargs):
        super()._configure(*args, **kwargs)

    def _close(self, *args, **kwargs):
        super()._close(*args, **kwargs)

    def _render(self, *args, **kwargs):
        super()._render(*args, **kwargs)
