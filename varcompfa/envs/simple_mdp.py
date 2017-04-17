"""
A simple Markov Decision Process environment.
"""
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class SimpleMDP(gym.Env):
    """
    An extremely simple MDP.
    """
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(3)
        self.reward_range = (-1.0, 1.0)
        self._terminals = tuple([self.observation_space.n - 1])
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def state(self):
        """The current state of the environment.."""
        return int(self._state)

    def _reset(self):
        self._state = 0
        return self.state

    def _transition(self, s, a):
        sp = s + 1 if s not in self._terminals else s
        return np.array(sp)

    def _reward(self, s, a, sp):
        if s in self._terminals:
            return 0
        else:
            return 1 if a == 0 else -1

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

    # TODO: Override
    def _configure(self, *args, **kwargs):
        super()._configure(*args, **kwargs)

    def _close(self, *args, **kwargs):
        super()._close(*args, **kwargs)

    def _render(self, *args, **kwargs):
        super()._render(*args, **kwargs)
