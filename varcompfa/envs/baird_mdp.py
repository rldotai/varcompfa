"""
Implementation of the "Complex" environment, from the variance paper.
"""
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding



class BairdMDP(gym.Env):
    """
    An environment implementing the "Baird's Counterexample".
    """
    initial_state = 0

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(7)
        self._terminals = tuple()
        self._state = self.initial_state
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def state(self):
        """Ensure that state is represented as an integer."""
        return int(self._state)

    def _reset(self):
        self._state = self.initial_state
        return self.state

    def _transition(self, s, a):
        if a == 0:
            return self.np_random.randint(0, 7)
        elif a == 1:
            return 0
        else:
            raise Exception("Bad action: %s"%s)

    def _reward(self, s, a, sp):
        return 0

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
