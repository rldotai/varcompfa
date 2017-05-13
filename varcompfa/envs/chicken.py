"""
Environment based on a game of "chicken"

Actions:
    0: Move forward
    1: Return to the start
"""
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class ChickenMDP(gym.Env):
    _num_states = 7
    action_space = spaces.Discrete(2)
    observation_space = spaces.Discrete(_num_states)
    reward_range = (-1, 0)
    _initial_state = 0
    _chicken_state = _num_states - 1
    _bumped_state = _num_states - 2
    _terminals = (_chicken_state, _bumped_state)

    def __init__(self):
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.state = self._initial_state
        return self.state

    def _transition(self, s, a):
        if s in self._terminals:
            return s
        elif a == 0:
            return s + 1
        elif a == 1:
            return self._chicken_state
        else:
            raise Exception("Bad state/action passed:", s, a)

    def _reward(self, s, a, sp):
        if s in self._terminals:
            return 0
        elif sp == self._bumped_state:
            return 1
        else:
            return 0

    def _step(self, action):
        assert(self.action_space.contains(action))
        s = self.state
        sp = self._transition(s, action)
        r = self._reward(s, action, sp)
        done = (sp in self._terminals)
        info = {}
        self.state = sp
        return (sp, r, done, info)

    def _configure(self, *args, **kwargs):
        super()._configure(*args, **kwargs)

    def _close(self, *args, **kwargs):
        super()._close(*args, **kwargs)

    def _render(self, *args, **kwargs):
        super()._render(*args, **kwargs)
