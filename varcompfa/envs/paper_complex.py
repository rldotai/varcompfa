"""
Implementation of the "Complex" environment, from the variance paper.
"""
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding



class PaperComplexMDP(gym.Env):
    """
    An environment implementing the "Complex" MDP from the variance paper.
    """
    initial_state = 0

    # Transitions are specified in state->action->result
    __transitions = {
        0: {
            0 : {'reward': -0.5, 'next_state': 1},
        },
        1: {
            0 : {'reward': -1, 'next_state': 2},
            1 : {'reward': 1, 'next_state': 2},
            2 : {'reward': -0.5, 'next_state': 3},
            3 : {'reward': 0.5, 'next_state': 3},
            4 : {'reward': 0, 'next_state': 0},
        },
        2: {
            0 : {'reward': 0, 'next_state': 0},
            1 : {'reward': 1, 'next_state': 4},
            2 : {'reward': 3, 'next_state': 4},
        },
        3: {
            0 : {'reward': 1, 'next_state': 1},
            1 : {'reward': 1, 'next_state': 4},
            2 : {'reward': 2, 'next_state': 4},
        },
        4: {
            0 : {'reward': 0, 'next_state': 2},
            1 : {'reward': 0, 'next_state': 0},
        },
    }


    def __init__(self):
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Discrete(5)
        # self.reward_range = (-100.0, 100.0)
        self._terminals = tuple()
        self._state = self.initial_state
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def state(self):
        """Ensure that state is represented as an integer."""
        return np.int(self._state)

    def _reset(self):
        self._state = self.initial_state
        return self.state

    def _transition(self, s, a):
        try:
            trans = self.__transitions[s][a]
        except KeyError as e:
            raise(e)
        except Exception as e:
            raise(e)
        return trans['next_state']

    def _reward(self, s, a, sp):
        try:
            trans = self.__transitions[s][a]
        except KeyError as e:
            raise(e)
        except Exception as e:
            raise(e)
        return trans['reward']

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
