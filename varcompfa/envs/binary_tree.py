"""
A tree-like environment where rewards predict future rewards.

Going to the 'left' results in a negative reward, while a transition to the
right yields a positive reward.

Note that we don't specify the discount here, but assuming constant discounting
and that transitioning from a terminal state has γ = 0, the following is true:

-   Under random policy, the states enumerated as (-7, -6, ..., 6, 7), have the
    same value as their state 'number', except for the "odd" states (the leafs of
    the tree) which have value zero as they are supposed to be terminal.
-   The choice of action determines the variance of the rewards.
-   Solving for value/variance is easily accomplished analytically.
"""
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class BinaryTreeMDP(gym.Env):
    """
    An environment engineered to have correlated TD-errors.
    """
    _initial_state = 0

    _transitions = {
        0: {
            0: {'reward': -4, 'next_state': -4},
            1: {'reward': 4, 'next_state': 4},
            2: {'reward': 0, 'next_state': 0}, # for aperiodicity
        },
        -4: {
            0: {'reward': -6, 'next_state': -6},
            1: {'reward': -2, 'next_state': -2},
        },
        -6: {
            0: {'reward': -7, 'next_state': -7},
            1: {'reward': -5, 'next_state': -5},
        },
        -7: {
            0: {'reward': 0, 'next_state': 0},
            1: {'reward': 0, 'next_state': 0},
        },
        -5: {
            0: {'reward': 0, 'next_state': 0},
            1: {'reward': 0, 'next_state': 0},
        },
        -2: {
            0: {'reward': -3, 'next_state': -3},
            1: {'reward': -1, 'next_state': -1},
        },
        -3: {
            0: {'reward': 0, 'next_state': 0},
            1: {'reward': 0, 'next_state': 0},
        },
        -1: {
            0: {'reward': 0, 'next_state': 0},
            1: {'reward': 0, 'next_state': 0},
        },
        4: {
            0: {'reward': 2, 'next_state': 2},
            1: {'reward': 6, 'next_state': 6},
        },
        2: {
            0: {'reward': 1, 'next_state': 1},
            1: {'reward': 3, 'next_state': 3},
        },
        3: {
            0: {'reward': 0, 'next_state': 0},
            1: {'reward': 0, 'next_state': 0},
        },
        1: {
            0: {'reward': 0, 'next_state': 0},
            1: {'reward': 0, 'next_state': 0},
        },
        6: {
            0: {'reward': 5, 'next_state': 5},
            1: {'reward': 7, 'next_state': 7},
        },
        5: {
            0: {'reward': 0, 'next_state': 0},
            1: {'reward': 0, 'next_state': 0},
        },
        7: {
            0: {'reward': 0, 'next_state': 0},
            1: {'reward': 0, 'next_state': 0},
        },
    }

    _lowest = min(_transitions.keys())

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(15)
        self.reward_range = (-7, 7)
        self._state_map = {k: k - self._lowest for k in self._transitions}
        self._terminals = tuple([])
        self._state = self._initial_state
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def state(self):
        """Ensure that state is represented as an integer."""
        return np.int(self._state_map[self._state])

    def _reset(self):
        self._state = self._initial_state
        return self.state

    def _transition(self, s, a):
        try:
            trans = self._transitions[s][a]
        except KeyError as e:
            raise(e)
        except Exception as e:
            raise(e)
        return trans['next_state']

    def _reward(self, s, a, sp):
        try:
            trans = self._transitions[s][a]
        except KeyError as e:
            raise(e)
        except Exception as e:
            raise(e)
        return trans['reward']

    def _step(self, action):
        assert(self.action_space.contains(action))
        # Action '2' is effectively random except in starting state
        if (action == 2) and (self._state != 0):
            action = np.random.randint(0, 2)
        s       = self._state
        sp      = self._transition(s, action)
        reward  = self._reward(s, action, sp)
        done    = (sp in self._terminals)
        info    = {}

        # Modify state and return the step tuple
        self._state = sp
        obs = self._state_map[s]
        obs_p = self._state_map[sp]
        return (obs_p, reward, done, info)

    # TODO: Override
    def _configure(self, *args, **kwargs):
        super()._configure(*args, **kwargs)

    def _close(self, *args, **kwargs):
        super()._close(*args, **kwargs)

    def _render(self, *args, **kwargs):
        super()._render(*args, **kwargs)
