"""
Implementation of Boyan's chain problem, a Markov process (i.e., an MDP with
only one action.)

Taken from https://papers.nips.cc/paper/3092-ilstd-eligibility-traces-and-convergence-analysis.pdf
"""
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class BoyanChainMDP(gym.Env):
    """
    An environment implementing the Markov chain associatd with Boyan.


    Notes
    -----
    The canonical features for the MDP are specified in `varcompfa.features` as
    `BoyanFeatures` (see a description of the problem for details).
    """
    initial_state = 12
    def __init__(self):
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Discrete(13)
        self.reward_range = (-100.0, 100.0)
        self._terminals = (0,)
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
        if s in self._terminals:
            ret = s
        elif s > 1:
            if (self.np_random.uniform() < 0.5):
                ret = (s - 1)
            else:
                ret = (s- 2)
        elif s == 1:
            ret = 0
        return ret

    def _reward(self, s, a, sp):
        if s in self._terminals:
            return 0
        elif s == 1 and sp == 0:
            return -2
        elif s >= 2:
            return -3
        else:
            raise Exception("Unspecified reward for transition: (%d, %d, %d)"%(s, a, sp))

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
