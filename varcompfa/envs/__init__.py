from gym.envs.registration import register 

# import logging
# logger = logging.getLogger(__name__)

from .simple_mdp import SimpleMDP

register(
    id='SimpleMDP-v0',
    entry_point='varcompfa.envs:SimpleMDP',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic=False,
)