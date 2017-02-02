from gym.envs.registration import register

# import logging
# logger = logging.getLogger(__name__)

from .paper_chain import PaperChainMDP
from .simple_mdp import SimpleMDP

register(
    id='SimpleMDP-v0',
    entry_point='varcompfa.envs:SimpleMDP',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic=False,
)

register(
    id='PaperChainMDP-v0',
    entry_point='varcompfa.envs:PaperChainMDP',
    timestep_limit=1000,
    reward_threshold=10.0,
    nondeterministic=True,
)
