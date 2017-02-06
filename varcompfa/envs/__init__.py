from gym.envs.registration import register
from .boyan_chain import BoyanChainMDP
from .paper_chain import PaperChainMDP
from .simple_mdp import SimpleMDP

register(
    id='BoyanChainMDP-v0',
    entry_point='varcompfa.envs:BoyanChainMDP',
    timestep_limit=1000,
    reward_threshold=10.0,
    nondeterministic=True,
)

register(
    id='PaperChainMDP-v0',
    entry_point='varcompfa.envs:PaperChainMDP',
    timestep_limit=1000,
    reward_threshold=10.0,
    nondeterministic=True,
)

register(
    id='SimpleMDP-v0',
    entry_point='varcompfa.envs:SimpleMDP',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic=False,
)

