from gym.envs.registration import register
from .binary_tree import BinaryTreeMDP
from .boyan_chain import BoyanChainMDP
from .paper_chain import PaperChainMDP
from .paper_complex import PaperComplexMDP
from .simple_mdp import SimpleMDP

register(
    id='BoyanChainMDP-v0',
    entry_point='varcompfa.envs:BoyanChainMDP',
    nondeterministic=True,
)

register(
    id='PaperChainMDP-v0',
    entry_point='varcompfa.envs:PaperChainMDP',
    nondeterministic=True,
)

register(
    id='SimpleMDP-v0',
    entry_point='varcompfa.envs:SimpleMDP',
    nondeterministic=False,
)

register(
    id='PaperComplexMDP-v0',
    entry_point='varcompfa.envs:PaperComplexMDP',
    nondeterministic=False,
)

register(
    id='BinaryTreeMDP-v0',
    entry_point='varcompfa.envs:BinaryTreeMDP',
    nondeterministic=True,
)
