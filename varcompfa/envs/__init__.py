from gym.envs.registration import register
from .baird_mdp import BairdMDP
from .binary_tree import BinaryTreeMDP
from .boyan_chain import BoyanChainMDP
from .chicken import ChickenMDP
from .paper_chain import PaperChainMDP
from .paper_complex import PaperComplexMDP
from .simple_mdp import SimpleMDP
from .tamar_chain import TamarChainMDP

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

register(
    id='BairdMDP-v0',
    entry_point='varcompfa.envs:BairdMDP',
    nondeterministic=False,
)

register(
    id='ChickenMDP-v0',
    nondeterministic=False,
    entry_point='varcompfa.envs:ChickenMDP',
)

register(
    id='TamarChainMDP-v0',
    nondeterministic=False,
    entry_point='varcompfa.envs:TamarChainMDP',
)