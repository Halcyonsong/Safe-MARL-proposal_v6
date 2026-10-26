"""Standalone V6: safe MAPPO with a V6-learned h(s).

The main path uses a V6 self-trained, safety-only h and ``grad_s h`` as policy
features.  The actor is trained from scratch; no official h, QP/u_ref, expert
action, or V5 artifact is loaded.  The relative task scalar ``A_T`` and the
state-space task gradient ``grad_s V_T`` are a later, optional stage.
"""

from .features import torch_local_obs
from .graph_nets import LocalGraphNet
from .safety import LocalSafetyNet, LocalSafetyProvider
from .task_value import (
    LocalTaskValueNet, TaskAdvantageEMA, build_task_value_net, task_state_grad)

__all__ = [
    'torch_local_obs',
    'LocalGraphNet',
    'LocalSafetyNet',
    'LocalSafetyProvider',
    'LocalTaskValueNet',
    'build_task_value_net',
    'TaskAdvantageEMA',
    'task_state_grad',
]
