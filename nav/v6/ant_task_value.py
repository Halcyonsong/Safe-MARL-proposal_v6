"""Ant task-value V_T: offline predictor and frozen A_T feature.

The Ant V_T is deliberately separate from the PPO critic.  It is trained on
the official Ant task reward only (no h gate, no safety cost) and is frozen
before the actor warm-start.  This pilot feeds the actor the 1-D relative
scalar ``A_T``; the navigation 5-D state-gradient feature is a separate
follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .task_value import LocalTaskValueNet, TaskAdvantageEMA


@dataclass
class AntTaskValue:
    net: LocalTaskValueNet
    ema: TaskAdvantageEMA
    payload: dict


def load_ant_task_value(
    path: str | Path,
    *,
    device: str | torch.device = 'cpu',
) -> AntTaskValue:
    dev = torch.device(device)
    payload = torch.load(Path(path), map_location=dev, weights_only=False)
    config = dict(payload.get('config', {}))
    hidden = tuple(int(v) for v in config.get('hidden', (128, 128)))
    net = LocalTaskValueNet(int(payload['obs_dim']), hidden=hidden)
    net.load_state_dict(payload['task_net'])
    net.to(dev).eval()
    for param in net.parameters():
        param.requires_grad_(False)
    ema = TaskAdvantageEMA.from_state_dict(payload['task_ema'])
    return AntTaskValue(net=net, ema=ema, payload=payload)


@torch.no_grad()
def ant_task_feature(
    task_value: AntTaskValue,
    obs: np.ndarray,
    *,
    device: str | torch.device = 'cpu',
) -> np.ndarray:
    """Return the [N, 1] relative task scalar A_T for one environment step."""
    dev = torch.device(device)
    obs_t = torch.as_tensor(
        np.asarray(obs, dtype=np.float32), device=dev)
    value = task_value.net(obs_t).detach()
    a_t = task_value.ema.normalize(value.detach().cpu()).to(dev)
    return a_t[:, None].detach().cpu().numpy().astype(np.float32)


def make_task_feature_fn(
    task_value: AntTaskValue,
    *,
    device: str | torch.device = 'cpu',
) -> Callable[[np.ndarray], np.ndarray]:
    def _fn(obs: np.ndarray) -> np.ndarray:
        return ant_task_feature(task_value, obs, device=device)
    return _fn


__all__ = [
    'AntTaskValue', 'load_ant_task_value', 'ant_task_feature',
    'make_task_feature_fn',
]
