"""Ant task-value V_T: offline predictor and frozen task features.

The Ant V_T is deliberately separate from the PPO critic.  It is trained on
the official Ant task reward only (no h gate, no safety cost) and is frozen
before the actor warm-start.  Two feature modes are supported:

``A_T``
    1-D relative scalar (the original diagnostic).
``A_T_grad``
    ``[A_T, grad_s V_T]``, with the state gradient unit-normalized per agent;
    this mirrors the navigation 5-D feature and gives the actor an action
    direction rather than only a value baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .task_value import (
    LocalTaskValueNet, TaskAdvantageEMA, normalize_gradient)


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


@torch.enable_grad()
def ant_task_grad_feature(
    task_value: AntTaskValue,
    obs: np.ndarray,
    *,
    device: str | torch.device = 'cpu',
    grad_scale: float = 1.0,
) -> np.ndarray:
    """Return [N, 1 + obs_dim] = [A_T, unit_norm(grad_s V_T)]."""
    dev = torch.device(device)
    obs_t = torch.as_tensor(
        np.asarray(obs, dtype=np.float32), device=dev).detach().clone()
    obs_t.requires_grad_(True)
    value = task_value.net(obs_t)
    grad = torch.autograd.grad(
        value.sum(), obs_t, create_graph=False, retain_graph=False)[0]
    grad = normalize_gradient(grad, max_norm=1.0) * float(grad_scale)
    a_t = task_value.ema.normalize(value.detach().cpu()).to(dev)[:, None]
    feat = torch.cat([a_t, grad.detach()], dim=-1)
    return feat.detach().cpu().numpy().astype(np.float32)


def make_task_feature_fn(
    task_value: AntTaskValue,
    *,
    device: str | torch.device = 'cpu',
    mode: str = 'A_T',
    grad_scale: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    mode = str(mode)
    if mode == 'A_T':
        def _fn(obs: np.ndarray) -> np.ndarray:
            return ant_task_feature(task_value, obs, device=device)
    elif mode == 'A_T_grad':
        def _fn(obs: np.ndarray) -> np.ndarray:
            return ant_task_grad_feature(
                task_value, obs, device=device, grad_scale=grad_scale)
    else:
        raise ValueError(
            f'unknown Ant task feature mode {mode!r}; '
            "expected 'A_T' or 'A_T_grad'")
    return _fn


__all__ = [
    'AntTaskValue', 'load_ant_task_value', 'ant_task_feature',
    'ant_task_grad_feature', 'make_task_feature_fn',
]
