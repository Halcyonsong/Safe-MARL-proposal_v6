"""End-to-end smoke: official env + fixed h + QP-free policy, no QP."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from envs.gcbfplus import OfficialDoubleIntegrator
from algos.safe_policy.fixed_h import OfficialFixedH
from algos.safe_policy.qp_free import QpFreePolicy, qp_free_losses
from algos.safe_policy.training import _gae

CKPT = (ROOT / 'third_party/gcbfplus/pretrained/DoubleIntegrator/'
        'gcbf+/models/1000/cbf.pkl')


@pytest.mark.skipif(not CKPT.exists(),
                    reason='official GCBF+ cbf.pkl is not shipped with this repo')
def test_official_env_h_policy_step():
    env = OfficialDoubleIntegrator(num_agents=8, area_size=4, max_step=256,
                                   num_obstacles=8, n_rays=32, seed=0)
    obs = env.reset(seed=0)
    ckpt = CKPT
    hnet = OfficialFixedH(ckpt, env, n_agents=8, gnn_layers=1)
    model = QpFreePolicy(obs_dim=obs.shape[-1], hidden=64)
    h, grad = hnet.value_and_local_grad()
    h = torch.as_tensor(np.asarray(h), dtype=torch.float32)[None]
    grad = torch.as_tensor(np.asarray(grad), dtype=torch.float32)[None]
    obs_t = torch.as_tensor(obs, dtype=torch.float32)[None]
    out = model(obs_t, h[..., None], grad)
    action = out['action'][0].detach().numpy()
    assert action.shape == (8, 2)
    assert np.abs(action).max() <= 1.0 + 1e-6
    prev_states = env.get_agent_states()
    assert np.abs(obs).max() < 1e3  # no infinite-ray sentinel in policy obs
    _, _, terminated, _, info = env.step(action)
    assert terminated is False
    h_next, grad_next = hnet.value_and_local_grad()
    h_next = torch.as_tensor(np.asarray(h_next), dtype=torch.float32)[None]
    states = env.get_agent_states()
    delta = torch.as_tensor(states - prev_states, dtype=torch.float32)[None]
    losses = qp_free_losses(out, h, h_next, grad, delta)
    assert all(torch.isfinite(v).all() for v in losses.values())
    assert set(info) >= {'safety', 'unsafe', 'reached'}


def test_gae_terminal_bootstrap():
    rewards = torch.tensor([[1.0], [1.0]])
    values = torch.tensor([[0.5], [0.5]])
    dones = torch.tensor([False, True])
    adv, ret = _gae(rewards, values, dones, torch.zeros(1), gamma=0.9, lam=1.0)
    assert torch.allclose(adv[:, 0], torch.tensor([1.4, 0.5]))
    assert torch.allclose(ret[:, 0], torch.tensor([1.9, 1.0]))
