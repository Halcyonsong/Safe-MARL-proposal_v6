"""Unit tests for V6 Ant MAPPO model, GAE and PPO update."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

NAV = Path(__file__).resolve().parents[1]
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from v6.ant_mappo import (  # noqa: E402
    AntRollout,
    MappoAntModel,
    compute_ant_gae,
    concat_ant_rollouts,
    load_ant_mappo_model,
    ppo_update_ant,
    save_ant_mappo_checkpoint,
)

from scripts.train_v6_ant_mappo import _cost_penalty_at  # noqa: E402


def _rollout(T=12):
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(T, 2, 31)).astype(np.float32)
    h = rng.normal(size=(T, 2, 1)).astype(np.float32)
    grad = rng.normal(size=(T, 2, 9)).astype(np.float32)
    states = rng.normal(size=(T, 31)).astype(np.float32)
    actions = rng.uniform(-1, 1, size=(T, 2, 4)).astype(np.float32)
    log_probs = rng.normal(size=(T, 2)).astype(np.float32)
    values = rng.normal(size=T).astype(np.float32)
    rewards = rng.normal(size=T).astype(np.float32)
    costs = (rng.random(T) < 0.1).astype(np.float32)
    terminated = np.zeros(T, dtype=bool)
    truncated = np.zeros(T, dtype=bool)
    terminated[-1] = True
    completed = [dict(
        safe_episode=0.0, unsafe_episode=1.0, wall_episode=0.0,
        topple_episode=1.0, episode_return=1.0, episode_length=12.0,
        survived=0.0, cost_rate=1.0, h_mean=0.0, h_min=-1.0,
        h_neg_frac=1.0)]
    return AntRollout(
        obs=obs, h=h, grad=grad,
        task=np.zeros((T, 2, 0), dtype=np.float32),
        states=states, actions=actions,
        log_probs=log_probs, values=values, rewards=rewards, costs=costs,
        terminated=terminated, truncated=truncated, last_value=0.0,
        completed=completed)


def _model(use_h=True):
    return MappoAntModel(
        obs_dim=31, state_dim=31, action_dim=4, num_agents=2,
        use_h_features=use_h, actor_hidden=(16, 16),
        critic_hidden=(16, 16), initial_log_std=-0.5)


def test_model_shapes_and_bounds():
    model = _model()
    obs = torch.randn(5, 2, 31)
    h = torch.randn(5, 2, 1)
    grad = torch.randn(5, 2, 9)
    actions, log_probs = model.get_actions(obs, h, grad)
    assert actions.shape == (5, 2, 4)
    assert log_probs.shape == (5, 2, 1)
    assert torch.all(actions.abs() <= 1.0)
    deterministic, _ = model.get_actions(obs, h, grad, deterministic=True)
    assert deterministic.shape == (5, 2, 4)
    logp, ent = model.evaluate_actions(obs, h, grad, actions)
    assert logp.shape == (5, 2, 1)
    assert ent.shape == (5, 2, 1)
    assert model.value(torch.randn(5, 31)).shape == (5,)


def test_no_h_features_are_not_appended():
    model = _model(use_h=False)
    obs = torch.zeros(1, 2, 31)
    h = torch.ones(1, 2, 1)
    grad = torch.ones(1, 2, 9)
    x = model.actor_input(obs[:, 0], h[:, 0], grad[:, 0])
    assert x.shape == (1, 31)
    actions, _ = model.get_actions(obs, h, grad)
    assert actions.shape == (1, 2, 4)


def test_gae_masks_termination_and_truncation():
    r = _rollout(T=4)
    r.terminated[:] = False
    r.truncated[:] = False
    r.terminated[-1] = True
    r.values[:] = 1.0
    r.rewards[:] = 1.0
    r.last_value = 100.0
    adv, ret, info = compute_ant_gae(
        r, gamma=0.9, gae_lambda=0.5, normalize_advantage=False)
    assert adv.shape == ret.shape == (4,)
    # Last transition is a true termination, so its bootstrap is masked.
    assert np.isclose(adv[-1], 1.0 - 1.0, atol=1e-6)
    assert np.isclose(ret[-1], 1.0, atol=1e-6)
    assert np.isfinite(info['adv_mean'])


def test_concat_rollouts_preserves_episodes():
    r1 = _rollout(T=8)
    r2 = _rollout(T=8)
    merged = concat_ant_rollouts([r1, r2])
    assert len(merged) == 16
    assert len(merged.completed) == 2
    assert merged.obs.shape[1:] == (2, 31)
    assert merged.log_probs.shape == (16, 2)


def test_act_single_returns_per_agent_log_prob():
    model = _model()
    obs = np.zeros((2, 31), dtype=np.float32)
    h = np.zeros((2, 1), dtype=np.float32)
    grad = np.zeros((2, 9), dtype=np.float32)
    action, log_prob = model.act_single(obs, h, grad)
    assert action.shape == (2, 4)
    assert log_prob.shape == (2,)


def test_gaussian_action_dist_has_no_tanh_bound():
    model = MappoAntModel(
        obs_dim=31, state_dim=31, action_dim=4, num_agents=2,
        use_h_features=False, actor_hidden=(16, 16),
        critic_hidden=(16, 16), initial_log_std=0.0,
        action_dist='gaussian')
    obs = torch.zeros(64, 2, 31)
    h = torch.zeros(64, 2, 1)
    grad = torch.zeros(64, 2, 9)
    actions, log_probs = model.get_actions(obs, h, grad)
    assert actions.shape == (64, 2, 4)
    assert log_probs.shape == (64, 2, 1)
    # Gaussian samples are not forced into [-1, 1].
    assert torch.any(actions.abs() > 1.0)


def test_ppo_update_changes_parameters():
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    r = _rollout()
    adv, ret, _ = compute_ant_gae(
        r, gamma=0.99, gae_lambda=0.95, normalize_advantage=True)
    before = model.actors[0].action_mean.weight.detach().clone()
    logs = ppo_update_ant(
        model, opt, r, adv, ret, epochs=1, minibatch_size=4,
        max_grad_norm=10.0)
    after = model.actors[0].action_mean.weight.detach().clone()
    assert np.isfinite(logs['loss'])
    assert not torch.allclose(before, after)


def test_ppo_separate_optimizers_and_value_norm():
    model = MappoAntModel(
        obs_dim=31, state_dim=31, action_dim=4, num_agents=2,
        use_h_features=False, actor_hidden=(16, 16),
        critic_hidden=(16, 16), actor_feature_norm=True,
        actor_layer_norm=True, critic_feature_norm=True,
        critic_layer_norm=True, use_value_norm=True,
        action_dist='gaussian')
    actor_opt = torch.optim.Adam(model.actors.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(model.critic.parameters(), lr=1e-3)
    r = _rollout()
    adv, ret, _ = compute_ant_gae(
        r, gamma=0.99, gae_lambda=0.95, normalize_advantage=True)
    logs = ppo_update_ant(
        model, actor_opt, r, adv, ret, epochs=1, minibatch_size=4,
        actor_optimizer=actor_opt, critic_optimizer=critic_opt,
        value_huber_delta=10.0, update_value_norm=True)
    assert np.isfinite(logs['loss'])
    assert model.critic.value_norm.count.item() > 0


def test_cost_penalty_schedule_is_prescheduled_not_lagrangian():
    cfg = dict(cost_penalty=0.1, cost_penalty_end=0.5,
               cost_penalty_ramp_start=500_000,
               cost_penalty_ramp_steps=1_500_000)
    assert _cost_penalty_at(0, cfg, 0.1) == 0.1
    assert abs(_cost_penalty_at(500_000, cfg, 0.1) - 0.1) < 1e-9
    assert abs(_cost_penalty_at(1_250_000, cfg, 0.1) - 0.3) < 1e-9
    assert abs(_cost_penalty_at(2_000_000, cfg, 0.1) - 0.5) < 1e-9
    assert abs(_cost_penalty_at(9_000_000, cfg, 0.1) - 0.5) < 1e-9


def test_checkpoint_roundtrip_preserves_architecture(tmp_path):
    model = MappoAntModel(
        obs_dim=31, state_dim=31, action_dim=4, num_agents=2,
        use_h_features=True, actor_hidden=(16, 16),
        critic_hidden=(16, 16), actor_activation='relu',
        critic_activation='relu', actor_feature_norm=True,
        actor_layer_norm=True, critic_feature_norm=True,
        critic_layer_norm=True, use_value_norm=True,
        action_dist='gaussian')
    cfg = dict(policy=dict(
        actor_hidden=[16, 16], critic_hidden=[16, 16],
        actor_activation='relu', critic_activation='relu',
        actor_feature_norm=True, actor_layer_norm=True,
        critic_feature_norm=True, critic_layer_norm=True,
        use_value_norm=True, action_dist='gaussian',
        h_feature_scale=1.0, grad_feature_scale=1.0))
    path = save_ant_mappo_checkpoint(
        tmp_path / 'policy_arch.pth', model, config=cfg, total_steps=10)
    loaded, _ = load_ant_mappo_model(path)
    assert loaded.actor_activation == 'relu'
    assert loaded.critic_activation == 'relu'
    assert loaded.actor_feature_norm
    assert loaded.use_value_norm
    assert loaded.action_dist == 'gaussian'


def test_checkpoint_roundtrip(tmp_path):
    model = _model()
    cfg = dict(policy=dict(
        actor_hidden=[16, 16], critic_hidden=[16, 16],
        h_feature_scale=1.0, grad_feature_scale=1.0))
    path = save_ant_mappo_checkpoint(
        tmp_path / 'policy.pth', model, config=cfg, total_steps=10)
    loaded, ckpt = load_ant_mappo_model(path)
    assert ckpt['total_steps'] == 10
    for p0, p1 in zip(model.parameters(), loaded.parameters()):
        assert torch.allclose(p0, p1)
