"""Contract tests for independent V6: learned h, optional V_T, V6 policy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

NAV = Path(__file__).resolve().parents[1]
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from envs.gcbfplus import OfficialDoubleIntegrator
from v6.policy import QpFreePolicy
from v6.features import torch_local_obs
from v6.graph_nets import LocalGraphNet
from v6.safety import (
    LocalSafetyNet, LocalSafetyProvider, dense_margin_values,
    forward_margin_sequence, horizon_margin_targets, horizon_min_margin,
    save_safety_checkpoint)
from v6.task_value import (
    LocalTaskValueNet, TaskAdvantageEMA, build_task_value_net,
    task_state_grad,
)
from v6.training import (
    apply_safety_task_gate, apply_task_gate_components, safety_task_gate)


def _env():
    env = OfficialDoubleIntegrator(
        num_agents=8, area_size=4.0, max_step=32, num_obstacles=8,
        n_rays=32, max_neighbors=5, seed=0)
    obs = env.reset(seed=0)
    return env, obs


def test_torch_local_obs_matches_env_observation():
    env, obs = _env()
    states = torch.tensor(env.get_agent_states(), dtype=torch.float32,
                          requires_grad=True)
    goals = torch.tensor(env.get_goal_states(), dtype=torch.float32)
    lidar = torch.tensor(env.get_lidar(), dtype=torch.float32)
    rebuilt = torch_local_obs(
        states, goals, lidar, max_neighbors=5, n_rays=32,
        comm_radius=env.comm_radius)
    assert rebuilt.shape == obs.shape
    assert torch.allclose(rebuilt.detach(), torch.tensor(obs), atol=1e-5)


def test_learned_h_and_task_gradients_are_finite():
    env, obs = _env()
    states = torch.tensor(env.get_agent_states(), dtype=torch.float32)
    goals = torch.tensor(env.get_goal_states(), dtype=torch.float32)
    lidar = torch.tensor(env.get_lidar(), dtype=torch.float32)
    hnet = LocalSafetyNet(obs.shape[-1], (32,))
    h, grad_h = hnet.value_and_local_grad(
        states, goals, lidar, max_neighbors=5, n_rays=32,
        comm_radius=env.comm_radius)
    assert h.shape == (8,)
    assert grad_h.shape == (8, 4)
    assert torch.isfinite(h).all() and torch.isfinite(grad_h).all()

    task = LocalTaskValueNet(obs.shape[-1], (32,))
    value, grad_t = task_state_grad(
        task, states, goals, lidar, max_neighbors=5, n_rays=32,
        comm_radius=env.comm_radius)
    assert value.shape == (8,)
    assert grad_t.shape == (8, 4)
    assert torch.isfinite(value).all() and torch.isfinite(grad_t).all()


def test_policy_extra_features_and_checkpoint(tmp_path):
    env, obs = _env()
    hnet = LocalSafetyNet(obs.shape[-1], (32,))
    path = save_safety_checkpoint(tmp_path / 'h.pt', hnet)
    provider = LocalSafetyProvider.from_checkpoint(
        path, env, device='cpu', max_neighbors=5, n_rays=32,
        comm_radius=env.comm_radius)
    h, grad_h = provider.value_and_local_grad()
    assert h.shape == (8,) and grad_h.shape == (8, 4)

    model = QpFreePolicy(obs.shape[-1], hidden=32, extra_dim=5)
    obs_t = torch.tensor(obs, dtype=torch.float32)[None]
    h_t = torch.tensor(h, dtype=torch.float32)[None, :, None]
    grad_t = torch.tensor(grad_h, dtype=torch.float32)[None]
    extra = torch.zeros(1, 8, 5)
    out = model(obs_t, h_t, grad_t, extra)
    assert out['action'].shape == (1, 8, 2)
    assert torch.all(out['action'].abs() <= 1.0)
    sampled = model.sample(obs_t, h_t, grad_t, extra)
    assert sampled['log_prob'].shape == (1, 8)
    evaluated = model.evaluate_actions(
        obs_t, h_t, grad_t, sampled['raw_action'], extra)
    assert evaluated['log_prob'].shape == (1, 8)


def test_local_graph_nets_and_checkpoint(tmp_path):
    env, obs = _env()
    states = torch.tensor(env.get_agent_states(), dtype=torch.float32)
    goals = torch.tensor(env.get_goal_states(), dtype=torch.float32)
    lidar = torch.tensor(env.get_lidar(), dtype=torch.float32)
    hnet = LocalGraphNet(obs.shape[-1], hidden=32, layers=2,
                         max_neighbors=5, n_rays=32,
                         output_activation='tanh')
    h, grad_h = hnet.value_and_local_grad(
        states, goals, lidar, max_neighbors=5, n_rays=32,
        comm_radius=env.comm_radius)
    assert h.shape == (8,) and grad_h.shape == (8, 4)
    assert torch.isfinite(h).all() and torch.isfinite(grad_h).all()

    path = save_safety_checkpoint(tmp_path / 'h_graph.pt', hnet)
    provider = LocalSafetyProvider.from_checkpoint(
        path, env, device='cpu', max_neighbors=5, n_rays=32,
        comm_radius=env.comm_radius)
    h2, grad2 = provider.value_and_local_grad()
    assert h2.shape == (8,) and grad2.shape == (8, 4)

    task = build_task_value_net(
        dict(arch='graph', graph_hidden=32, graph_layers=2),
        obs.shape[-1], max_neighbors=5, n_rays=32)
    value, grad_t = task_state_grad(
        task, states, goals, lidar, max_neighbors=5, n_rays=32,
        comm_radius=env.comm_radius)
    assert value.shape == (8,) and grad_t.shape == (8, 4)
    assert torch.isfinite(value).all() and torch.isfinite(grad_t).all()


def test_task_ema_relative_scalar():
    ema = TaskAdvantageEMA.zeros(4, warmup_count=1.0, clip=5.0)
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    assert torch.all(ema.normalize(torch.ones(4)) == 0)
    ema.update(values)
    out = ema.normalize(torch.full((4,), 3.0))
    assert out.shape == (4,)
    assert torch.isfinite(out).all()
    assert ema.count == 2


def test_task_ema_tracks_recent_policy_not_cumulative_history():
    ema = TaskAdvantageEMA.zeros(1, warmup_count=1.0, decay=0.5, var_floor=0.0)
    ema.update(torch.zeros(128, 1))
    ema.update(torch.full((128, 1), 10.0))
    # With a recency EMA the baseline must move halfway, not stay near zero.
    assert abs(float(ema.mean[0]) - 5.0) < 1e-4
    assert abs(float(ema.normalize(torch.tensor([10.0]))[0]) - 1.0) < 1e-4


def test_horizon_labels_use_forward_safe_window():
    from v6.safety_labels import forward_safe_mask, horizon_labels
    # Six training rows plus a terminal state.  Unsafe at row 4.
    unsafe = np.array([[0], [0], [0], [0], [1], [0], [0]], dtype=bool)
    safe = forward_safe_mask(unsafe, horizon=2)
    labels, _ = horizon_labels(unsafe, horizon=2)
    # Rows 2..4 are within two steps before the unsafe event; row 4 is the
    # negative state itself and the intermediate rows are ignored.
    assert safe[:, 0].tolist() == [True, True, False, False, False, True]
    assert labels[:, 0].tolist() == [1, 1, 0, 0, -1, 1]

    # A terminal unsafe event (one past the last training row) must close the
    # forward window of the preceding horizon states.
    terminal = np.array([[0], [0], [0], [0], [0], [0], [1]], dtype=bool)
    safe_terminal = forward_safe_mask(terminal, horizon=2)
    labels_terminal, _ = horizon_labels(terminal, horizon=2)
    assert safe_terminal[:, 0].tolist() == [True, True, True, True, False, False]
    assert labels_terminal[:, 0].tolist() == [1, 1, 1, 1, 0, 0]


def test_action_space_cbf_loss_depends_on_actor_action():
    from v6.safety_labels import cbf_violation_torch
    h = torch.tensor([[0.1, 0.2]])
    grad = torch.tensor([[[0.1, 0.2, 0.3, 0.4],
                          [0.0, 0.1, 0.2, 0.3]]])
    states = torch.tensor([[[1.0, 0.0, 0.1, 0.2],
                            [1.0, 1.0, -0.1, 0.0]]])
    action = torch.full((1, 2, 2), -1.0, requires_grad=True)
    _, violation = cbf_violation_torch(
        h, grad, states, action, mass=0.1, alpha=1.0, eps=0.02)
    loss = violation.mean()
    loss.backward()
    assert float(loss.detach()) > 0.0
    assert action.grad is not None
    assert float(action.grad.abs().sum()) > 0.0


def test_navigation_safety_task_gate_is_soft_and_positive_only():
    h_next = np.array([-0.1, 0.0, 0.065, 0.13, 0.2], dtype=np.float32)
    gate = safety_task_gate(h_next, low=0.0, high=0.13)
    assert np.allclose(gate, [0.0, 0.0, 0.5, 1.0, 1.0])
    assert np.all(safety_task_gate(h_next, low=0.1, high=0.1) == 1.0)

    task = np.array([2.0, -0.4, 5.0, -2.0, 0.0], dtype=np.float32)
    gated = apply_safety_task_gate(task, gate, positive_only=True)
    # Positive progress/reach is gated; negative progress is still penalized.
    assert np.allclose(gated, [0.0, -0.4, 2.5, -2.0, 0.0])
    fully = apply_safety_task_gate(task, gate, positive_only=False)
    assert np.allclose(fully, [0.0, 0.0, 2.5, -2.0, 0.0])


def test_stage_a_gate_scope_leaves_reach_and_terminal_ungated():
    progress = np.array([2.0, -1.0, 3.0], dtype=np.float32)
    reach = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    terminal = np.array([0.0, 20.0, 0.0], dtype=np.float32)
    gate = np.array([0.0, 1.0, 0.5], dtype=np.float32)

    # Only dense progress is scaled.  Negative progress stays penalized, and
    # reach/terminal components are added unchanged.
    stage_a = apply_task_gate_components(
        progress, reach, terminal, gate, scope='progress_only')
    assert np.allclose(stage_a, [5.0, 19.0, 1.5])

    # The old scope scales the summed positive reward and preserves the
    # negative component, reproducing the first gate experiment.
    old = apply_task_gate_components(
        progress, reach, terminal, gate, scope='all_positive')
    assert np.allclose(old, [0.0, 19.0, 1.5])

    try:
        apply_task_gate_components(
            progress, reach, terminal, gate, scope='unknown')
    except ValueError as exc:
        assert 'task_gate_scope' in str(exc)
    else:
        raise AssertionError('unknown gate scope must fail')


def test_navigation_future_min_margin_target():
    margins = np.array([[0.2], [0.1], [-0.2], [0.3]], dtype=np.float32)
    risk = horizon_min_margin(margins, horizon=2)
    assert np.allclose(risk[:, 0], [-0.2, -0.2, -0.2, 0.3])
    target, raw = horizon_margin_targets(
        margins, horizon=2, scale=0.2)
    assert np.allclose(target, np.tanh(raw / 0.2), atol=1e-6)
    assert target[0] < 0.0 and target[-1] > 0.0


def test_navigation_dense_margin_is_signed_geometry():
    env, obs = _env()
    margin = dense_margin_values(
        obs, max_neighbors=5, n_rays=32,
        area_size=4.0, comm_radius=env.comm_radius)
    assert margin.shape == (env.num_agents,)
    assert np.isfinite(margin).all()
    assert np.all(margin <= 0.5) and np.all(margin >= -0.5)


def test_navigation_forward_margin_sequence_does_not_wrap_or_duplicate():
    margins = np.array([[-1.0], [0.2], [0.3], [0.4]], dtype=np.float32)
    terminal = np.array([0.5], dtype=np.float32)
    seq = forward_margin_sequence(margins, terminal)
    assert seq.shape == (5, 1)
    assert np.allclose(seq[:, 0], [-1.0, 0.2, 0.3, 0.4, 0.5])
    _, risk = horizon_margin_targets(seq, horizon=32, scale=0.2)
    # The last current state must see only itself and the terminal state, not
    # an early-episode margin from a duplicated rollout.  ``risk[-1]`` is the
    # non-training terminal row; the last training row is ``len(margins)-1``.
    assert np.isclose(risk[len(margins) - 1, 0], 0.4)
    assert np.isclose(risk[0, 0], -1.0)
