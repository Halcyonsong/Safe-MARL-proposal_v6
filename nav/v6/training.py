"""Standalone V6 rollout and training utilities.

The main V6 path uses a V6-learned, safety-only h and its state gradient as
actor features.  The actor is trained from scratch with PPO/MAPPO and a cost
critic; there is no QP, projection, expert action, official h, or V5 weight
initialization.  The task channel (``A_T`` / ``grad_s V_T``) is a later stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .policy import (
    QpFreePolicy,
    barrier_loss,
    correction_penalty,
    forward_invariance_loss,
    gradient_direction_loss,
)
from .task_value import (
    LocalTaskValueNet,
    TaskAdvantageEMA,
    build_task_value_net,
    normalize_gradient,
    task_state_grad,
)
from .co_train import graph_to_arrays
from .safety_labels import cbf_terms_numpy, cbf_violation_torch


def _tensor(array, device):
    return torch.as_tensor(np.array(array, copy=True), dtype=torch.float32,
                           device=device)


def _gae(rewards, values, dones, last_value, gamma, lam):
    """Per-agent GAE for a single environment."""
    steps = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for t in range(steps - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < steps else last_value
        nonterminal = 1.0 - dones[t].to(rewards.dtype)
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + values


def _normalize(x):
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


def safety_task_gate(h_next, low: float, high: float):
    """Soft state-dependent gate for task progress.

    This is the navigation analogue of the Ant predictive-h forward gate.  It
    returns a per-agent value in [0, 1]; ``high <= low`` disables the gate.
    The gate must be used only on positive task progress/reach reward, never
    on the official collision cost.
    """
    h_next = np.asarray(h_next, dtype=np.float32)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.ones_like(h_next, dtype=np.float32)
    return np.clip((h_next - float(low)) / (float(high) - float(low)),
                   0.0, 1.0).astype(np.float32)


def apply_safety_task_gate(task_reward, gate, positive_only: bool = True):
    """Apply a safety gate to task reward, preserving negative penalties."""
    task_reward = np.asarray(task_reward, dtype=np.float32)
    gate = np.asarray(gate, dtype=np.float32)
    if positive_only:
        return np.where(task_reward > 0.0,
                        task_reward * gate, task_reward).astype(np.float32)
    return (task_reward * gate).astype(np.float32)


def apply_task_gate_components(
    progress_reward,
    reach_reward,
    terminal_reward,
    gate,
    *,
    scope: str = 'progress_only',
    positive_only: bool = True,
):
    """Gate only the selected task components.

    ``all_positive`` reproduces the first navigation gate experiment.
    ``progress_only`` is the Ant-style Stage-A design: dense progress is
    gated, while reach and terminal/success bonuses are left unchanged.
    """
    progress = np.asarray(progress_reward, dtype=np.float32)
    reach = np.asarray(reach_reward, dtype=np.float32)
    terminal = np.asarray(terminal_reward, dtype=np.float32)
    scope = str(scope).strip().lower()
    if scope in ('all_positive', 'all', 'positive_all'):
        return apply_safety_task_gate(
            progress + reach + terminal, gate,
            positive_only=positive_only)
    if scope in ('progress_only', 'progress', 'dense_progress'):
        return (apply_safety_task_gate(
            progress, gate, positive_only=positive_only) + reach + terminal
                ).astype(np.float32)
    raise ValueError(
        f'unknown task_gate_scope={scope!r}; expected all_positive or '
        'progress_only')


@dataclass
class V6Rollout:
    obs: torch.Tensor
    h: torch.Tensor
    grad_h: torch.Tensor
    task_feat: torch.Tensor
    action_raw: torch.Tensor
    action_env: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    cost: torch.Tensor
    reward: torch.Tensor
    task_reward: torch.Tensor
    task_gate: torch.Tensor
    cost_target: torch.Tensor
    done: torch.Tensor
    h_next: torch.Tensor
    delta_state: torch.Tensor
    agent_states: torch.Tensor
    task_value: torch.Tensor
    last_value: torch.Tensor
    last_cost_value: torch.Tensor
    last_task_value: torch.Tensor
    completed: list[dict[str, float]]
    h_graph: list[dict] | None = None
    h_next_graph: list[dict] | None = None
    h_labels: np.ndarray | None = None
    h_hdot_mask: np.ndarray | None = None

    def flatten(self):
        t, n = self.h.shape
        return dict(
            obs=self.obs.reshape(t * n, -1),
            h=self.h.reshape(-1),
            grad_h=self.grad_h.reshape(-1, 4),
            task_feat=self.task_feat.reshape(t * n, -1),
            action_raw=self.action_raw.reshape(t * n, -1),
            log_prob=self.log_prob.reshape(-1),
            value=self.value.reshape(-1),
            cost=self.cost.reshape(-1),
            reward=self.reward.reshape(-1),
            task_reward=self.task_reward.reshape(-1),
            task_gate=self.task_gate.reshape(-1),
            cost_target=self.cost_target.reshape(-1),
            h_next=self.h_next.reshape(-1),
            delta_state=self.delta_state.reshape(-1, 4),
            agent_states=self.agent_states.reshape(-1, 4),
            task_value=self.task_value.reshape(-1),
            valid=(~self.done).to(self.obs.dtype)[:, None].expand(t, n).reshape(-1),
        )


def _state_tensors(env, device):
    states = torch.tensor(
        env.get_agent_states(), dtype=torch.float32, device=device)
    goals = torch.tensor(
        env.get_goal_states(), dtype=torch.float32, device=device)
    lidar = torch.tensor(
        env.get_lidar(), dtype=torch.float32, device=device)
    return states, goals, lidar


def _task_channel(
    task_net: LocalTaskValueNet,
    ema: TaskAdvantageEMA,
    states: torch.Tensor,
    goals: torch.Tensor,
    lidar: torch.Tensor,
    *,
    max_neighbors: int,
    n_rays: int,
    comm_radius: float,
    use_scalar: bool,
    use_grad: bool,
    grad_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return V_T [N], A_T [N] and the [N,5] policy feature."""
    s = states.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        value, grad = task_state_grad(
            task_net, s, goals, lidar,
            max_neighbors=max_neighbors, n_rays=n_rays,
            comm_radius=comm_radius)
    value = value.detach()
    if use_scalar:
        a_t = ema.normalize(value.cpu()).to(value.device)
    else:
        a_t = torch.zeros_like(value)
    if use_grad:
        g = normalize_gradient(grad.detach()) * float(grad_scale)
    else:
        g = torch.zeros_like(grad)
    feat = torch.cat([a_t[:, None], g], dim=-1)
    return value, a_t, feat


def collect_rollout_v6(
    env,
    hnet,
    task_net: LocalTaskValueNet,
    ema: TaskAdvantageEMA,
    model: QpFreePolicy,
    steps: int,
    seed: int,
    device,
    config: dict[str, Any],
    *,
    global_step: int = 0,
) -> V6Rollout:
    model.eval()
    task_net.eval()
    training = config['training']
    use_h_features = bool(config.get('h', {}).get('use_features', True))
    h_feature_scale = float(config.get('h', {}).get('feature_scale', 1.0))
    num_agents = env.num_agents
    max_neighbors = int(config['env']['max_neighbors'])
    n_rays = int(config['env']['n_rays'])
    comm_radius = float(env.comm_radius)
    task_grad_start = int(training.get('task_grad_start_steps', 0))
    task_grad_scale = float(training.get('task_grad_scale', 1.0))
    use_task_scalar = bool(training.get('use_task_scalar', True))
    use_task_grad = (bool(training.get('use_task_grad', True))
                     and int(global_step) >= task_grad_start)
    task_gate_low = float(training.get('task_gate_low', 0.0))
    task_gate_high = float(training.get('task_gate_high', 0.0))
    task_gate_positive_only = bool(
        training.get('task_gate_positive_only', True))
    task_gate_scope = str(
        training.get('task_gate_scope', 'all_positive')).strip().lower()
    collect_h = bool(config.get('h', {}).get('co_train', False))

    obs = env.reset(seed=seed)
    h, grad_h = hnet.value_and_local_grad()
    h_policy = (h * h_feature_scale if use_h_features
                else np.zeros_like(h))
    grad_policy = (grad_h * h_feature_scale if use_h_features
                   else np.zeros_like(grad_h))
    states, goals, lidar = _state_tensors(env, device)
    task_v, a_t, task_feat = _task_channel(
        task_net, ema, states, goals, lidar,
        max_neighbors=max_neighbors, n_rays=n_rays,
        comm_radius=comm_radius, use_scalar=use_task_scalar,
        use_grad=use_task_grad, grad_scale=task_grad_scale)

    episode = 0
    ep_return = np.zeros(num_agents, dtype=np.float64)
    ep_safety = []
    ep_unsafe = []
    ep_reach = []
    ep_h = []
    ep_gate = []
    completed = []
    buffers = {key: [] for key in (
        'obs', 'h', 'grad_h', 'task_feat', 'action_raw', 'action_env',
        'log_prob', 'value', 'cost', 'reward', 'task_reward', 'task_gate',
        'cost_target', 'done', 'h_next', 'delta_state', 'agent_states',
        'task_value')}
    h_graphs: list[dict] = []
    h_next_graphs: list[dict] = []
    h_labels: list[np.ndarray] = []
    h_hdot_mask: list[float] = []
    last_done = False

    for _ in range(steps):
        prev_states = env.get_agent_states()
        graph_t = env.get_graph()
        if collect_h:
            unsafe_t = np.asarray(
                env._env.unsafe_mask(graph_t), dtype=bool)
            collision_t = np.asarray(
                env._env.collision_mask(graph_t), dtype=bool)
            label_t = np.where(
                unsafe_t | collision_t, -1.0, 1.0).astype(np.float32)
            h_graphs.append(graph_to_arrays(graph_t))
        obs_t = _tensor(obs, device)
        h_t = _tensor(h_policy, device).unsqueeze(-1)
        grad_t = _tensor(grad_policy, device)
        with torch.no_grad():
            out = model.sample(
                obs_t, h_t, grad_t, extra=task_feat.detach())
            value = out['value']
            cost_pred = out['cost']
        env_action = out['env_action'].detach().cpu().numpy()
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        h_next, grad_next = hnet.value_and_local_grad()
        h_next_policy = (h_next * h_feature_scale if use_h_features
                         else np.zeros_like(h_next))
        grad_next_policy = (grad_next * h_feature_scale if use_h_features
                            else np.zeros_like(grad_next))
        dstate = env.get_agent_states() - prev_states
        done = bool(terminated or truncated)
        last_done = done
        if collect_h:
            h_next_graphs.append(graph_to_arrays(env.get_graph()))
            h_labels.append(label_t)
            h_hdot_mask.append(0.0 if done else 1.0)

        collision_target = np.asarray(info['collision'], dtype=np.float32)
        reach = np.asarray(info['reached'], dtype=np.float32)
        # Optional dense safety cost from the frozen V6 h.  This is a
        # safety-only signal: it never uses task return, h is not updated by
        # the actor, and the no-h control disables it.  The cost critic and
        # reward both see the same non-negative safety cost.
        barrier_coef = float(training.get('h_barrier_coef', 0.0))
        if barrier_coef > 0.0 and use_h_features:
            barrier_margin = float(training.get('h_barrier_margin', 0.05))
            barrier = np.maximum(
                0.0, barrier_margin - np.asarray(h, dtype=np.float32))
            cost_target = collision_target + barrier_coef * barrier
        else:
            cost_target = collision_target
        progress_reward = reward * float(training['reward_scale'])
        reach_reward = float(training.get('reach_bonus', 0.0)) * reach
        terminal_reward = np.zeros_like(reach_reward, dtype=np.float32)
        if terminated:
            terminal_reward = (
                float(training.get('terminal_reach_bonus', 0.0)) * reach)
        # Ant-style safety gate.  Stage A uses progress_only: dense progress
        # is gated, while reach and terminal/success bonuses stay unchanged.
        # The official collision/unsafe cost is added afterwards and is never
        # gated.  h_next is the predictive V6 h at the post-transition state.
        task_gate = safety_task_gate(
            h_next, task_gate_low, task_gate_high)
        task_reward = apply_task_gate_components(
            progress_reward, reach_reward, terminal_reward, task_gate,
            scope=task_gate_scope,
            positive_only=task_gate_positive_only)
        # Generic task reward plus the environment/dense safety cost.
        ppo_reward = task_reward - (
            float(training.get('unsafe_penalty', 0.0)) * cost_target)

        buffers['obs'].append(obs_t.detach().cpu())
        buffers['h'].append(_tensor(h_policy, device))
        buffers['grad_h'].append(grad_t.detach().cpu())
        buffers['task_feat'].append(task_feat.detach().cpu())
        buffers['action_raw'].append(out['raw_action'].detach().cpu())
        buffers['action_env'].append(out['env_action'].detach().cpu())
        buffers['log_prob'].append(out['log_prob'].detach().cpu())
        buffers['value'].append(value.detach().cpu())
        buffers['cost'].append(cost_pred.detach().cpu())
        buffers['reward'].append(_tensor(ppo_reward, device).cpu())
        buffers['task_reward'].append(_tensor(task_reward, device).cpu())
        buffers['task_gate'].append(_tensor(task_gate, device).cpu())
        buffers['cost_target'].append(_tensor(cost_target, device).cpu())
        buffers['done'].append(torch.tensor(done, dtype=torch.bool))
        buffers['h_next'].append(_tensor(h_next_policy, device).cpu())
        buffers['delta_state'].append(_tensor(dstate, device).cpu())
        buffers['agent_states'].append(_tensor(prev_states, device).cpu())
        buffers['task_value'].append(task_v.detach().cpu())

        ep_return += task_reward
        ep_safety.append(float(info['safety']))
        ep_unsafe.append(float(cost_target.mean()))
        ep_reach.append(float(np.asarray(info['reached']).mean()))
        ep_h.append(float(np.asarray(h).mean()))
        ep_gate.append(float(np.mean(task_gate)))

        if done:
            completed.append(dict(
                episode_return=float(ep_return.mean()),
                safety=float(np.mean(ep_safety)),
                unsafe=float(np.mean(ep_unsafe)),
                reach=float(np.mean(ep_reach)),
                h_mean=float(np.mean(ep_h)),
                task_gate_mean=float(np.mean(ep_gate)),
            ))
            episode += 1
            obs = env.reset(seed=seed + episode)
            h, grad_h = hnet.value_and_local_grad()
            h_policy = (h * h_feature_scale if use_h_features
                        else np.zeros_like(h))
            grad_policy = (grad_h * h_feature_scale if use_h_features
                           else np.zeros_like(grad_h))
            states, goals, lidar = _state_tensors(env, device)
            task_v, a_t, task_feat = _task_channel(
                task_net, ema, states, goals, lidar,
                max_neighbors=max_neighbors, n_rays=n_rays,
                comm_radius=comm_radius, use_scalar=use_task_scalar,
                use_grad=use_task_grad, grad_scale=task_grad_scale)
            ep_return[:] = 0.0
            ep_safety.clear()
            ep_unsafe.clear()
            ep_reach.clear()
            ep_h.clear()
            ep_gate.clear()
        else:
            obs = next_obs
            h, grad_h = h_next, grad_next
            h_policy, grad_policy = h_next_policy, grad_next_policy
            states, goals, lidar = _state_tensors(env, device)
            task_v, a_t, task_feat = _task_channel(
                task_net, ema, states, goals, lidar,
                max_neighbors=max_neighbors, n_rays=n_rays,
                comm_radius=comm_radius, use_scalar=use_task_scalar,
                use_grad=use_task_grad, grad_scale=task_grad_scale)

    if last_done:
        # Rollout buffers are CPU tensors; keep the bootstrap values on CPU too.
        last_value = torch.zeros(num_agents, dtype=torch.float32)
        last_cost = torch.zeros_like(last_value)
        last_task_value = torch.zeros_like(last_value)
    else:
        with torch.no_grad():
            out = model(
                _tensor(obs, device), _tensor(h_policy, device).unsqueeze(-1),
                _tensor(grad_policy, device), extra=task_feat.detach())
        last_value = out['value'].detach().cpu()
        last_cost = out['cost'].detach().cpu()
        last_task_value = task_v.detach().cpu()

    return V6Rollout(
        obs=torch.stack(buffers['obs']).float(),
        h=torch.stack(buffers['h']).float(),
        grad_h=torch.stack(buffers['grad_h']).float(),
        task_feat=torch.stack(buffers['task_feat']).float(),
        action_raw=torch.stack(buffers['action_raw']).float(),
        action_env=torch.stack(buffers['action_env']).float(),
        log_prob=torch.stack(buffers['log_prob']).float(),
        value=torch.stack(buffers['value']).float(),
        cost=torch.stack(buffers['cost']).float(),
        reward=torch.stack(buffers['reward']).float(),
        task_reward=torch.stack(buffers['task_reward']).float(),
        task_gate=torch.stack(buffers['task_gate']).float(),
        cost_target=torch.stack(buffers['cost_target']).float(),
        done=torch.stack(buffers['done']),
        h_next=torch.stack(buffers['h_next']).float(),
        delta_state=torch.stack(buffers['delta_state']).float(),
        agent_states=torch.stack(buffers['agent_states']).float(),
        task_value=torch.stack(buffers['task_value']).float(),
        last_value=last_value.float(),
        last_cost_value=last_cost.float(),
        last_task_value=last_task_value.float(),
        completed=completed,
        h_graph=h_graphs if collect_h else None,
        h_next_graph=h_next_graphs if collect_h else None,
        h_labels=(np.stack(h_labels) if collect_h else None),
        h_hdot_mask=(np.asarray(h_hdot_mask, dtype=np.float32)
                     if collect_h else None),
    )


def add_gae_v6(rollout: V6Rollout, gamma=0.99, lam=0.95):
    advantage, returns = _gae(
        rollout.reward, rollout.value, rollout.done, rollout.last_value,
        gamma, lam)
    cost_advantage, cost_returns = _gae(
        rollout.cost_target, rollout.cost, rollout.done,
        rollout.last_cost_value, gamma, lam)
    task_advantage, task_returns = _gae(
        rollout.task_reward, rollout.task_value, rollout.done,
        rollout.last_task_value, gamma, lam)
    rollout.advantage = advantage
    rollout.returns = returns
    rollout.cost_advantage = cost_advantage
    rollout.cost_returns = cost_returns
    rollout.task_advantage = task_advantage
    rollout.task_returns = task_returns
    return rollout


def ppo_update_v6(
    model: QpFreePolicy,
    optimizer: torch.optim.Optimizer,
    rollout: V6Rollout,
    config: dict[str, Any],
    device,
):
    training = config['training']
    use_h_losses = bool(config.get('h', {}).get('use_losses', True))
    data = rollout.flatten()
    obs = data['obs'].to(device)
    h = data['h'].to(device)
    grad_h = data['grad_h'].to(device)
    task_feat = data['task_feat'].to(device)
    action_raw = data['action_raw'].to(device)
    old_log_prob = data['log_prob'].to(device)
    old_value = data['value'].to(device)
    old_cost = data['cost'].to(device)
    h_next = data['h_next'].to(device)
    delta_state = data['delta_state'].to(device)
    agent_states = data['agent_states'].to(device)
    cbf_coef = float(training.get('cbf_coef', 0.0))
    cbf_alpha = float(training.get('cbf_alpha', 1.0))
    cbf_eps = float(training.get('cbf_eps', 0.02))
    mass = float(training.get('mass', 0.1))

    adv = _normalize(rollout.advantage.reshape(-1).to(device))
    ret = rollout.returns.reshape(-1).to(device)
    cost_adv = _normalize(rollout.cost_advantage.reshape(-1).to(device))
    cost_ret = rollout.cost_returns.reshape(-1).to(device)

    n = obs.shape[0]
    batch_size = min(int(training['minibatch_size']), n)
    logs = []
    model.train()
    for _ in range(int(training['epochs'])):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            out = model.evaluate_actions(
                obs[idx], h[idx].unsqueeze(-1), grad_h[idx], action_raw[idx],
                extra=task_feat[idx])
            ratio = torch.exp(out['log_prob'] - old_log_prob[idx])
            clipped = torch.clamp(
                ratio, 1.0 - training['clip_range'],
                1.0 + training['clip_range'])
            policy_loss = -torch.min(
                ratio * adv[idx], clipped * adv[idx]).mean()
            value_clipped = old_value[idx] + torch.clamp(
                out['value'] - old_value[idx],
                -training['clip_range'], training['clip_range'])
            value_loss = 0.5 * torch.maximum(
                (out['value'] - ret[idx]).square(),
                (value_clipped - ret[idx]).square()).mean()
            cost_clipped = old_cost[idx] + torch.clamp(
                out['cost'] - old_cost[idx],
                -training['clip_range'], training['clip_range'])
            cost_loss = 0.5 * torch.maximum(
                (out['cost'] - cost_ret[idx]).square(),
                (cost_clipped - cost_ret[idx]).square()).mean()
            cost_actor = (
                torch.clamp(ratio, 0.0, 2.0) * cost_adv[idx]).mean()

            if use_h_losses:
                inv = forward_invariance_loss(
                    h[idx], h_next[idx], alpha=training['alpha'])
                grad_loss = gradient_direction_loss(
                    h[idx], grad_h[idx], delta_state[idx],
                    margin=training['gradient_margin'])
                barrier = barrier_loss(
                    h[idx], margin=training.get('barrier_margin', 0.1))
            else:
                zero = obs.new_zeros(())
                inv = grad_loss = barrier = zero
            corr = correction_penalty(out['delta'])
            entropy = out['entropy'].mean()
            if cbf_coef > 0:
                # Action-space CBF condition.  Unlike the old h losses, this
                # depends on the actor's action and therefore has a non-zero
                # gradient w.r.t. the policy.  DoubleIntegrator:
                # g = [0, I/m], so h_dot = grad_p(h).v + grad_v(h).u/m.
                _, cbf_viol = cbf_violation_torch(
                    h[idx], grad_h[idx], agent_states[idx], out['action'],
                    mass=mass, alpha=cbf_alpha, eps=cbf_eps)
                cbf = cbf_viol.mean()
            else:
                cbf = obs.new_zeros(())

            actor_loss = (
                policy_loss
                + training['cost_actor_coef'] * cost_actor
                + training['invariance_coef'] * inv
                + training['gradient_coef'] * grad_loss
                + training.get('barrier_coef', 0.0) * barrier
                + cbf_coef * cbf
                + training['correction_coef'] * corr
            )
            loss = (
                actor_loss
                + training['value_coef'] * value_loss
                + training['cost_coef'] * cost_loss
                - training['entropy_coef'] * entropy
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss: {loss.item()}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training['max_grad_norm'])
            optimizer.step()
            logs.append(dict(
                loss=float(loss.detach()),
                policy=float(policy_loss.detach()),
                value=float(value_loss.detach()),
                cost=float(cost_loss.detach()),
                invariance=float(inv.detach()),
                gradient=float(grad_loss.detach()),
                barrier=float(barrier.detach()),
                cbf=float(cbf.detach()),
                correction=float(corr.detach()),
                entropy=float(entropy.detach()),
                ratio=float(ratio.mean().detach()),
            ))
    model.eval()
    return {key: float(np.mean([row[key] for row in logs])) for key in logs[0]}


def task_value_update(
    task_net: LocalTaskValueNet,
    optimizer: torch.optim.Optimizer,
    rollout: V6Rollout,
    *,
    device,
    epochs: int = 4,
    batch_size: int = 256,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    data = rollout.flatten()
    obs = data['obs'].to(device)
    target = rollout.task_returns.reshape(-1).detach().to(device)
    n = obs.shape[0]
    logs = []
    task_net.train()
    for _ in range(int(epochs)):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            pred = task_net(obs[idx])
            loss = 0.5 * (pred - target[idx]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(task_net.parameters(), max_grad_norm)
            optimizer.step()
            logs.append(dict(loss=float(loss.detach()),
                             value_mean=float(pred.detach().mean()),
                             target_mean=float(target[idx].mean())))
    task_net.eval()
    return {key: float(np.mean([row[key] for row in logs])) for key in logs[0]}


@torch.no_grad()
def evaluate_v6(
    env,
    hnet,
    task_net: LocalTaskValueNet,
    ema: TaskAdvantageEMA,
    model: QpFreePolicy,
    seeds,
    device,
    *,
    reward_scale: float = 1.0,
    reach_bonus: float = 0.0,
    use_h_features: bool = True,
    h_feature_scale: float = 1.0,
    max_neighbors: int = 5,
    n_rays: int = 32,
    task_grad_scale: float = 1.0,
    use_task_scalar: bool = True,
    use_task_grad: bool = True,
    cbf_alpha: float = 1.0,
    cbf_eps: float = 0.02,
    mass: float | None = None,
    stochastic: bool = False,
):
    """Evaluate with V6 h features; deterministic mean action by default.

    ``stochastic=True`` samples from the policy distribution, matching the
    Ant protocol's separate stochastic robustness evaluation.  The gate is
    never active during evaluation in either mode.
    """
    model.eval()
    task_net.eval()
    if mass is None:
        mass = float(env._env.params.get('m', 0.1))
    dt = float(getattr(env, 'dt', 0.03))
    rows = []
    for seed in seeds:
        obs = env.reset(seed=int(seed))
        h, grad_h = hnet.value_and_local_grad()
        h_policy = (h * h_feature_scale if use_h_features
                    else np.zeros_like(h))
        grad_policy = (grad_h * h_feature_scale if use_h_features
                       else np.zeros_like(grad_h))
        states, goals, lidar = _state_tensors(env, device)
        task_v, a_t, task_feat = _task_channel(
            task_net, ema, states, goals, lidar,
            max_neighbors=max_neighbors, n_rays=n_rays,
            comm_radius=float(env.comm_radius),
            use_scalar=use_task_scalar, use_grad=use_task_grad,
            grad_scale=task_grad_scale)
        total_return = np.zeros(env.num_agents, dtype=np.float64)
        safe_mask_values = []
        collision_values = []
        warning_values = []
        h_values = []
        action_abs = []
        cbf_pred_sat = []
        cbf_true_sat = []
        cbf_feasible = []
        cbf_violation = []
        grad_vel_norm = []
        reached_any = np.zeros(env.num_agents, dtype=bool)
        reached_final = np.zeros(env.num_agents, dtype=bool)
        collision_any = np.zeros(env.num_agents, dtype=bool)
        collision_final = np.zeros(env.num_agents, dtype=bool)
        warning_any = np.zeros(env.num_agents, dtype=bool)
        for _ in range(env.max_step):
            obs_t = _tensor(obs, device)
            h_t = _tensor(h_policy, device).unsqueeze(-1)
            grad_t = _tensor(grad_policy, device)
            if stochastic:
                out = model.sample(
                    obs_t, h_t, grad_t, extra=task_feat.detach())
                action = out['env_action'].detach().cpu().numpy()
            else:
                out = model(obs_t, h_t, grad_t, extra=task_feat.detach())
                action = out['action'].detach().cpu().numpy()
            action_abs.append(float(np.abs(action).mean()))
            state_np = env.get_agent_states()
            h_dot_pred, violation, feasible = cbf_terms_numpy(
                h, grad_h, state_np, action, mass=mass,
                alpha=cbf_alpha, eps=cbf_eps)
            cbf_pred_sat.append(float(np.mean(h_dot_pred + cbf_alpha * h >= 0.0)))
            cbf_feasible.append(float(np.mean(feasible)))
            cbf_violation.append(float(np.mean(violation)))
            grad_vel_norm.append(float(np.mean(
                np.linalg.norm(grad_h[:, 2:] / mass, axis=1))))
            obs, reward, terminated, truncated, info = env.step(action)
            reached = np.asarray(info['reached'], dtype=bool)
            collision_step = np.asarray(info['collision'], dtype=bool)
            warning_step = np.asarray(info['unsafe'], dtype=bool)
            reached_any |= reached
            reached_final = reached
            collision_any |= collision_step
            collision_final = collision_step
            warning_any |= warning_step
            total_return += (reward * float(reward_scale)
                             + float(reach_bonus) * reached)
            safe_mask_values.append(float(info['safety']))
            collision_values.append(float(collision_step.mean()))
            warning_values.append(float(warning_step.mean()))
            h_values.append(float(np.asarray(h).mean()))
            h_prev = h
            h, grad_h = hnet.value_and_local_grad()
            h_dot_true = (h - h_prev) / dt
            cbf_true_sat.append(float(np.mean(
                h_dot_true + cbf_alpha * h_prev >= 0.0)))
            h_policy = (h * h_feature_scale if use_h_features
                        else np.zeros_like(h))
            grad_policy = (grad_h * h_feature_scale if use_h_features
                           else np.zeros_like(grad_h))
            states, goals, lidar = _state_tensors(env, device)
            task_v, a_t, task_feat = _task_channel(
                task_net, ema, states, goals, lidar,
                max_neighbors=max_neighbors, n_rays=n_rays,
                comm_radius=float(env.comm_radius),
                use_scalar=use_task_scalar, use_grad=use_task_grad,
                grad_scale=task_grad_scale)
            if terminated or truncated:
                break
        collision_rate = float(np.mean(collision_values))
        collision_episode = bool(collision_any.any())
        rows.append(dict(
            episode_return=float(total_return.mean()),
            safety=1.0 - collision_rate,
            unsafe=collision_rate,
            safe_mask=float(np.mean(safe_mask_values)),
            warning=float(np.mean(warning_values)),
            collision_episode=float(collision_episode),
            safe_episode=float(not collision_episode),
            warning_episode=float(warning_any.any()),
            collision_final_any=float(collision_final.any()),
            reach=float(reached_any.mean()),
            reach_episode_any=float(reached_any.any()),
            reach_final=float(reached_final.mean()),
            reach_final_any=float(reached_final.any()),
            # Legacy JSON field names for environment safety metrics; these do
            # not refer to official h.  Primary V6 metrics are
            # collision_episode / safe_episode.
            official_safety=float((~collision_any).mean()),
            official_reach=float(reached_any.mean()),
            official_success=float(((~collision_any) & reached_any).mean()),
            h_mean=float(np.mean(h_values)),
            h_min=float(np.min(h_values)),
            action_abs=float(np.mean(action_abs)),
            cbf_pred_sat=float(np.mean(cbf_pred_sat)),
            cbf_true_sat=float(np.mean(cbf_true_sat)),
            cbf_feasible=float(np.mean(cbf_feasible)),
            cbf_violation=float(np.mean(cbf_violation)),
            grad_vel_norm=float(np.mean(grad_vel_norm)),
        ))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def save_v6_checkpoint(
    path,
    model: QpFreePolicy,
    task_net: LocalTaskValueNet,
    ema: TaskAdvantageEMA,
    config: dict[str, Any],
    steps: int,
    obs_dim: int,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        schema='v6-qp-free-task-feature',
        model=model.state_dict(),
        task_net=task_net.state_dict(),
        task_ema=ema.state_dict(),
        config=config,
        obs_dim=int(obs_dim),
        extra_dim=int(model.extra_dim),
        steps=int(steps),
    ), path)


def load_v6_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt['config']
    extra_dim = int(ckpt.get('extra_dim', 5))
    model = QpFreePolicy(
        ckpt['obs_dim'], hidden=config['policy']['hidden'],
        correction_radius=config['policy']['correction_radius'],
        extra_dim=extra_dim).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    task_state = ckpt['task_net']
    task_cfg = dict(config.get('task_value', {}))
    # Old checkpoints were trained with the MLP V_T; infer from parameter names
    # so both architectures remain loadable.
    if any(k.startswith('agent_enc.') for k in task_state):
        task_cfg['arch'] = 'graph'
    elif 'arch' not in task_cfg:
        task_cfg['arch'] = 'mlp'
    env_cfg = config.get('env', {})
    task_net = build_task_value_net(
        task_cfg, ckpt['obs_dim'],
        max_neighbors=int(env_cfg.get('max_neighbors', 5)),
        n_rays=int(env_cfg.get('n_rays', 32))).to(device)
    task_net.load_state_dict(task_state)
    task_net.eval()
    ema = TaskAdvantageEMA.from_state_dict(ckpt['task_ema'])
    return model, task_net, ema, ckpt
