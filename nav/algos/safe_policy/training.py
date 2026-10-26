"""Training utilities for the V5 QP-free policy.

The policy is trained against the official GCBF+ DoubleIntegrator environment.
The official h(s) is frozen; it only supplies h and grad h as features/losses.
No QP solver, QP teacher, or expert u_ref is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .qp_free import (
    QpFreePolicy,
    barrier_loss,
    correction_penalty,
    forward_invariance_loss,
    gradient_direction_loss,
)
from v6.safety_labels import cbf_terms_numpy


@dataclass
class Rollout:
    obs: torch.Tensor
    h: torch.Tensor
    grad_h: torch.Tensor
    action_raw: torch.Tensor
    action_env: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    cost: torch.Tensor
    reward: torch.Tensor
    cost_target: torch.Tensor
    done: torch.Tensor
    h_next: torch.Tensor
    delta_state: torch.Tensor
    last_value: torch.Tensor
    last_cost_value: torch.Tensor
    completed: list[dict[str, float]]

    def flatten(self):
        t, n = self.h.shape
        out = dict(
            obs=self.obs.reshape(t * n, -1),
            h=self.h.reshape(-1),
            grad_h=self.grad_h.reshape(-1, 4),
            action_raw=self.action_raw.reshape(t * n, -1),
            log_prob=self.log_prob.reshape(-1),
            value=self.value.reshape(-1),
            cost=self.cost.reshape(-1),
            reward=self.reward.reshape(-1),
            cost_target=self.cost_target.reshape(-1),
            h_next=self.h_next.reshape(-1),
            delta_state=self.delta_state.reshape(-1, 4),
            valid=(~self.done).to(self.obs.dtype)[:, None].expand(t, n).reshape(-1),
        )
        return out


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


def collect_rollout(env, hnet, model, steps, seed, device, config):
    """Collect one rollout from the official environment.

    ``h.use_features`` and ``h.use_losses`` support controlled ablations.
    With features disabled the policy still has the same 97-dim input layout,
    but the h/grad slots are zero; evaluation can still compute h for logging.
    """
    model.eval()
    training = config['training']
    use_h_features = bool(config.get('h', {}).get('use_features', True))
    num_agents = env.num_agents
    obs = env.reset(seed=seed)
    h, grad_h = hnet.value_and_local_grad()
    h_policy = h if use_h_features else np.zeros_like(h)
    grad_policy = grad_h if use_h_features else np.zeros_like(grad_h)
    episode = 0
    ep_return = np.zeros(num_agents, dtype=np.float64)
    ep_safety = []
    ep_unsafe = []
    ep_reach = []
    ep_h = []
    completed = []

    buffers = {key: [] for key in (
        'obs', 'h', 'grad_h', 'action_raw', 'action_env', 'log_prob',
        'value', 'cost', 'reward', 'cost_target', 'done', 'h_next',
        'delta_state')}
    last_done = False

    for _ in range(steps):
        prev_states = env.get_agent_states()
        obs_t = _tensor(obs, device)
        h_t = _tensor(h_policy, device).unsqueeze(-1)
        grad_t = _tensor(grad_policy, device)
        with torch.no_grad():
            out = model.sample(obs_t, h_t, grad_t)
            value = out['value']
            cost_pred = out['cost']
        env_action = out['env_action'].detach().cpu().numpy()
        next_obs, reward, terminated, truncated, info = env.step(env_action)
        h_next, grad_next = hnet.value_and_local_grad()
        h_next_policy = h_next if use_h_features else np.zeros_like(h_next)
        grad_next_policy = grad_next if use_h_features else np.zeros_like(grad_next)
        delta_state = env.get_agent_states() - prev_states
        done = bool(terminated or truncated)
        last_done = done

        # Official evaluation safety is collision-based; use collision as the
        # cost target and keep the broader warning mask only for diagnostics.
        cost_target = np.asarray(info['collision'], dtype=np.float32)
        reach = np.asarray(info['reached'], dtype=np.float32)
        step_reward = (reward * float(training['reward_scale'])
                       + float(training.get('reach_bonus', 0.0)) * reach
                       - float(training.get('unsafe_penalty', 0.0))
                       * cost_target)
        if terminated:
            # Explicitly reward reaching a goal, matching the official
            # finish condition.
            step_reward = step_reward + (
                float(training.get('terminal_reach_bonus', 0.0)) * reach)
        buffers['obs'].append(obs_t.detach().cpu())
        buffers['h'].append(_tensor(h_policy, device))
        buffers['grad_h'].append(grad_t.detach().cpu())
        buffers['action_raw'].append(out['raw_action'].detach().cpu())
        buffers['action_env'].append(out['env_action'].detach().cpu())
        buffers['log_prob'].append(out['log_prob'].detach().cpu())
        buffers['value'].append(value.detach().cpu())
        buffers['cost'].append(cost_pred.detach().cpu())
        buffers['reward'].append(_tensor(step_reward, device).cpu())
        buffers['cost_target'].append(_tensor(cost_target, device).cpu())
        buffers['done'].append(torch.tensor(done, dtype=torch.bool))
        buffers['h_next'].append(_tensor(h_next_policy, device).cpu())
        buffers['delta_state'].append(_tensor(delta_state, device).cpu())

        ep_return += step_reward
        ep_safety.append(float(info['safety']))
        ep_unsafe.append(float(cost_target.mean()))
        ep_reach.append(float(np.asarray(info['reached']).mean()))
        ep_h.append(float(np.asarray(h).mean()))

        if done:
            completed.append(dict(
                episode_return=float(ep_return.mean()),
                safety=float(np.mean(ep_safety)),
                unsafe=float(np.mean(ep_unsafe)),
                reach=float(np.mean(ep_reach)),
                h_mean=float(np.mean(ep_h)),
            ))
            episode += 1
            obs = env.reset(seed=seed + episode)
            h, grad_h = hnet.value_and_local_grad()
            h_policy = h if use_h_features else np.zeros_like(h)
            grad_policy = grad_h if use_h_features else np.zeros_like(grad_h)
            ep_return[:] = 0.0
            ep_safety.clear()
            ep_unsafe.clear()
            ep_reach.clear()
            ep_h.clear()
        else:
            obs = next_obs
            h, grad_h = h_next, grad_next
            h_policy, grad_policy = h_next_policy, grad_next_policy

    # Bootstrap the last transition unless the rollout ended on a terminal step.
    if last_done:
        last_value = torch.zeros(num_agents, dtype=torch.float32, device=device)
        last_cost = torch.zeros_like(last_value)
    else:
        with torch.no_grad():
            out = model(
                _tensor(obs, device), _tensor(h_policy, device).unsqueeze(-1),
                _tensor(grad_policy, device))
        last_value = out['value'].detach().cpu()
        last_cost = out['cost'].detach().cpu()

    rollout = Rollout(
        obs=torch.stack(buffers['obs']).float(),
        h=torch.stack(buffers['h']).float(),
        grad_h=torch.stack(buffers['grad_h']).float(),
        action_raw=torch.stack(buffers['action_raw']).float(),
        action_env=torch.stack(buffers['action_env']).float(),
        log_prob=torch.stack(buffers['log_prob']).float(),
        value=torch.stack(buffers['value']).float(),
        cost=torch.stack(buffers['cost']).float(),
        reward=torch.stack(buffers['reward']).float(),
        cost_target=torch.stack(buffers['cost_target']).float(),
        done=torch.stack(buffers['done']),
        h_next=torch.stack(buffers['h_next']).float(),
        delta_state=torch.stack(buffers['delta_state']).float(),
        last_value=last_value.float(),
        last_cost_value=last_cost.float(),
        completed=completed,
    )
    return rollout


def add_gae(rollout, gamma=0.99, lam=0.95):
    advantage, returns = _gae(
        rollout.reward, rollout.value, rollout.done, rollout.last_value, gamma, lam)
    cost_advantage, cost_returns = _gae(
        rollout.cost_target, rollout.cost, rollout.done, rollout.last_cost_value,
        gamma, lam)
    rollout.advantage = advantage
    rollout.returns = returns
    rollout.cost_advantage = cost_advantage
    rollout.cost_returns = cost_returns
    return rollout


def _normalize(x):
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


def ppo_update(model, optimizer, rollout, config, device):
    """PPO update with h/grad safety losses. Returns scalar diagnostics."""
    training = config['training']
    use_h_losses = bool(config.get('h', {}).get('use_losses', True))
    data = rollout.flatten()
    obs = data['obs'].to(device)
    h = data['h'].to(device)
    grad_h = data['grad_h'].to(device)
    action_raw = data['action_raw'].to(device)
    old_log_prob = data['log_prob'].to(device)
    old_value = data['value'].to(device)
    old_cost = data['cost'].to(device)
    h_next = data['h_next'].to(device)
    delta_state = data['delta_state'].to(device)
    valid = data['valid'].to(device)

    adv = rollout.advantage.reshape(-1).to(device)
    ret = rollout.returns.reshape(-1).to(device)
    cost_adv = rollout.cost_advantage.reshape(-1).to(device)
    cost_ret = rollout.cost_returns.reshape(-1).to(device)
    adv = _normalize(adv)
    cost_adv = _normalize(cost_adv)

    n = obs.shape[0]
    batch_size = min(int(training['minibatch_size']), n)
    logs = []
    model.train()
    for _ in range(int(training['epochs'])):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            out = model.evaluate_actions(
                obs[idx], h[idx].unsqueeze(-1), grad_h[idx], action_raw[idx])
            ratio = torch.exp(out['log_prob'] - old_log_prob[idx])
            clipped = torch.clamp(ratio, 1.0 - training['clip_range'],
                                  1.0 + training['clip_range'])
            policy_loss = -torch.min(ratio * adv[idx], clipped * adv[idx]).mean()
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
            # Cost advantage is positive for unsafe transitions; add a small
            # penalty rather than using a hand-designed safety rule.
            cost_actor = (torch.clamp(ratio, 0.0, 2.0) * cost_adv[idx]).mean()

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

            actor_loss = (
                policy_loss
                + training['cost_actor_coef'] * cost_actor
                + training['invariance_coef'] * inv
                + training['gradient_coef'] * grad_loss
                + training.get('barrier_coef', 0.0) * barrier
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), training['max_grad_norm'])
            optimizer.step()
            logs.append(dict(
                loss=float(loss.detach()),
                policy=float(policy_loss.detach()),
                value=float(value_loss.detach()),
                cost=float(cost_loss.detach()),
                invariance=float(inv.detach()),
                gradient=float(grad_loss.detach()),
                barrier=float(barrier.detach()),
                correction=float(corr.detach()),
                entropy=float(entropy.detach()),
                ratio=float(ratio.mean().detach()),
            ))
    model.eval()
    return {key: float(np.mean([row[key] for row in logs])) for key in logs[0]}


@torch.no_grad()
def evaluate(env, hnet, model, seeds, device, reward_scale=1.0,
             reach_bonus=0.0, use_h_features=True,
             cbf_alpha=1.0, cbf_eps=0.02, mass=None):
    """Deterministic evaluation on fixed scene seeds.

    ``use_h_features`` must match the checkpoint's training ablation.  The
    official h is still computed for diagnostics when the policy does not use
    it, so h_mean/h_min remain comparable across arms.
    """
    model.eval()
    if mass is None:
        mass = float(env._env.params.get('m', 0.1))
    dt = float(getattr(env, 'dt', 0.03))
    rows = []
    for seed in seeds:
        obs = env.reset(seed=int(seed))
        h, grad_h = hnet.value_and_local_grad()
        h_policy = h if use_h_features else np.zeros_like(h)
        grad_policy = grad_h if use_h_features else np.zeros_like(grad_h)
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
            out = model(
                _tensor(obs, device), _tensor(h_policy, device).unsqueeze(-1),
                _tensor(grad_policy, device))
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
            h_policy = h if use_h_features else np.zeros_like(h)
            grad_policy = grad_h if use_h_features else np.zeros_like(grad_h)
            if terminated or truncated:
                break
        collision_rate = float(np.mean(collision_values))
        collision_episode = bool(collision_any.any())
        rows.append(dict(
            episode_return=float(total_return.mean()),
            # `safety` and `unsafe` are collision-based, matching the official
            # evaluation cost. `safe_mask` and `warning` are separately reported.
            safety=1.0 - collision_rate,
            unsafe=collision_rate,
            safe_mask=float(np.mean(safe_mask_values)),
            warning=float(np.mean(warning_values)),
            collision_episode=float(collision_episode),
            safe_episode=float(not collision_episode),
            warning_episode=float(warning_any.any()),
            collision_final_any=float(collision_final.any()),
            # Any-time agent success; also keep final-step success separately.
            reach=float(reached_any.mean()),
            reach_episode_any=float(reached_any.any()),
            reach_final=float(reached_final.mean()),
            reach_final_any=float(reached_final.any()),
            # Official GCBF+ test.py definitions: per-agent collision over the
            # whole episode, per-agent any-time reach, and their conjunction.
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


def save_checkpoint(path, model, config, steps, obs_dim):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        schema='v5-qp-free',
        model=model.state_dict(),
        config=config,
        obs_dim=int(obs_dim),
        steps=int(steps),
    ), path)


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = QpFreePolicy(
        checkpoint['obs_dim'], hidden=config['policy']['hidden'],
        correction_radius=config['policy']['correction_radius']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint
