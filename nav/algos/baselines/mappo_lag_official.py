"""MAPPO-Lagrangian baseline on the official GCBF+ DoubleIntegrator.

This is a centralized-critic MAPPO actor with a learned cost critic and a
Lagrangian multiplier. The reward is the pure task reward (progress + reach),
and the cost is the official collision indicator. There is no h(s), no
correction head, and no QP.

The cost constraint is specified per environment step in
``training.cost_limit``. The multiplier is updated after each rollout:
    lambda <- max(0, lambda + lambda_lr * (mean_cost - cost_limit)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from networks.base_policy import BasePolicyNetwork, Critic

from .mappo_official import evaluate, global_state


@dataclass
class LagRollout:
    obs: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    costs: torch.Tensor
    cost_values: torch.Tensor
    dones: torch.Tensor
    last_value: torch.Tensor
    last_cost_value: torch.Tensor
    completed: list[dict[str, float]]

    def flatten(self):
        steps, agents = self.obs.shape[:2]
        return dict(
            obs=self.obs.reshape(steps * agents, -1),
            states=self.states[:, None, :].expand(
                steps, agents, self.states.shape[-1]).reshape(
                    steps * agents, -1),
            actions=self.actions.reshape(steps * agents, -1),
            log_probs=self.log_probs.reshape(-1),
            old_values=self.values[:, None].expand(
                steps, agents).reshape(-1),
            old_cost_values=self.cost_values[:, None].expand(
                steps, agents).reshape(-1),
        )


class MappoLagPolicy(nn.Module):
    def __init__(self, obs_dim, state_dim, config):
        super().__init__()
        policy = config['policy']
        self.base = BasePolicyNetwork(
            obs_dim, state_dim, 2,
            dict(
                actor_hidden_dims=policy.get('actor_hidden_dims', [256, 256]),
                critic_hidden_dims=policy.get('critic_hidden_dims', [256, 256]),
                actor_activation=policy.get('actor_activation', 'tanh'),
                critic_activation=policy.get('critic_activation', 'tanh'),
                actor_init_gain=policy.get('actor_init_gain', np.sqrt(2)),
                actor_output_gain=policy.get('actor_output_gain', 0.01),
                use_value_norm=False,
            ))
        self.cost_critic = Critic(
            state_dim,
            hidden_dims=policy.get('cost_hidden_dims', [256, 256]),
            activation=policy.get('critic_activation', 'tanh'),
            use_value_norm=False)
        with torch.no_grad():
            self.base.actor.action_log_std.fill_(
                float(policy.get('initial_log_std', -0.5)))

    @property
    def actor(self):
        return self.base.actor

    def value(self, state):
        return self.base.critic(state)

    def cost_value(self, state):
        return self.cost_critic(state)


def build_mappo_lag_model(obs_dim, state_dim, config):
    return MappoLagPolicy(obs_dim, state_dim, config)


def _reward(progress, reached, terminated, config):
    training = config['training']
    reward = (np.asarray(progress, dtype=np.float32)
              * float(training.get('reward_scale', 20.0))
              + float(training.get('reach_bonus', 5.0))
              * np.asarray(reached, dtype=np.float32))
    if terminated:
        reward += (float(training.get('terminal_reach_bonus', 20.0))
                   * np.asarray(reached, dtype=np.float32))
    return reward.astype(np.float32)


def collect_rollout(env, model, steps, seed, device, config):
    """Collect full-horizon episodes with task reward and collision cost."""
    model.eval()
    num_agents = env.num_agents
    obs = env.reset(seed=seed)
    state = global_state(env)
    completed = []
    ep_return = 0.0
    ep_collision_any = np.zeros(num_agents, dtype=bool)
    ep_reached_any = np.zeros(num_agents, dtype=bool)
    ep_collision_steps = []
    buffers = {key: [] for key in (
        'obs', 'states', 'actions', 'log_probs', 'values', 'rewards',
        'costs', 'cost_values', 'dones')}
    last_done = False

    for _ in range(steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            actions, log_prob = model.actor.get_action(
                obs_t, deterministic=False)
            value = model.base.critic(
                state_t.unsqueeze(0)).squeeze(-1)
            cost_value = model.cost_critic(
                state_t.unsqueeze(0)).squeeze(-1)
        action_np = actions.detach().cpu().numpy()
        next_obs, progress, terminated, truncated, info = env.step(action_np)
        done = bool(terminated or truncated)
        last_done = done
        reached = np.asarray(info['reached'], dtype=bool)
        collision = np.asarray(info['collision'], dtype=bool)
        step_reward = _reward(progress, reached, terminated, config)
        cost = float(np.mean(collision))
        team_reward = float(np.mean(step_reward))

        buffers['obs'].append(obs_t.detach().cpu())
        buffers['states'].append(state_t.detach().cpu())
        buffers['actions'].append(actions.detach().cpu())
        buffers['log_probs'].append(log_prob.detach().cpu().reshape(-1))
        buffers['values'].append(value.detach().cpu().reshape(()))
        buffers['rewards'].append(torch.tensor(team_reward, dtype=torch.float32))
        buffers['costs'].append(torch.tensor(cost, dtype=torch.float32))
        buffers['cost_values'].append(cost_value.detach().cpu().reshape(()))
        buffers['dones'].append(torch.tensor(done, dtype=torch.bool))

        ep_return += team_reward
        ep_collision_any |= collision
        ep_reached_any |= reached
        ep_collision_steps.append(cost)
        if done:
            completed.append(dict(
                episode_return=float(ep_return),
                official_safety=float((~ep_collision_any).mean()),
                official_reach=float(ep_reached_any.mean()),
                official_success=float(
                    ((~ep_collision_any) & ep_reached_any).mean()),
                safe_episode=float(not ep_collision_any.any()),
                collision_episode=float(ep_collision_any.any()),
                per_step_collision=float(np.mean(ep_collision_steps)),
            ))
            obs = env.reset(seed=seed + len(completed))
            state = global_state(env)
            ep_return = 0.0
            ep_collision_any[:] = False
            ep_reached_any[:] = False
            ep_collision_steps.clear()
        else:
            obs = next_obs
            state = global_state(env)

    if last_done:
        last_value = torch.tensor(0.0)
        last_cost_value = torch.tensor(0.0)
    else:
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
            last_value = model.base.critic(
                state_t.unsqueeze(0)).squeeze(-1).detach().cpu()
            last_cost_value = model.cost_critic(
                state_t.unsqueeze(0)).squeeze(-1).detach().cpu()
    return LagRollout(
        obs=torch.stack(buffers['obs']).float(),
        states=torch.stack(buffers['states']).float(),
        actions=torch.stack(buffers['actions']).float(),
        log_probs=torch.stack(buffers['log_probs']).float(),
        values=torch.stack(buffers['values']).float(),
        rewards=torch.stack(buffers['rewards']).float(),
        costs=torch.stack(buffers['costs']).float(),
        cost_values=torch.stack(buffers['cost_values']).float(),
        dones=torch.stack(buffers['dones']),
        last_value=last_value.float(),
        last_cost_value=last_cost_value.float(),
        completed=completed,
    )


def _gae(rewards, values, dones, last_value, gamma, lam):
    steps = rewards.shape[0]
    advantage = torch.zeros_like(rewards)
    gae = torch.zeros((), dtype=rewards.dtype)
    for t in range(steps - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < steps else last_value
        nonterminal = 1.0 - dones[t].to(rewards.dtype)
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantage[t] = gae
    return advantage, advantage + values


def add_gae(rollout, gamma=0.99, lam=0.95):
    rollout.advantage, rollout.returns = _gae(
        rollout.rewards, rollout.values, rollout.dones,
        rollout.last_value, gamma, lam)
    rollout.cost_advantage, rollout.cost_returns = _gae(
        rollout.costs, rollout.cost_values, rollout.dones,
        rollout.last_cost_value, gamma, lam)
    return rollout


def _normalize(x):
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


def ppo_update(model, optimizer, rollout, config, device, multiplier):
    training = config['training']
    data = rollout.flatten()
    obs = data['obs'].to(device)
    states = data['states'].to(device)
    actions = data['actions'].to(device)
    old_log_probs = data['log_probs'].to(device)
    old_values = data['old_values'].to(device)
    old_cost_values = data['old_cost_values'].to(device)
    steps = rollout.rewards.shape[0]
    agents = rollout.obs.shape[1]
    advantage = _normalize(rollout.advantage.reshape(-1).to(device))
    advantage = advantage[:, None].expand(steps, agents).reshape(-1)
    returns = rollout.returns[:, None].expand(
        steps, agents).reshape(-1).to(device)
    cost_adv = _normalize(rollout.cost_advantage.reshape(-1).to(device))
    cost_adv = cost_adv[:, None].expand(steps, agents).reshape(-1)
    cost_returns = rollout.cost_returns[:, None].expand(
        steps, agents).reshape(-1).to(device)

    n = obs.shape[0]
    batch_size = min(int(training['minibatch_size']), n)
    logs = []
    model.train()
    for _ in range(int(training['epochs'])):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            log_prob, value, entropy = model.base.evaluate_actions(
                obs[idx], states[idx], actions[idx])
            cost_value = model.cost_critic(states[idx])
            log_prob = log_prob.squeeze(-1)
            value = value.squeeze(-1)
            cost_value = cost_value.squeeze(-1)
            entropy = entropy.squeeze(-1)
            ratio = torch.exp(log_prob - old_log_probs[idx])
            clipped = torch.clamp(ratio, 1.0 - training['clip_range'],
                                  1.0 + training['clip_range'])
            policy_loss = -torch.min(
                ratio * advantage[idx], clipped * advantage[idx]).mean()
            value_clipped = old_values[idx] + torch.clamp(
                value - old_values[idx],
                -training['clip_range'], training['clip_range'])
            value_loss = 0.5 * torch.maximum(
                (value - returns[idx]).square(),
                (value_clipped - returns[idx]).square()).mean()
            cost_clipped = old_cost_values[idx] + torch.clamp(
                cost_value - old_cost_values[idx],
                -training['clip_range'], training['clip_range'])
            cost_loss = 0.5 * torch.maximum(
                (cost_value - cost_returns[idx]).square(),
                (cost_clipped - cost_returns[idx]).square()).mean()
            cost_actor = (torch.clamp(ratio, 0.0, 2.0)
                          * cost_adv[idx]).mean()
            loss = (policy_loss
                    + training['value_coef'] * value_loss
                    + training.get('cost_value_coef', 0.5) * cost_loss
                    + float(multiplier) * cost_actor
                    - training['entropy_coef'] * entropy.mean())
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
                cost_value=float(cost_loss.detach()),
                cost_actor=float(cost_actor.detach()),
                entropy=float(entropy.mean().detach()),
                ratio=float(ratio.mean().detach()),
            ))
    model.eval()
    return {key: float(np.mean([row[key] for row in logs]))
            for key in logs[0]}


def save_mappo_lag_checkpoint(path, model, config, steps, obs_dim, state_dim,
                              multiplier):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        schema='v5-mappo-lag-official',
        model=model.state_dict(),
        config=config,
        obs_dim=int(obs_dim),
        state_dim=int(state_dim),
        steps=int(steps),
        multiplier=float(multiplier),
    ), path)


def load_mappo_lag_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = build_mappo_lag_model(
        checkpoint['obs_dim'], checkpoint['state_dim'], config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


__all__ = [
    'LagRollout',
    'MappoLagPolicy',
    'add_gae',
    'build_mappo_lag_model',
    'collect_rollout',
    'evaluate',
    'load_mappo_lag_checkpoint',
    'ppo_update',
    'save_mappo_lag_checkpoint',
]
