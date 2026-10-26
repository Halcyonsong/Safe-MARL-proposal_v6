"""Pure MAPPO baseline on the official GCBF+ DoubleIntegrator.

This is a task-only centralized-critic MAPPO baseline. It deliberately does not
load or use the fixed h(s), its gradient, a cost critic, a correction head, or a
QP solver. The actor receives only the local 92-dim observation. The centralized
critic receives a global state made from all agent states, all goals, and the
obstacle geometry.

The baseline supports two reward settings through ``training.collision_penalty``:

* ``0.0``: pure task reward (progress + reach bonus), matching the legacy
  pure-task MAPPO idea.
* ``10.0``: the same collision-penalized reward used by V5, without h or any
  safety mechanism in the policy.

Both are still pure MAPPO; only the environment reward differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from networks.base_policy import BasePolicyNetwork


@dataclass
class MappoRollout:
    obs: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    last_value: torch.Tensor
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
            valid=(~self.dones).to(self.obs.dtype)[:, None].expand(
                steps, agents).reshape(-1),
        )


def _tensor(array, device):
    return torch.as_tensor(np.array(array, copy=True), dtype=torch.float32,
                           device=device)


def _obstacle_features(env):
    obstacle = env.get_graph().env_states.obstacle
    return np.concatenate([
        np.asarray(obstacle.center, dtype=np.float32).reshape(-1),
        np.asarray(obstacle.width, dtype=np.float32).reshape(-1),
        np.asarray(obstacle.height, dtype=np.float32).reshape(-1),
        np.asarray(obstacle.theta, dtype=np.float32).reshape(-1),
    ])


def global_state_dim(env):
    """Dimension of the centralized critic state for this environment."""
    num_obstacles = int(np.asarray(
        env.get_graph().env_states.obstacle.center).shape[0])
    return 8 * env.num_agents + 5 * num_obstacles


def global_state(env):
    """Full centralized state: all agents, goals, and obstacle geometry."""
    obstacle = _obstacle_features(env)
    return np.concatenate([
        np.asarray(env.get_agent_states(), dtype=np.float32).reshape(-1),
        np.asarray(env.get_goal_states(), dtype=np.float32).reshape(-1),
        obstacle,
    ]).astype(np.float32)


def build_mappo_model(obs_dim, state_dim, config):
    policy = config['policy']
    model = BasePolicyNetwork(
        obs_dim, state_dim, 2,
        dict(
            actor_hidden_dims=policy.get('actor_hidden_dims', [256, 256]),
            critic_hidden_dims=policy.get('critic_hidden_dims', [256, 256]),
            actor_activation=policy.get('actor_activation', 'tanh'),
            critic_activation=policy.get('critic_activation', 'tanh'),
            actor_init_gain=policy.get('actor_init_gain', np.sqrt(2)),
            actor_output_gain=policy.get('actor_output_gain', 0.01),
            use_value_norm=bool(policy.get('use_value_norm', False)),
        ))
    # MAPPO's standard initial action std is smaller than the default 1.0.
    with torch.no_grad():
        model.actor.action_log_std.fill_(
            float(policy.get('initial_log_std', -0.5)))
    return model


def _reward(step_progress, step_reached, step_collision, terminated, config):
    training = config['training']
    reward = (np.asarray(step_progress, dtype=np.float32)
              * float(training.get('reward_scale', 20.0))
              + float(training.get('reach_bonus', 5.0))
              * np.asarray(step_reached, dtype=np.float32))
    penalty = float(training.get('collision_penalty', 0.0))
    if penalty:
        reward -= penalty * np.asarray(step_collision, dtype=np.float32)
    if terminated:
        reward += (float(training.get('terminal_reach_bonus', 20.0))
                   * np.asarray(step_reached, dtype=np.float32))
    return reward.astype(np.float32)


def collect_rollout(env, model, steps, seed, device, config):
    """Collect one with-replacement rollout of environment steps.

    Rewards are shared across agents (team mean), as in the standard MAPPO
    shared-reward setting. The critic is centralized; the actor remains local.
    """
    model.eval()
    num_agents = env.num_agents
    obs = env.reset(seed=seed)
    state = global_state(env)
    episode = 0
    ep_return = 0.0
    ep_collision_any = np.zeros(num_agents, dtype=bool)
    ep_reached_any = np.zeros(num_agents, dtype=bool)
    ep_collision_steps = []
    completed = []

    buffers = {key: [] for key in (
        'obs', 'states', 'actions', 'log_probs', 'values', 'rewards', 'dones')}
    last_done = False

    for _ in range(steps):
        obs_t = _tensor(obs, device)
        state_t = _tensor(state, device)
        with torch.no_grad():
            actions, log_prob = model.actor.get_action(
                obs_t, deterministic=False)
            value = model.critic(state_t.unsqueeze(0)).squeeze(-1)
        action_np = actions.detach().cpu().numpy()
        next_obs, progress, terminated, truncated, info = env.step(action_np)
        done = bool(terminated or truncated)
        last_done = done
        step_reached = np.asarray(info['reached'], dtype=bool)
        step_collision = np.asarray(info['collision'], dtype=bool)
        step_reward = _reward(
            progress, step_reached, step_collision, terminated, config)
        team_reward = float(np.mean(step_reward))

        buffers['obs'].append(obs_t.detach().cpu())
        buffers['states'].append(state_t.detach().cpu())
        buffers['actions'].append(actions.detach().cpu())
        buffers['log_probs'].append(log_prob.detach().cpu().reshape(-1))
        buffers['values'].append(value.detach().cpu().reshape(()))
        buffers['rewards'].append(torch.tensor(team_reward, dtype=torch.float32))
        buffers['dones'].append(torch.tensor(done, dtype=torch.bool))

        ep_return += team_reward
        ep_collision_any |= step_collision
        ep_reached_any |= step_reached
        ep_collision_steps.append(float(step_collision.mean()))

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
            episode += 1
            obs = env.reset(seed=seed + episode)
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
    else:
        with torch.no_grad():
            last_value = model.critic(
                _tensor(state, device).unsqueeze(0)).squeeze(-1).detach().cpu()

    rollout = MappoRollout(
        obs=torch.stack(buffers['obs']).float(),
        states=torch.stack(buffers['states']).float(),
        actions=torch.stack(buffers['actions']).float(),
        log_probs=torch.stack(buffers['log_probs']).float(),
        values=torch.stack(buffers['values']).float(),
        rewards=torch.stack(buffers['rewards']).float(),
        dones=torch.stack(buffers['dones']),
        last_value=last_value.float(),
        completed=completed,
    )
    return rollout


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
    advantage, returns = _gae(
        rollout.rewards, rollout.values, rollout.dones, rollout.last_value,
        gamma, lam)
    rollout.advantage = advantage
    rollout.returns = returns
    return rollout


def _normalize(x):
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


def ppo_update(model, optimizer, rollout, config, device):
    training = config['training']
    data = rollout.flatten()
    obs = data['obs'].to(device)
    states = data['states'].to(device)
    actions = data['actions'].to(device)
    old_log_probs = data['log_probs'].to(device)
    old_values = data['old_values'].to(device)
    steps = rollout.rewards.shape[0]
    advantage = _normalize(rollout.advantage.reshape(-1).to(device))
    advantage = advantage[:, None].expand(steps, rollout.obs.shape[1]).reshape(-1)
    returns = rollout.returns[:, None].expand(
        steps, rollout.obs.shape[1]).reshape(-1).to(device)

    n = obs.shape[0]
    batch_size = min(int(training['minibatch_size']), n)
    logs = []
    model.train()
    for _ in range(int(training['epochs'])):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            log_prob, value, entropy = model.evaluate_actions(
                obs[idx], states[idx], actions[idx])
            log_prob = log_prob.squeeze(-1)
            value = value.squeeze(-1)
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
            loss = (policy_loss
                    + training['value_coef'] * value_loss
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
                entropy=float(entropy.mean().detach()),
                ratio=float(ratio.mean().detach()),
            ))
    model.eval()
    return {key: float(np.mean([row[key] for row in logs])) for key in logs[0]}


@torch.no_grad()
def evaluate(env, model, seeds, device, reward_scale=1.0, reach_bonus=0.0):
    """Deterministic evaluation with the same official metrics as V5."""
    model.eval()
    rows = []
    for seed in seeds:
        obs = env.reset(seed=int(seed))
        total_return = np.zeros(env.num_agents, dtype=np.float64)
        safe_mask_values = []
        collision_values = []
        warning_values = []
        action_abs = []
        reached_any = np.zeros(env.num_agents, dtype=bool)
        reached_final = np.zeros(env.num_agents, dtype=bool)
        collision_any = np.zeros(env.num_agents, dtype=bool)
        collision_final = np.zeros(env.num_agents, dtype=bool)
        warning_any = np.zeros(env.num_agents, dtype=bool)
        for _ in range(env.max_step):
            obs_t = _tensor(obs, device)
            action = model.actor.get_action(
                obs_t, deterministic=True)[0].cpu().numpy()
            action_abs.append(float(np.abs(action).mean()))
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
            official_safety=float((~collision_any).mean()),
            official_reach=float(reached_any.mean()),
            official_success=float(((~collision_any) & reached_any).mean()),
            action_abs=float(np.mean(action_abs)),
        ))
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def save_mappo_checkpoint(path, model, config, steps, obs_dim, state_dim):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        schema='v5-mapP0-official',
        model=model.state_dict(),
        config=config,
        obs_dim=int(obs_dim),
        state_dim=int(state_dim),
        steps=int(steps),
    ), path)


def load_mappo_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = build_mappo_model(
        checkpoint['obs_dim'], checkpoint['state_dim'], config).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint
