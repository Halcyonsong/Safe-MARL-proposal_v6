"""Shared MAPPO/PPO implementation: fresh rollouts, GAE, per-agent CTDE values."""

import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from networks.cbf_network import create_graph_data, create_batched_graph_data
from networks.correction_policy import compute_correction_reward
from utils.runtime import SCHEMA_VERSION, graph_kwargs


class RolloutBuffer:
    """Transitions remain ordered as [time, environment, agent]. Never replayed."""

    def __init__(self):
        self.rows = []

    def add(self, inputs, critic_state, actions, old_log_prob, rewards,
            values, next_values, terminated, done):
        self.rows.append(dict(inputs={k: v.detach().cpu() for k, v in inputs.items()},
                              critic_state=critic_state.detach().cpu(),
                              actions=actions.detach().cpu(), old_log_prob=old_log_prob.detach().cpu(),
                              rewards=torch.as_tensor(rewards, dtype=torch.float32),
                              values=values.detach().cpu(), next_values=next_values.detach().cpu(),
                              terminated=torch.as_tensor(terminated, dtype=torch.float32),
                              done=torch.as_tensor(done, dtype=torch.float32)))

    def get(self, gamma, gae_lambda, device):
        if not self.rows:
            raise ValueError('Empty rollout')
        inputs = {k: torch.stack([r['inputs'][k] for r in self.rows]).flatten(0, 2).to(device)
                  for k in self.rows[0]['inputs']}
        data = {k: torch.stack([r[k] for r in self.rows])
                for k in self.rows[0] if k != 'inputs'}
        advantages = torch.zeros_like(data['rewards'])
        gae = torch.zeros_like(advantages[0])
        for t in reversed(range(len(self.rows))):
            # Timeout bootstraps from the *terminal* state but ends the GAE
            # recurrence. True termination does neither bootstrap nor recur.
            delta = (data['rewards'][t] + gamma*(1-data['terminated'][t])*data['next_values'][t]
                     - data['values'][t])
            gae = delta + gamma*gae_lambda*(1-data['done'][t])*gae
            advantages[t] = gae
        returns = advantages + data['values']
        advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
        flat = {k: data[k].flatten(0, 2).to(device) for k in ('critic_state', 'actions', 'old_log_prob')}
        flat.update(advantages=advantages.reshape(-1, 1).to(device), returns=returns.reshape(-1, 1).to(device))
        return inputs, flat


def cbf_features(envs, cbf_net, env_config, device):
    """Batched h and local dh_i/ds_i for all vectorized environments.

    The old implementation built one graph and ran one autograd backward per
    agent per environment. This version builds one disconnected batched graph
    and obtains all local Jacobian rows in a single batched backward pass.
    """
    with torch.enable_grad():
        states = torch.as_tensor(envs.get_agent_states(), device=device).requires_grad_(True)
        goals = torch.as_tensor(envs.get_goal(), device=device)
        obstacles = torch.as_tensor(envs.get_obstacles(), device=device)
        graph = create_batched_graph_data(states, goals, obstacles,
                                          **graph_kwargs(env_config))
        h = cbf_net(*graph).reshape(states.shape[0], states.shape[1], 1)
        n_scene, n_agent = states.shape[:2]
        n = n_scene * n_agent
        rows = torch.eye(n, device=device, dtype=states.dtype)
        grads = torch.autograd.grad(
            h.reshape(-1), states, grad_outputs=rows,
            is_grads_batched=True, create_graph=False)[0]
        idx = torch.arange(n, device=device)
        local = grads[idx, idx // n_agent, idx % n_agent].reshape_as(states)
    return h.detach(), local.detach()


class Features:
    def __init__(self, config, device, cbf_net=None, base_policy=None):
        self.config, self.device = config, device
        self.cbf_net, self.base_policy = cbf_net, base_policy

    def __call__(self, envs):
        obs = torch.as_tensor(envs.observations, device=self.device)
        global_state = torch.as_tensor(envs.get_state(), device=self.device)
        state = global_state[:, None].expand(-1, envs.num_agents, -1)
        inputs = dict(obs=obs)
        if self.cbf_net is not None:
            h, grad = cbf_features(envs, self.cbf_net, self.config['env'], self.device)
            with torch.no_grad():
                u_ref, _ = self.base_policy.get_action(obs, deterministic=True)
            inputs.update(u_ref=u_ref, h=h, grad_h=grad)
        return inputs, torch.cat([state, *inputs.values()], dim=-1)


def actor_action(actor, inputs, deterministic=False):
    return actor.get_action(*inputs.values(), deterministic=deterministic)


def task_reward(obs, next_obs, info, settings):
    progress = (np.linalg.norm(obs[..., :2]-obs[..., 4:6], axis=-1)
                - np.linalg.norm(next_obs[..., :2]-obs[..., 4:6], axis=-1))
    return (settings.get('distance_improvement', 1.)*progress
            + settings.get('success_bonus', 10.)*np.asarray(info['newly_reached'])
            + settings.get('time_penalty', -.01)*(~np.asarray(info['reached']) | np.asarray(info['newly_reached'])))


def collect_rollout(envs, model, feature_fn, config, stage, n_steps):
    buffer = RolloutBuffer()
    if not hasattr(envs, 'episode_rewards'):
        envs.episode_rewards = np.zeros((envs.num_envs, envs.num_agents))
        envs.episode_lengths = np.zeros(envs.num_envs)
    completed, correction_sizes = [], []
    inputs, state = feature_fn(envs)
    for _ in range(n_steps):
        obs = envs.observations.copy()
        physical = envs.get_agent_states()
        with torch.no_grad():
            result = actor_action(model.actor, inputs)
            actions, log_prob = result[:2]
            values = model.critic(state).squeeze(-1)
        next_obs, _, done, info = envs.step(actions.cpu().numpy())
        next_inputs, next_state = feature_fn(envs)
        with torch.no_grad():
            next_values = model.critic(next_state).squeeze(-1)
        reward = np.stack([task_reward(obs[i], next_obs[i], inf, config['stage2']['reward'])
                           for i, inf in enumerate(info)])
        if stage == 3:
            safety_settings = dict(config['stage3']['reward'])
            safety_settings['dt'] = config['env']['dt']
            safety_settings['max_acceleration'] = config['env']['max_acceleration']
            rewards = []
            for i, inf in enumerate(info):
                r, _ = compute_correction_reward(
                    physical[i], inputs['u_ref'][i].cpu().numpy(), actions[i].cpu().numpy(),
                    envs.get_agent_states()[i], inf, inputs['h'][i, :, 0].cpu().numpy(),
                    next_inputs['h'][i, :, 0].cpu().numpy(), task_reward=reward[i], **safety_settings)
                rewards.append(r)
            reward = np.stack(rewards)
            correction_sizes.append(float((actions-inputs['u_ref']).norm(dim=-1).mean()))
        terminated = np.asarray([inf['terminated'] for inf in info])[:, None]
        buffer.add(inputs, state, actions, log_prob, reward, values, next_values,
                   np.broadcast_to(terminated, reward.shape).copy(),
                   np.broadcast_to(done[:, None], reward.shape).copy())
        envs.episode_rewards += reward
        envs.episode_lengths += 1
        for i in np.flatnonzero(done):
            completed.append(dict(reward=float(envs.episode_rewards[i].mean()),
                                  length=float(envs.episode_lengths[i]),
                                  safety_rate=float(info[i]['safe'].mean()),
                                  reach_rate=float(info[i]['reached'].mean()),
                                  success_rate=float(info[i]['success'].mean())))
            envs.episode_rewards[i] = 0
            envs.episode_lengths[i] = 0
        envs.reset_done()
        if done.any():
            inputs, state = feature_fn(envs)
        else:
            inputs, state = next_inputs, next_state
    stats = {key: float(np.mean([row[key] for row in completed])) for key in completed[0]} if completed else {}
    stats['num_episodes'] = len(completed)
    if correction_sizes:
        stats['avg_correction'] = float(np.mean(correction_sizes))
    return buffer, stats


def update_policy(model, buffer, optimizer_actor, optimizer_critic, settings, device):
    inputs, data = buffer.get(settings['gamma'], settings['gae_lambda'], device)
    model.train()
    if model.critic.use_value_norm:
        model.critic.update_value_norm(data['returns'])
    stats = []
    for _ in range(settings['n_epochs']):
        order = torch.randperm(len(data['actions']), device=device)
        for idx in order.split(settings['batch_size']):
            log_prob, entropy = model.actor.evaluate_actions(
                *[value[idx] for value in inputs.values()], data['actions'][idx])
            ratio = (log_prob-data['old_log_prob'][idx]).exp()
            advantage = data['advantages'][idx]
            actor_loss = -torch.minimum(ratio*advantage,
                                        ratio.clamp(1-settings['clip_range'], 1+settings['clip_range'])*advantage).mean()
            bc_loss = torch.zeros((), device=device)
            if settings.get('bc_coef', 0.) and 'u_ref' in inputs:
                mean, _ = model.actor.distribution_parameters(
                    *[value[idx] for value in inputs.values()])
                bc_loss = (torch.tanh(mean) - inputs['u_ref'][idx]).square().mean()
            raw_value = model.critic.value_net(data['critic_state'][idx])
            target = data['returns'][idx]
            if model.critic.use_value_norm:
                target = model.critic.value_norm.normalize(target)
            value_loss = F.mse_loss(raw_value, target)
            loss = (actor_loss + settings['value_coef']*value_loss
                    - settings['entropy_coef']*entropy.mean()
                    + settings.get('bc_coef', 0.)*bc_loss)
            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.actor.parameters(), settings['max_grad_norm'])
            nn.utils.clip_grad_norm_(model.critic.parameters(), settings['max_grad_norm'])
            optimizer_actor.step()
            optimizer_critic.step()
            stats.append((float(actor_loss.detach()), float(value_loss.detach()),
                          float(entropy.mean().detach()), float(bc_loss.detach())))
    return dict(zip(('actor_loss', 'critic_loss', 'entropy', 'bc_loss'),
                    np.mean(stats, axis=0).tolist()))


def train_ppo(model, envs, features, config, stage, output_dir, device, model_config,
              resume_from=None):
    settings = config[f'stage{stage}']['training']
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(output/'tensorboard')) if config.get('logging', {}).get('use_tensorboard', True) else None
    actor_lr = settings.get('lr_actor', settings.get('lr', 3e-4))
    critic_lr = settings.get('lr_critic', actor_lr)
    optimizer_actor = torch.optim.Adam(model.actor.parameters(), lr=actor_lr, eps=settings['eps'])
    optimizer_critic = torch.optim.Adam(model.critic.parameters(), lr=critic_lr, eps=settings['eps'])
    timestep, next_save = 0, settings['save_interval']
    next_eval = settings.get('eval_interval', 0)
    n_steps = settings.get('n_steps', max(1, settings.get('update_frequency', 128)//envs.num_envs))
    if settings['total_timesteps'] < 1 or n_steps < 1:
        raise ValueError('Training budget and rollout length must be positive')
    prefix = 'base_policy' if stage == 2 else 'correction_policy'

    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)
        if checkpoint.get('schema_version') != SCHEMA_VERSION:
            raise ValueError(f'{resume_from}: incompatible checkpoint schema')
        if checkpoint.get('env_config') != config['env']:
            raise ValueError(f'{resume_from}: environment configuration differs')
        if checkpoint.get('config') != model_config:
            raise ValueError(f'{resume_from}: model configuration differs')
        model.load_state_dict(checkpoint['policy_state_dict'])
        optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
        optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
        timestep = int(checkpoint['timestep'])
        next_save = timestep + settings['save_interval']
        eval_interval = settings.get('eval_interval', 0)
        next_eval = timestep + eval_interval if eval_interval > 0 else 0
        print(f'Resuming Stage {stage} from {resume_from} at {timestep} timesteps', flush=True)

    def checkpoint():
        return dict(schema_version=SCHEMA_VERSION, policy_state_dict=model.state_dict(),
                    config=model_config, env_config=config['env'], timestep=timestep,
                    optimizer_actor=optimizer_actor.state_dict(), optimizer_critic=optimizer_critic.state_dict())

    try:
        while timestep < settings['total_timesteps']:
            if settings.get('use_linear_lr_decay', False):
                fraction = 1-timestep/settings['total_timesteps']
                optimizer_actor.param_groups[0]['lr'] = actor_lr*fraction
                optimizer_critic.param_groups[0]['lr'] = critic_lr*fraction
            steps = min(n_steps, math.ceil((settings['total_timesteps']-timestep)/envs.num_envs))
            buffer, stats = collect_rollout(envs, model, features, config, stage, steps)
            updates = update_policy(model, buffer, optimizer_actor, optimizer_critic, settings, device)
            timestep += steps*envs.num_envs  # Joint world steps, not agent samples.
            print(f'Stage {stage} steps={timestep}: {stats} {updates}', flush=True)
            if writer:
                for key, value in {**stats, **updates}.items():
                    writer.add_scalar(key, value, timestep)
            if timestep >= next_save:
                torch.save(checkpoint(), output/f'{prefix}_step_{timestep}.pth')
                next_save = timestep+settings['save_interval']
            if next_eval > 0 and timestep >= next_eval:
                from utils.evaluation import evaluate_policy
                metrics = evaluate_policy(model, features, config['env'],
                                           settings.get('eval_episodes', 100), seed=config.get('seed', 42)+10000)
                print(f'Stage {stage} evaluation: {metrics}', flush=True)
                if writer:
                    for key, value in metrics.items():
                        writer.add_scalar(f'eval/{key}', value, timestep)
                next_eval = timestep+settings['eval_interval']
        torch.save(checkpoint(), output/f'{prefix}_final.pth')
    finally:
        envs.close()
        if writer:
            writer.close()
    return model
