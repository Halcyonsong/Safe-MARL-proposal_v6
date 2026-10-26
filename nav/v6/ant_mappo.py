"""V6 from-scratch MAPPO for Safe MAMuJoCo Ant-2x4.

The base actor is decentralized and receives only task observation plus the
frozen safety features ``h`` and ``grad_s h``.  The critic is centralized and
sees the official global Ant observation.  There is no MAPPO-Lagrangian
multiplier, no QP/CBF correction, and no task-value input to the actor in the
promoted main arms.

A separate diagnostic arm can append a frozen, offline-trained task scalar
``A_T`` to the actor input (``use_task_value=True``); it is not part of the
promoted Ant claim and does not change the critic.  Safety enters the actor
update only as a fixed cost penalty on the official cost
(``reward - cost_penalty * cost``).  The h network is frozen and is never
updated by the task/cost policy loss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from networks.base_policy import Actor, Critic
except ImportError:  # package import (`python -m nav...`)
    from nav.networks.base_policy import Actor, Critic

from .ant_safety import GEOM_DIM


def _get(cfg: dict, key: str, default: Any) -> Any:
    value = cfg.get(key, default)
    return default if value is None else value


class MappoAntModel(nn.Module):
    """Per-agent actors plus a centralized critic."""

    def __init__(
        self,
        *,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        num_agents: int,
        use_h_features: bool = True,
        use_task_value: bool = False,
        task_dim: int = 1,
        actor_hidden: tuple[int, ...] = (64, 64),
        critic_hidden: tuple[int, ...] = (128, 128),
        actor_activation: str = 'tanh',
        critic_activation: str = 'tanh',
        initial_log_std: float = -0.5,
        h_feature_scale: float = 1.0,
        grad_feature_scale: float = 1.0,
        actor_feature_norm: bool = False,
        actor_layer_norm: bool = False,
        critic_feature_norm: bool = False,
        critic_layer_norm: bool = False,
        use_value_norm: bool = False,
        action_dist: str = 'bounded',
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.num_agents = int(num_agents)
        self.use_h_features = bool(use_h_features)
        self.use_task_value = bool(use_task_value)
        self.task_dim = int(task_dim)
        if self.use_task_value and self.task_dim <= 0:
            raise ValueError('use_task_value=true requires task_dim > 0')
        if not self.use_task_value and self.task_dim != 0:
            # Keep the control path's rollout representation unambiguous.
            self.task_dim = 0
        self.h_feature_scale = float(h_feature_scale)
        self.grad_feature_scale = float(grad_feature_scale)
        self.actor_activation = str(actor_activation)
        self.critic_activation = str(critic_activation)
        self.actor_feature_norm = bool(actor_feature_norm)
        self.actor_layer_norm = bool(actor_layer_norm)
        self.critic_feature_norm = bool(critic_feature_norm)
        self.critic_layer_norm = bool(critic_layer_norm)
        self.use_value_norm = bool(use_value_norm)
        self.action_dist = str(action_dist)
        if self.action_dist not in ('bounded', 'gaussian'):
            raise ValueError(
                f"unknown action_dist {self.action_dist!r}; "
                "expected 'bounded' or 'gaussian'")
        self.bounded_actions = self.action_dist == 'bounded'
        self.base_input_dim = (
            self.obs_dim + 1 + GEOM_DIM
            if self.use_h_features else self.obs_dim)
        self.actor_input_dim = (
            self.base_input_dim + self.task_dim
            if self.use_task_value else self.base_input_dim)
        self.actor_hidden = tuple(int(v) for v in actor_hidden)
        self.critic_hidden = tuple(int(v) for v in critic_hidden)

        self.actors = nn.ModuleList([
            Actor(
                obs_dim=self.actor_input_dim,
                action_dim=self.action_dim,
                hidden_dims=list(actor_hidden),
                activation=actor_activation,
                init_gain=np.sqrt(2.0),
                output_gain=0.01,
                feature_norm=self.actor_feature_norm,
                layer_norm=self.actor_layer_norm,
                norm_dim=self.base_input_dim,
                extra_dim=self.task_dim if self.use_task_value else 0,
            )
            for _ in range(self.num_agents)
        ])
        for actor in self.actors:
            actor.action_log_std.data.fill_(float(initial_log_std))

        self.critic = Critic(
            state_dim=self.state_dim,
            hidden_dims=list(critic_hidden),
            activation=critic_activation,
            use_value_norm=self.use_value_norm,
            feature_norm=self.critic_feature_norm,
            layer_norm=self.critic_layer_norm,
        )

    def actor_input(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        grad: torch.Tensor,
        task: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_h_features:
            x = obs
        else:
            hf = h * self.h_feature_scale
            gf = grad * self.grad_feature_scale
            x = torch.cat([obs, hf, gf], dim=-1)
        if self.use_task_value:
            if task is None:
                task = torch.zeros(
                    *x.shape[:-1], self.task_dim,
                    dtype=x.dtype, device=x.device)
            if task.shape[-1] != self.task_dim:
                raise ValueError(
                    f'expected task_dim={self.task_dim}; '
                    f'got {tuple(task.shape)}')
            x = torch.cat([x, task], dim=-1)
        return x

    def get_actions(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        grad: torch.Tensor,
        task: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = []
        log_probs = []
        for i, actor in enumerate(self.actors):
            x = self.actor_input(obs[:, i], h[:, i], grad[:, i], task[:, i]
                                 if task is not None else None)
            action, log_prob = actor.get_action(
                x, deterministic=deterministic,
                bounded=self.bounded_actions)
            actions.append(action)
            if log_prob is None:
                log_prob = torch.zeros(
                    (action.shape[0], 1), dtype=action.dtype, device=action.device)
            log_probs.append(log_prob)
        return torch.stack(actions, dim=1), torch.stack(log_probs, dim=1)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        grad: torch.Tensor,
        actions: torch.Tensor,
        task: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs = []
        entropies = []
        for i, actor in enumerate(self.actors):
            x = self.actor_input(obs[:, i], h[:, i], grad[:, i], task[:, i]
                                 if task is not None else None)
            log_prob, entropy = actor.evaluate_actions(
                x, actions[:, i], bounded=self.bounded_actions)
            log_probs.append(log_prob)
            entropies.append(entropy)
        return torch.stack(log_probs, dim=1), torch.stack(entropies, dim=1)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)

    @torch.no_grad()
    def act_single(
        self,
        obs: np.ndarray,
        h: np.ndarray,
        grad: np.ndarray,
        task: np.ndarray | None = None,
        *,
        deterministic: bool = False,
        device: str | torch.device = 'cpu',
    ) -> tuple[np.ndarray, np.ndarray]:
        dev = torch.device(device)
        obs_t = torch.as_tensor(
            np.asarray(obs, dtype=np.float32)[None], device=dev)
        h_t = torch.as_tensor(
            np.asarray(h, dtype=np.float32)[None], device=dev)
        grad_t = torch.as_tensor(
            np.asarray(grad, dtype=np.float32)[None], device=dev)
        task_t = None
        if self.use_task_value:
            if task is None:
                task = np.zeros(
                    (obs_t.shape[1], self.task_dim), dtype=np.float32)
            task_t = torch.as_tensor(
                np.asarray(task, dtype=np.float32)[None], device=dev)
        actions, log_probs = self.get_actions(
            obs_t, h_t, grad_t, task_t, deterministic=deterministic)
        return (actions[0].cpu().numpy(),
                log_probs[0].cpu().numpy().reshape(-1))

    @torch.no_grad()
    def value_single(
        self,
        state: np.ndarray,
        *,
        device: str | torch.device = 'cpu',
    ) -> float:
        dev = torch.device(device)
        state_t = torch.as_tensor(
            np.asarray(state, dtype=np.float32)[None], device=dev)
        return float(self.value(state_t)[0].item())

    def model_config(self) -> dict[str, Any]:
        return dict(
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_agents=self.num_agents,
            use_h_features=self.use_h_features,
            use_task_value=self.use_task_value,
            task_dim=self.task_dim,
            base_input_dim=self.base_input_dim,
            actor_hidden=list(self.actor_hidden),
            critic_hidden=list(self.critic_hidden),
            actor_activation=self.actor_activation,
            critic_activation=self.critic_activation,
            actor_input_dim=self.actor_input_dim,
            actor_feature_norm=self.actor_feature_norm,
            actor_layer_norm=self.actor_layer_norm,
            critic_feature_norm=self.critic_feature_norm,
            critic_layer_norm=self.critic_layer_norm,
            use_value_norm=self.use_value_norm,
            action_dist=self.action_dist,
            h_feature_scale=self.h_feature_scale,
            grad_feature_scale=self.grad_feature_scale,
        )


@dataclass
class AntRollout:
    obs: np.ndarray
    h: np.ndarray
    grad: np.ndarray
    task: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    costs: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    last_value: float
    completed: list[dict[str, float]]

    def __len__(self) -> int:
        return int(len(self.rewards))

    def summary(self) -> dict[str, float]:
        if not self.completed:
            return dict(
                episodes=0.0, safe_episode=0.0, unsafe_episode=0.0,
                wall_episode=0.0, topple_episode=0.0,
                episode_return=0.0, forward_return=0.0, ctrl_return=0.0,
                survive_return=0.0, episode_length=0.0, survived=0.0,
                cost_rate=0.0, cost_fraction=0.0, wall_fraction=0.0,
                topple_fraction=0.0, h_mean=0.0, h_min=0.0,
                h_neg_frac=0.0, forward_gate_mean=0.0)
        keys = self.completed[0].keys()
        return {
            k: float(np.mean([e[k] for e in self.completed]))
            for k in keys
        }


def concat_ant_rollouts(rollouts: list[AntRollout]) -> AntRollout:
    """Concatenate independent per-environment rollouts for one PPO update."""
    if not rollouts:
        raise ValueError('at least one rollout is required')
    if len(rollouts) == 1:
        return rollouts[0]
    return AntRollout(
        obs=np.concatenate([r.obs for r in rollouts], axis=0),
        h=np.concatenate([r.h for r in rollouts], axis=0),
        grad=np.concatenate([r.grad for r in rollouts], axis=0),
        task=np.concatenate([r.task for r in rollouts], axis=0),
        states=np.concatenate([r.states for r in rollouts], axis=0),
        actions=np.concatenate([r.actions for r in rollouts], axis=0),
        log_probs=np.concatenate([r.log_probs for r in rollouts], axis=0),
        values=np.concatenate([r.values for r in rollouts], axis=0),
        rewards=np.concatenate([r.rewards for r in rollouts], axis=0),
        costs=np.concatenate([r.costs for r in rollouts], axis=0),
        terminated=np.concatenate([r.terminated for r in rollouts], axis=0),
        truncated=np.concatenate([r.truncated for r in rollouts], axis=0),
        last_value=0.0,
        completed=sum((r.completed for r in rollouts), []),
    )


def _to_torch(array: np.ndarray, device: torch.device, dtype=torch.float32):
    return torch.as_tensor(array, dtype=dtype, device=device)


def collect_ant_rollout(
    env,
    model: MappoAntModel,
    *,
    rollout_steps: int,
    seed: int,
    device: str | torch.device = 'cpu',
    cost_penalty: float = 0.0,
    h_barrier_coef: float = 0.0,
    h_barrier_margin: float = 0.0,
    forward_gate_low: float = 0.0,
    forward_gate_high: float = 0.0,
    deterministic: bool = False,
    task_feature_fn=None,
) -> AntRollout:
    dev = torch.device(device)
    model.eval()
    obs, h, grad, _hvec, _hs, _geom, state = env.reset(seed=int(seed))
    ep_id = 0
    ep_return = 0.0
    ep_forward = 0.0
    ep_ctrl = 0.0
    ep_survive = 0.0
    ep_cost = 0.0
    ep_wall = 0.0
    ep_topple = 0.0
    ep_cost_steps = 0
    ep_wall_steps = 0
    ep_topple_steps = 0
    ep_len = 0
    ep_h: list[float] = []
    ep_gate_sum = 0.0
    ep_gate_count = 0
    completed: list[dict[str, float]] = []

    obs_buf, h_buf, grad_buf, state_buf = [], [], [], []
    task_buf = []
    act_buf, logp_buf, val_buf = [], [], []
    rew_buf, cost_buf, term_buf, trunc_buf = [], [], [], []
    last_next_state = state.copy()

    for _ in range(int(rollout_steps)):
        task = None
        if task_feature_fn is not None:
            task = np.asarray(task_feature_fn(obs), dtype=np.float32)
            if task.shape != (env.num_agents, model.task_dim):
                raise ValueError(
                    f'task feature shape {task.shape} != '
                    f'{(env.num_agents, model.task_dim)}')
        else:
            task = np.zeros(
                (env.num_agents, model.task_dim), dtype=np.float32)
        actions, log_probs = model.act_single(
            obs, h, grad, task, deterministic=deterministic, device=dev)
        value = model.value_single(state, device=dev)

        obs_buf.append(obs.copy())
        h_buf.append(h.copy())
        grad_buf.append(grad.copy())
        task_buf.append(task.copy())
        state_buf.append(state.copy())
        act_buf.append(actions.copy())
        logp_buf.append(log_probs.copy())
        val_buf.append(float(value))

        (obs_next, h_next, grad_next, _hvec, _hs, _geom, state_next,
         reward, cost, terminated, truncated, info) = env.step(actions)
        # Soft safety barrier: penalize states whose frozen h is below the
        # configured margin.  This is a pre-specified reward term, not a QP or
        # an action correction.  It is zero for no-h controls.
        h_barrier = 0.0
        if float(h_barrier_coef) > 0.0 and h_next.size:
            h_barrier = float(h_barrier_coef) * max(
                0.0, float(h_barrier_margin) - float(h_next[0, 0]))
        # Safety-gated forward reward.  Only the forward component is
        # modulated; control/contact/survive stay unchanged.  The gate is a
        # smooth task-reward coupling, not an actor input or action filter.
        forward = float(info.get('forward_reward', 0.0))
        forward_gate = 1.0
        if float(forward_gate_high) > float(forward_gate_low) and h_next.size:
            forward_gate = float(np.clip(
                (float(h_next[0, 0]) - float(forward_gate_low))
                / (float(forward_gate_high) - float(forward_gate_low)),
                0.0, 1.0))
        shaped_reward = (
            float(reward) - forward + forward_gate * forward
            - float(cost_penalty) * float(cost) - h_barrier)
        rew_buf.append(float(shaped_reward))
        ep_gate_sum += float(forward_gate)
        ep_gate_count += 1
        cost_buf.append(float(cost))
        term_buf.append(bool(terminated))
        trunc_buf.append(bool(truncated))
        last_next_state = state_next.copy()

        ep_return += float(reward)
        ep_forward += float(info.get('forward_reward', 0.0))
        ep_ctrl += float(info.get('ctrl_reward', 0.0))
        ep_survive += float(info.get('survive_reward', 0.0))
        ep_cost = max(ep_cost, float(cost))
        ep_wall = max(ep_wall, float(info.get('wall', 0.0)))
        ep_topple = max(ep_topple, float(info.get('topple', 0.0)))
        ep_cost_steps += int(float(cost) > 0.0)
        ep_wall_steps += int(float(info.get('wall', 0.0)) > 0.0)
        ep_topple_steps += int(float(info.get('topple', 0.0)) > 0.0)
        ep_len += 1
        ep_h.append(float(info.get('h', 0.0)))

        done = bool(terminated or truncated)
        if done:
            completed.append(dict(
                safe_episode=1.0 if ep_cost <= 0.0 else 0.0,
                unsafe_episode=1.0 if ep_cost > 0.0 else 0.0,
                wall_episode=1.0 if ep_wall > 0.0 else 0.0,
                topple_episode=1.0 if ep_topple > 0.0 else 0.0,
                episode_return=float(ep_return),
                forward_return=float(ep_forward),
                ctrl_return=float(ep_ctrl),
                survive_return=float(ep_survive),
                episode_length=float(ep_len),
                survived=1.0 if truncated else 0.0,
                cost_rate=float(ep_cost),
                cost_fraction=float(ep_cost_steps / max(ep_len, 1)),
                wall_fraction=float(ep_wall_steps / max(ep_len, 1)),
                topple_fraction=float(ep_topple_steps / max(ep_len, 1)),
                h_mean=float(np.mean(ep_h)) if ep_h else 0.0,
                h_min=float(np.min(ep_h)) if ep_h else 0.0,
                h_neg_frac=float(np.mean(np.asarray(ep_h) < 0.0))
                if ep_h else 0.0,
                forward_gate_mean=float(ep_gate_sum / max(ep_gate_count, 1)),
                final_x=float(info.get('x', 0.0)),
            ))
            ep_id += 1
            obs, h, grad, _hvec, _hs, _geom, state = env.reset(
                seed=int(seed) + ep_id)
            ep_return = 0.0
            ep_forward = 0.0
            ep_ctrl = 0.0
            ep_survive = 0.0
            ep_cost = 0.0
            ep_wall = 0.0
            ep_topple = 0.0
            ep_cost_steps = 0
            ep_wall_steps = 0
            ep_topple_steps = 0
            ep_len = 0
            ep_h = []
            ep_gate_sum = 0.0
            ep_gate_count = 0
        else:
            obs, h, grad, state = obs_next, h_next, grad_next, state_next

    # Bootstrap only at the end of the rollout.  A true termination is masked
    # in GAE; a time-limit truncation uses the value of the terminal next state.
    if term_buf and term_buf[-1]:
        bootstrap_value = 0.0
    elif trunc_buf and trunc_buf[-1]:
        bootstrap_value = model.value_single(last_next_state, device=dev)
    else:
        bootstrap_value = model.value_single(last_next_state, device=dev)

    model.train()
    return AntRollout(
        obs=np.asarray(obs_buf, dtype=np.float32),
        h=np.asarray(h_buf, dtype=np.float32),
        grad=np.asarray(grad_buf, dtype=np.float32),
        task=np.asarray(task_buf, dtype=np.float32),
        states=np.asarray(state_buf, dtype=np.float32),
        actions=np.asarray(act_buf, dtype=np.float32),
        log_probs=np.asarray(logp_buf, dtype=np.float32),
        values=np.asarray(val_buf, dtype=np.float32),
        rewards=np.asarray(rew_buf, dtype=np.float32),
        costs=np.asarray(cost_buf, dtype=np.float32),
        terminated=np.asarray(term_buf, dtype=bool),
        truncated=np.asarray(trunc_buf, dtype=bool),
        last_value=float(bootstrap_value),
        completed=completed,
    )


def compute_ant_gae(
    rollout: AntRollout,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    normalize_advantage: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    T = len(rollout)
    values = rollout.values
    rewards = rollout.rewards
    terminated = rollout.terminated.astype(np.float32)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in range(T - 1, -1, -1):
        next_value = (
            float(values[t + 1]) if t + 1 < T
            else float(rollout.last_value))
        nonterminal = 1.0 - float(terminated[t])
        delta = (float(rewards[t])
                 + float(gamma) * next_value * nonterminal
                 - float(values[t]))
        last_gae = (delta
                    + float(gamma) * float(gae_lambda)
                    * nonterminal * last_gae)
        advantages[t] = last_gae
    returns = advantages + values
    raw_mean = float(np.mean(advantages))
    raw_std = float(np.std(advantages))
    if normalize_advantage:
        advantages = (advantages - raw_mean) / (raw_std + 1e-8)
    return (advantages.astype(np.float32),
            returns.astype(np.float32),
            dict(adv_mean=raw_mean, adv_std=raw_std))


def ppo_update_ant(
    model: MappoAntModel,
    optimizer: torch.optim.Optimizer,
    rollout: AntRollout,
    advantages: np.ndarray,
    returns: np.ndarray,
    *,
    device: str | torch.device = 'cpu',
    epochs: int = 4,
    minibatch_size: int = 256,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.001,
    max_grad_norm: float = 0.5,
    actor_optimizer: Optional[torch.optim.Optimizer] = None,
    critic_optimizer: Optional[torch.optim.Optimizer] = None,
    value_huber_delta: float = 0.0,
    update_value_norm: bool = False,
) -> dict[str, float]:
    dev = torch.device(device)
    T = len(rollout)
    obs = _to_torch(rollout.obs, dev)
    h = _to_torch(rollout.h, dev)
    grad = _to_torch(rollout.grad, dev)
    task = _to_torch(rollout.task, dev)
    states = _to_torch(rollout.states, dev)
    actions = _to_torch(rollout.actions, dev)
    old_log_probs = _to_torch(rollout.log_probs, dev)
    old_values = _to_torch(rollout.values, dev)
    adv = _to_torch(advantages, dev)
    ret = _to_torch(returns, dev)

    actor_optimizer = actor_optimizer or optimizer
    critic_optimizer = critic_optimizer or optimizer
    same_optimizer = actor_optimizer is critic_optimizer
    actor_params = list(model.actors.parameters())
    critic_params = list(model.critic.parameters())

    if update_value_norm and model.critic.use_value_norm:
        # Match the upstream MAPPO value-normalization update.  Critic output
        # is denormalized, while the last layer is rescaled to preserve its
        # predictions as the running statistics move.
        model.critic.update_value_norm(ret)

    def _value_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if value_huber_delta and value_huber_delta > 0:
            return F.smooth_l1_loss(
                pred, target, beta=float(value_huber_delta), reduction='mean')
        return (pred - target).square().mean()

    batch_size = max(1, min(int(minibatch_size), T))
    rows: list[dict[str, float]] = []
    model.train()
    for _ in range(int(epochs)):
        perm = torch.randperm(T, device=dev)
        for start in range(0, T, batch_size):
            idx = perm[start:start + batch_size]
            value = model.value(states[idx])
            value_clipped = old_values[idx] + torch.clamp(
                value - old_values[idx], -float(clip_range), float(clip_range))
            value_loss = 0.5 * torch.maximum(
                _value_loss(value, ret[idx]),
                _value_loss(value_clipped, ret[idx]))

            policy_loss = value.new_zeros(())
            entropy = value.new_zeros(())
            for i in range(model.num_agents):
                x = model.actor_input(
                    obs[idx, i], h[idx, i], grad[idx, i], task[idx, i])
                log_prob, ent = model.actors[i].evaluate_actions(
                    x, actions[idx, i], bounded=model.bounded_actions)
                log_prob = log_prob.squeeze(-1)
                ent = ent.squeeze(-1)
                old_log_prob = old_log_probs[idx, i].squeeze(-1)
                ratio = torch.exp(log_prob - old_log_prob)
                clipped = torch.clamp(
                    ratio, 1.0 - float(clip_range), 1.0 + float(clip_range))
                policy_loss = policy_loss + (-torch.min(
                    ratio * adv[idx], clipped * adv[idx]).mean())
                entropy = entropy + ent.mean()
            policy_loss = policy_loss / float(model.num_agents)
            entropy = entropy / float(model.num_agents)
            loss = (policy_loss
                    + float(value_coef) * value_loss
                    - float(entropy_coef) * entropy)
            actor_optimizer.zero_grad(set_to_none=True)
            if not same_optimizer:
                critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if same_optimizer:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()), float(max_grad_norm))
                actor_optimizer.step()
            else:
                # V5/upstream MAPPO uses separate actor and critic learning
                # rates.  Clipping them separately prevents the much larger
                # value loss from suppressing the actor update.
                torch.nn.utils.clip_grad_norm_(
                    actor_params, float(max_grad_norm))
                torch.nn.utils.clip_grad_norm_(
                    critic_params, float(max_grad_norm))
                actor_optimizer.step()
                critic_optimizer.step()
            rows.append(dict(
                loss=float(loss.detach()),
                policy_loss=float(policy_loss.detach()),
                value_loss=float(value_loss.detach()),
                entropy=float(entropy.detach()),
            ))
    if not rows:
        return dict(loss=0.0, policy_loss=0.0, value_loss=0.0, entropy=0.0)
    return {
        k: float(np.mean([r[k] for r in rows]))
        for k in ('loss', 'policy_loss', 'value_loss', 'entropy')
    }


def evaluate_ant_mappo(
    env,
    model: MappoAntModel,
    *,
    episodes: int = 20,
    seed: int = 10000,
    device: str | torch.device = 'cpu',
    deterministic: bool = True,
    task_feature_fn=None,
) -> dict[str, float]:
    dev = torch.device(device)
    model.eval()
    rows: list[dict[str, float]] = []
    for episode in range(int(episodes)):
        obs, h, grad, _hvec, _hs, _geom, state = env.reset(
            seed=int(seed) + episode)
        ep_return = 0.0
        ep_forward = 0.0
        ep_ctrl = 0.0
        ep_survive = 0.0
        ep_cost = 0.0
        ep_wall = 0.0
        ep_topple = 0.0
        ep_cost_steps = 0
        ep_wall_steps = 0
        ep_topple_steps = 0
        ep_len = 0
        h_values: list[float] = []
        x_values: list[float] = []
        action_abs: list[float] = []
        truncated = False
        for _ in range(int(env.max_step)):
            task = None
            if task_feature_fn is not None:
                task = np.asarray(task_feature_fn(obs), dtype=np.float32)
                if task.shape != (env.num_agents, model.task_dim):
                    raise ValueError(
                        f'task feature shape {task.shape} != '
                        f'{(env.num_agents, model.task_dim)}')
            actions, _ = model.act_single(
                obs, h, grad, task, deterministic=bool(deterministic),
                device=dev)
            (obs, h, grad, _hvec, _hs, _geom, state,
             reward, cost, terminated, truncated, info) = env.step(actions)
            ep_return += float(reward)
            ep_forward += float(info.get('forward_reward', 0.0))
            ep_ctrl += float(info.get('ctrl_reward', 0.0))
            ep_survive += float(info.get('survive_reward', 0.0))
            ep_cost = max(ep_cost, float(cost))
            ep_wall = max(ep_wall, float(info.get('wall', 0.0)))
            ep_topple = max(ep_topple, float(info.get('topple', 0.0)))
            ep_cost_steps += int(float(cost) > 0.0)
            ep_wall_steps += int(float(info.get('wall', 0.0)) > 0.0)
            ep_topple_steps += int(float(info.get('topple', 0.0)) > 0.0)
            ep_len += 1
            h_values.append(float(info.get('h', 0.0)))
            x_values.append(float(info.get('x', 0.0)))
            action_abs.append(float(np.mean(np.abs(actions))))
            if terminated or truncated:
                break
        rows.append(dict(
            safe_episode=1.0 if ep_cost <= 0.0 else 0.0,
            unsafe_episode=1.0 if ep_cost > 0.0 else 0.0,
            wall_episode=1.0 if ep_wall > 0.0 else 0.0,
            topple_episode=1.0 if ep_topple > 0.0 else 0.0,
            episode_return=float(ep_return),
            forward_return=float(ep_forward),
            ctrl_return=float(ep_ctrl),
            survive_return=float(ep_survive),
            episode_length=float(ep_len),
            survived=1.0 if truncated else 0.0,
            cost_rate=float(ep_cost),
            cost_fraction=float(ep_cost_steps / max(ep_len, 1)),
            wall_fraction=float(ep_wall_steps / max(ep_len, 1)),
            topple_fraction=float(ep_topple_steps / max(ep_len, 1)),
            h_mean=float(np.mean(h_values)) if h_values else 0.0,
            h_min=float(np.min(h_values)) if h_values else 0.0,
            h_neg_frac=float(np.mean(np.asarray(h_values) < 0.0))
            if h_values else 0.0,
            final_x=float(x_values[-1]) if x_values else 0.0,
            mean_x=float(np.mean(x_values)) if x_values else 0.0,
            action_abs=float(np.mean(action_abs)) if action_abs else 0.0,
        ))
    summary = {
        k: float(np.mean([r[k] for r in rows]))
        for k in rows[0].keys()
    }
    summary['episodes'] = int(episodes)
    summary['eval_seed_start'] = int(seed)
    summary['deterministic'] = 1.0 if deterministic else 0.0
    return summary


def save_ant_mappo_checkpoint(
    path: str | Path,
    model: MappoAntModel,
    *,
    config: dict[str, Any],
    total_steps: int,
    h_checkpoint: Optional[str] = None,
    eval_metrics: Optional[dict[str, float]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(
        schema='v6-ant-mappo.v1',
        model=model.state_dict(),
        model_config=model.model_config(),
        config=config,
        total_steps=int(total_steps),
        h_checkpoint=h_checkpoint,
        eval_metrics=eval_metrics or {},
    ), path)
    return path


def load_ant_mappo_model(
    path: str | Path,
    *,
    device: str | torch.device = 'cpu',
    task_value_expand: bool = False,
    task_dim: int = 1,
) -> tuple[MappoAntModel, dict[str, Any]]:
    """Load an Ant MAPPO checkpoint.

    ``task_value_expand=True`` builds the diagnostic V_T actor (base input +
    ``task_dim``) and expands the first actor layer by zero-initializing the
    new columns.  The input LayerNorm still covers only the original base
    features, so the expanded policy is identical to the checkpoint at
    initialization.
    """
    ckpt = torch.load(Path(path), map_location='cpu', weights_only=False)
    cfg = dict(ckpt.get('config', {}))
    model_cfg = dict(ckpt.get('model_config', {}))
    policy_cfg = cfg.get('policy') or cfg.get('policy_cfg') or {}
    ckpt_use_task_value = bool(model_cfg.get('use_task_value', False))
    use_task_value = True if task_value_expand else ckpt_use_task_value
    resolved_task_dim = (
        int(task_dim) if task_value_expand
        else int(model_cfg.get('task_dim', 1 if ckpt_use_task_value else 0)))
    model = MappoAntModel(
        obs_dim=int(model_cfg['obs_dim']),
        state_dim=int(model_cfg['state_dim']),
        action_dim=int(model_cfg['action_dim']),
        num_agents=int(model_cfg['num_agents']),
        use_h_features=bool(model_cfg.get('use_h_features', True)),
        use_task_value=use_task_value,
        task_dim=resolved_task_dim,
        actor_hidden=tuple(model_cfg.get(
            'actor_hidden', policy_cfg.get('actor_hidden', (64, 64)))),
        critic_hidden=tuple(model_cfg.get(
            'critic_hidden', policy_cfg.get('critic_hidden', (128, 128)))),
        actor_activation=str(model_cfg.get(
            'actor_activation', policy_cfg.get('actor_activation', 'tanh'))),
        critic_activation=str(model_cfg.get(
            'critic_activation', policy_cfg.get('critic_activation', 'tanh'))),
        initial_log_std=float(policy_cfg.get('initial_log_std', -0.5)),
        h_feature_scale=float(policy_cfg.get('h_feature_scale', 1.0)),
        grad_feature_scale=float(
            policy_cfg.get('grad_feature_scale', 1.0)),
        actor_feature_norm=bool(model_cfg.get(
            'actor_feature_norm',
            policy_cfg.get('actor_feature_norm', False))),
        actor_layer_norm=bool(model_cfg.get(
            'actor_layer_norm',
            policy_cfg.get('actor_layer_norm', False))),
        critic_feature_norm=bool(model_cfg.get(
            'critic_feature_norm',
            policy_cfg.get('critic_feature_norm', False))),
        critic_layer_norm=bool(model_cfg.get(
            'critic_layer_norm',
            policy_cfg.get('critic_layer_norm', False))),
        use_value_norm=bool(model_cfg.get(
            'use_value_norm', policy_cfg.get('use_value_norm', False))),
        action_dist=str(model_cfg.get(
            'action_dist', policy_cfg.get('action_dist', 'bounded'))),
    )
    if not task_value_expand:
        model.load_state_dict(ckpt['model'])
    else:
        old_state = ckpt['model']
        new_state = model.state_dict()
        expanded = []
        for key, old_value in old_state.items():
            if key not in new_state:
                continue
            new_value = new_state[key]
            if new_value.shape == old_value.shape:
                new_value.copy_(old_value)
            elif (new_value.dim() == 2
                  and new_value.shape[0] == old_value.shape[0]
                  and new_value.shape[1]
                  == old_value.shape[1] + resolved_task_dim):
                new_value[:, :old_value.shape[1]].copy_(old_value)
                new_value[:, old_value.shape[1]:].zero_()
                expanded.append(key)
            else:
                raise RuntimeError(
                    f'cannot expand checkpoint key {key}: '
                    f'{tuple(old_value.shape)} -> {tuple(new_value.shape)}')
        if not expanded:
            raise RuntimeError(
                'task_value_expand=True but no actor input matrix was '
                'expanded; refusing a silently mismatched warm start')
        model.load_state_dict(new_state)
        print(
            f'expanded {len(expanded)} actor input matrices by '
            f'{resolved_task_dim} zero-initialized task columns',
            flush=True)
    model.to(torch.device(device)).eval()
    return model, ckpt


__all__ = [
    'AntRollout', 'MappoAntModel', 'collect_ant_rollout', 'compute_ant_gae',
    'concat_ant_rollouts',
    'evaluate_ant_mappo', 'load_ant_mappo_model', 'ppo_update_ant',
    'save_ant_mappo_checkpoint',
]
