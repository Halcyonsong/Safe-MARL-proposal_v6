"""Bounded residual actor and centralized critic for on-policy PPO."""

import numpy as np
import torch
from torch import nn
from .base_policy import Critic, init_weights
from .distributions import bounded_sample, bounded_log_prob, bounded_entropy


class CorrectionPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, config):
        super().__init__()
        self.state_dim, self.action_dim = state_dim, action_dim
        self.grad_dim = config.get('grad_dim', 4)
        widths = config.get('encoder_dims', [256, 256, 128])
        input_dim = state_dim+action_dim+1+self.grad_dim
        layers = []
        for width in widths:
            layers += [nn.Linear(input_dim, width), nn.Tanh()]
            input_dim = width
        self.encoder = nn.Sequential(*layers)
        self.delta_u_mean = nn.Linear(input_dim, action_dim)
        self.delta_u_log_std = nn.Parameter(torch.full((action_dim,), float(config.get('initial_log_std', -2))))
        # Bounded action-space residual. Zero init means the correction starts
        # exactly at the frozen base policy; the bound prevents a single update
        # from producing an arbitrarily large deviation.
        self.residual_scale = float(config.get('residual_scale', 0.5))
        self.apply(lambda layer: init_weights(layer, np.sqrt(2)))
        nn.init.zeros_(self.delta_u_mean.weight)
        nn.init.zeros_(self.delta_u_mean.bias)

    def distribution_parameters(self, state, u_ref, h, grad_h):
        raw = self.delta_u_mean(self.encoder(torch.cat([state, u_ref, h, grad_h], -1)))
        residual = self.residual_scale * torch.tanh(raw)
        mean = torch.atanh(u_ref.clamp(-1+1e-6, 1-1e-6)) + residual
        std = self.delta_u_log_std.clamp(-5, 1).exp().expand_as(mean)
        return mean, std

    def forward(self, state, u_ref, h, grad_h):
        mean, std = self.distribution_parameters(state, u_ref, h, grad_h)
        action = mean.tanh()
        return action, std, action-u_ref

    def get_action(self, state, u_ref, h, grad_h, deterministic=False):
        mean, std = self.distribution_parameters(state, u_ref, h, grad_h)
        action, log_prob = bounded_sample(mean, std, deterministic)
        return action, log_prob, action-u_ref

    def evaluate_actions(self, state, u_ref, h, grad_h, u_safe):
        mean, std = self.distribution_parameters(state, u_ref, h, grad_h)
        return bounded_log_prob(mean, std, u_safe), bounded_entropy(mean, std)


class CorrectionActorCritic(nn.Module):
    def __init__(self, obs_dim, value_state_dim, action_dim, config):
        super().__init__()
        self.actor = CorrectionPolicyNetwork(obs_dim, action_dim, config)
        self.critic = Critic(value_state_dim, config.get('critic_hidden_dims', [256, 256]),
                             use_value_norm=config.get('use_value_norm', True))


def compute_correction_reward(state, u_ref, u_safe, next_state, info,
                              h_current, h_next, grad_h=None, dynamics=None,
                              lambda_safety=100., lambda_deviation=.1, lambda_cbf=10.,
                              lambda_task=1., alpha=1., dt=.05, goal=None,
                              task_reward=None, max_acceleration=1.):
    """Per-agent reward. Soft penalties do not provide a safety guarantee."""
    collision = np.asarray(info['collision'], dtype=np.float32)
    deviation = ((np.asarray(u_safe)-np.asarray(u_ref))*max_acceleration)**2
    cbf = (np.asarray(h_next)-np.asarray(h_current))/dt+alpha*np.asarray(h_current)
    # A hinge penalty has no positive reward for idling in a large-h region.
    r_cbf = -np.maximum(-cbf, 0)
    if task_reward is None:
        if goal is None:
            raise ValueError('Explicit goal or task_reward required')
        task_reward = (np.linalg.norm(np.asarray(state)[..., :2]-goal, axis=-1)
                       - np.linalg.norm(np.asarray(next_state)[..., :2]-goal, axis=-1)
                       + 10*np.asarray(info.get('newly_reached', False))-.01)
    parts = dict(safety=-lambda_safety*collision,
                 deviation=-lambda_deviation*deviation.sum(axis=-1),
                 cbf=lambda_cbf*r_cbf, task=lambda_task*np.asarray(task_reward))
    total = sum(parts.values())
    return total.astype(np.float32), parts
