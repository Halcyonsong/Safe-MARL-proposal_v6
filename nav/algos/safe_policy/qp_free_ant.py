"""Dual-branch QP-free policy for Safe MAMuJoCo Ant-2x4.

The task branch consumes only the task observation.  The fixed safety margin
and its gradient enter a separate safety branch that produces a bounded action
correction.  This avoids letting the safety features distort a warm-started
walking policy while still allowing the correction to depend on h and grad h.
"""

from __future__ import annotations

import torch
from torch import nn


class QpFreeAntPolicy(nn.Module):
    def __init__(
        self,
        obs_dim=31,
        action_dim=4,
        h_dim=4,
        grad_dim=9,
        hidden=128,
        correction_radius=0.2,
        initial_log_std=-1.0,
        gate_margin=0.1,
        gate_tau=0.05,
        gate_on_wall=False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.h_dim = int(h_dim)
        self.grad_dim = int(grad_dim)
        self.correction_radius = float(correction_radius)
        self.gate_margin = float(gate_margin)
        self.gate_tau = float(gate_tau)
        self.gate_on_wall = bool(gate_on_wall)
        # Match the upstream MLP actor branch: LayerNorm + 2x64 + LayerNorm.
        # This lets the task branch be initialized exactly from a learned
        # MAPPO Ant actor while h only affects the correction branch.
        self.task_encoder = nn.Sequential(
            nn.LayerNorm(self.obs_dim),
            nn.Linear(self.obs_dim, 64), nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64), nn.ReLU(),
            nn.LayerNorm(64),
        )
        self.safety_encoder = nn.Sequential(
            nn.Linear(self.obs_dim + self.h_dim + self.grad_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.task_head = nn.Linear(64, self.action_dim)
        self.correction_head = nn.Linear(hidden, self.action_dim)
        self.feasibility_head = nn.Linear(hidden, 1)
        # Return critic belongs to the task branch only.  The safety branch has
        # its own value/cost heads; return gradients must not update safety
        # parameters.
        self.task_value_head = nn.Linear(64, 1)
        # Kept for checkpoint compatibility and as an auxiliary safety value.
        self.value_head = nn.Linear(hidden, 1)
        self.cost_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(initial_log_std)))

    def _std(self):
        return torch.exp(self.log_std).clamp(0.05, 1.0)

    def forward(self, obs, h=None, grad_h=None):
        if h is None:
            h = torch.zeros(*obs.shape[:-1], self.h_dim,
                            device=obs.device, dtype=obs.dtype)
        if grad_h is None:
            grad_h = torch.zeros(*obs.shape[:-1], self.grad_dim,
                                 device=obs.device, dtype=obs.dtype)
        task_features = self.task_encoder(obs)
        task = self.task_head(task_features)
        z = self.safety_encoder(torch.cat([obs, h, grad_h], dim=-1))
        h_scalar = h.min(dim=-1).values
        gate_scalar = h[..., 0] if self.gate_on_wall else h_scalar
        gate = torch.sigmoid(-(gate_scalar - self.gate_margin) / self.gate_tau)
        delta = (gate.unsqueeze(-1) * self.correction_radius
                 * torch.tanh(self.correction_head(z)))
        mean = task + delta
        return dict(
            action=torch.clamp(mean, -1.0, 1.0),
            mean=mean,
            task=task,
            delta=delta,
            gate=gate,
            h_scalar=h_scalar,
            feasibility=torch.sigmoid(self.feasibility_head(z)).squeeze(-1),
            value=self.task_value_head(task_features).squeeze(-1),
            safety_value=self.value_head(z).squeeze(-1),
            cost=self.cost_head(z).squeeze(-1),
        )

    @torch.no_grad()
    def load_base_actor_task(self, actor):
        """Copy the task branch from an upstream MAPPO R_Actor.

        The upstream actor uses ``LayerNorm -> Linear64 -> ReLU -> LayerNorm ->
        Linear64 -> ReLU -> LayerNorm -> Linear(action)``.  Its action mean is
        the deterministic policy output; the environment clips to [-1, 1].
        """
        src = actor.base
        self.task_encoder[0].load_state_dict(src.feature_norm.state_dict())
        self.task_encoder[1].load_state_dict(src.mlp.fc1[0].state_dict())
        self.task_encoder[3].load_state_dict(src.mlp.fc1[2].state_dict())
        self.task_encoder[4].load_state_dict(src.mlp.fc2[0][0].state_dict())
        self.task_encoder[6].load_state_dict(src.mlp.fc2[0][2].state_dict())
        self.task_head.load_state_dict(
            actor.act.action_out.fc_mean.state_dict())
        with torch.no_grad():
            self.correction_head.weight.zero_()
            self.correction_head.bias.zero_()

    def distribution(self, outputs):
        return torch.distributions.Normal(outputs['mean'], self._std())

    def sample(self, obs, h=None, grad_h=None):
        out = self.forward(obs, h, grad_h)
        dist = self.distribution(out)
        raw = dist.rsample()
        return dict(
            out,
            raw_action=raw,
            env_action=torch.clamp(raw, -1.0, 1.0),
            log_prob=dist.log_prob(raw).sum(dim=-1),
            entropy=dist.entropy().sum(dim=-1),
        )

    def evaluate_actions(self, obs, h, grad_h, raw_actions):
        out = self.forward(obs, h, grad_h)
        dist = self.distribution(out)
        return dict(
            out,
            log_prob=dist.log_prob(raw_actions).sum(dim=-1),
            entropy=dist.entropy().sum(dim=-1),
        )


__all__ = ['QpFreeAntPolicy']


class QpFreeAntEnsemble(nn.Module):
    """One dual-branch policy per Ant agent.

    Safe MAMuJoCo Ant-2x4 assigns different joint groups to its two agents.
    A single shared actor is therefore not equivalent to the upstream MAPPO
    actors; this wrapper keeps one task/correction policy per agent while
    preserving the same h and gradient inputs.
    """

    def __init__(self, num_agents, obs_dim=31, action_dim=4, h_dim=4,
                 grad_dim=9, hidden=128, correction_radius=0.2,
                 initial_log_std=-1.0, gate_margin=0.1, gate_tau=0.05,
                 gate_on_wall=False):
        super().__init__()
        self.num_agents = int(num_agents)
        self.models = nn.ModuleList([
            QpFreeAntPolicy(
                obs_dim=obs_dim, action_dim=action_dim, h_dim=h_dim,
                grad_dim=grad_dim, hidden=hidden,
                correction_radius=correction_radius,
                initial_log_std=initial_log_std,
                gate_margin=gate_margin, gate_tau=gate_tau,
                gate_on_wall=gate_on_wall)
            for _ in range(self.num_agents)
        ])

    def forward(self, obs, h=None, grad_h=None):
        outs = []
        for i, model in enumerate(self.models):
            outs.append(model(obs[i:i + 1] if obs.dim() == 2 else obs[..., i, :],
                              h[i:i + 1] if h is not None and h.dim() == 2 else
                              (None if h is None else h[..., i, :]),
                              grad_h[i:i + 1] if grad_h is not None and grad_h.dim() == 2 else
                              (None if grad_h is None else grad_h[..., i, :])))
        keys = outs[0].keys()
        return {k: torch.cat([o[k] for o in outs], dim=0) for k in keys}

    def distribution(self, outputs):
        # Only used through per-agent sample/evaluate_actions below.
        raise NotImplementedError

    def sample(self, obs, h=None, grad_h=None):
        outs = []
        for i, model in enumerate(self.models):
            oi = model.sample(obs[i:i + 1] if obs.dim() == 2 else obs[..., i, :],
                              h[i:i + 1] if h is not None and h.dim() == 2 else
                              (None if h is None else h[..., i, :]),
                              grad_h[i:i + 1] if grad_h is not None and grad_h.dim() == 2 else
                              (None if grad_h is None else grad_h[..., i, :]))
            outs.append(oi)
        keys = outs[0].keys()
        return {k: torch.cat([o[k] for o in outs], dim=0) for k in keys}

    def evaluate_actions(self, obs, h, grad_h, raw_actions):
        outs = []
        for i, model in enumerate(self.models):
            oi = model.evaluate_actions(
                obs[i:i + 1] if obs.dim() == 2 else obs[..., i, :],
                h[i:i + 1] if h.dim() == 2 else h[..., i, :],
                grad_h[i:i + 1] if grad_h.dim() == 2 else grad_h[..., i, :],
                raw_actions[i:i + 1] if raw_actions.dim() == 2 else raw_actions[..., i, :])
            outs.append(oi)
        keys = outs[0].keys()
        return {k: torch.cat([o[k] for o in outs], dim=0) for k in keys}

    @torch.no_grad()
    def load_base_actors_task(self, actors):
        for model, actor in zip(self.models, actors):
            model.load_base_actor_task(actor)

    def train(self, mode=True):
        super().train(mode)
        for model in self.models:
            model.train(mode)
        return self

    def eval(self):
        return self.train(False)


__all__ = ['QpFreeAntPolicy', 'QpFreeAntEnsemble']
