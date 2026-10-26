"""Permutation-invariant local graph networks for V6 navigation.

The V5 compact observation is a local star graph:

* one center token: own state + goal-relative state (8 dims);
* ``max_neighbors`` agent tokens: relative position/velocity (4 dims each);
* ``n_rays`` LiDAR point tokens: relative hit position (2 dims each).

This module implements message passing from the neighbor and LiDAR tokens to
the center token.  It is a GNN over the local graph, not an MLP over a flat
concatenation: every token has its own encoder and is aggregated with a
permutation-invariant mean/max.  The same class is used for safety h (tanh
output) and task value V_T (linear output), with independent weights.
"""

from __future__ import annotations

import torch
from torch import nn

from .features import diagonal_state_grad, torch_local_obs


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int | None = None,
                 layers: int = 2):
        super().__init__()
        out_dim = int(hidden if out_dim is None else out_dim)
        mods: list[nn.Module] = []
        width = int(in_dim)
        for _ in range(max(1, int(layers))):
            mods.extend([nn.Linear(width, int(hidden)), nn.SiLU()])
            width = int(hidden)
        mods.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


class _GraphBlock(nn.Module):
    """One message-passing layer from neighbor/LiDAR tokens to the center."""

    def __init__(self, hidden: int):
        super().__init__()
        # Inputs: center embedding, neighbor embedding, relative state+distance.
        self.edge = _MLP(2 * hidden + 5, hidden, hidden)
        # Inputs: center embedding, LiDAR embedding, relative xy+distance.
        self.point = _MLP(2 * hidden + 3, hidden, hidden)
        # Inputs: center, neighbor mean, neighbor max, LiDAR mean, LiDAR max.
        self.update = _MLP(5 * hidden, hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, center, neighbor, neighbor_geom, lidar, lidar_geom,
                lidar_mask):
        n = center.shape[0]
        center_q = center.unsqueeze(1).expand(-1, neighbor.shape[1], -1)
        n_msg = self.edge(torch.cat([center_q, neighbor, neighbor_geom], -1))
        n_mean = n_msg.mean(dim=1)
        n_max = n_msg.max(dim=1).values

        if lidar.shape[1] == 0:
            l_mean = torch.zeros_like(center)
            l_max = torch.zeros_like(center)
        else:
            center_l = center.unsqueeze(1).expand(-1, lidar.shape[1], -1)
            l_msg = self.point(torch.cat([center_l, lidar, lidar_geom], -1))
            mask = lidar_mask.unsqueeze(-1).to(l_msg.dtype)
            denom = mask.sum(dim=1).clamp_min(1.0)
            l_mean = (l_msg * mask).sum(dim=1) / denom
            l_max = l_msg.masked_fill(mask <= 0, -1e9).max(dim=1).values
            l_max = torch.where(
                lidar_mask.any(dim=1, keepdim=True), l_max,
                torch.zeros_like(l_max))

        delta = self.update(torch.cat([center, n_mean, n_max, l_mean, l_max], -1))
        return self.norm(center + delta)


class LocalGraphNet(nn.Module):
    """Local graph network over the 92D V5 navigation observation.

    Parameters
    ----------
    obs_dim:
        Must equal ``8 + 4*max_neighbors + 2*n_rays``.
    hidden:
        Token/message width.
    layers:
        Number of center-update message-passing layers.
    output_activation:
        ``'tanh'`` for safety h in [-1, 1], ``None`` for V_T.
    """

    arch = 'graph'

    def __init__(
        self,
        obs_dim: int = 92,
        hidden: int = 128,
        layers: int = 2,
        max_neighbors: int = 5,
        n_rays: int = 32,
        output_activation: str | None = 'tanh',
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.max_neighbors = int(max_neighbors)
        self.n_rays = int(n_rays)
        self.output_activation = output_activation
        expected = 8 + 4 * self.max_neighbors + 2 * self.n_rays
        if self.obs_dim != expected:
            raise ValueError(
                f'LocalGraphNet expects obs_dim={expected} for '
                f'max_neighbors={self.max_neighbors}, n_rays={self.n_rays}, '
                f'got {self.obs_dim}')

        self.agent_enc = _MLP(8, hidden, hidden)
        # Relative p/v plus distance.
        self.neighbor_enc = _MLP(5, hidden, hidden)
        # Relative xy plus distance.
        self.lidar_enc = _MLP(3, hidden, hidden)
        self.blocks = nn.ModuleList(
            [_GraphBlock(hidden) for _ in range(max(1, self.layers))])
        self.head = nn.Linear(hidden, 1)

    def graph_config(self) -> dict:
        return dict(
            arch='graph', graph_hidden=self.hidden, graph_layers=self.layers,
            max_neighbors=self.max_neighbors, n_rays=self.n_rays,
            output_activation=self.output_activation)

    def _parse(self, obs: torch.Tensor):
        if obs.ndim != 2 or obs.shape[-1] != self.obs_dim:
            raise ValueError(
                f'obs must be [N,{self.obs_dim}], got {tuple(obs.shape)}')
        center_feat = obs[:, :8]
        m = self.max_neighbors
        nbr = obs[:, 8:8 + 4 * m].reshape(-1, m, 4)
        lidar = obs[:, 8 + 4 * m:].reshape(-1, self.n_rays, 2)
        nbr_dist = torch.linalg.vector_norm(nbr[..., :2], dim=-1, keepdim=True)
        lidar_dist = torch.linalg.vector_norm(lidar, dim=-1, keepdim=True)
        # Zero LiDAR slots are the official no-hit sentinel replacement.
        lidar_mask = lidar_dist.squeeze(-1) > 1e-6
        return (center_feat, nbr, torch.cat([nbr, nbr_dist], -1),
                lidar, torch.cat([lidar, lidar_dist], -1), lidar_mask)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        center_feat, nbr, nbr_geom, lidar, lidar_geom, lidar_mask = self._parse(obs)
        center = self.agent_enc(center_feat)
        neighbor = self.neighbor_enc(nbr_geom)
        lidar_emb = self.lidar_enc(lidar_geom)
        for block in self.blocks:
            center = block(
                center, neighbor, nbr_geom, lidar_emb, lidar_geom, lidar_mask)
        out = self.head(center).squeeze(-1)
        if self.output_activation == 'tanh':
            out = torch.tanh(out)
        elif self.output_activation not in (None, 'none', 'linear'):
            raise ValueError(f'unknown output activation: {self.output_activation}')
        return out

    def value_and_local_grad(
        self,
        states: torch.Tensor,
        goals: torch.Tensor,
        lidar: torch.Tensor,
        *,
        max_neighbors: int,
        n_rays: int,
        comm_radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return output [N] and diagonal d output_i / d s_i [N,4]."""
        if not states.requires_grad:
            states = states.detach().clone().requires_grad_(True)
        obs = torch_local_obs(
            states, goals, lidar,
            max_neighbors=max_neighbors, n_rays=n_rays,
            comm_radius=comm_radius)
        out = self.forward(obs)
        return out, diagonal_state_grad(out, states)


__all__ = ['LocalGraphNet']
