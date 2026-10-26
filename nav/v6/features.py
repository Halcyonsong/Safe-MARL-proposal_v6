"""Differentiable local observation features shared by learned h and V_T.

The V5 policy consumes the compact 92-dimensional local view produced by
``OfficialDoubleIntegrator.observation``.  For gradients we rebuild the same
view from a torch state tensor, holding goals/LiDAR geometry fixed and
differentiating only through agent states.
"""

from __future__ import annotations

import torch


def diagonal_state_grad(outputs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """Return d outputs_i / d states_i for a per-agent scalar output.

    ``is_grads_batched`` computes all agent Jacobian rows in one autograd call;
    this is roughly 2x faster than a Python loop over agents for N=8.
    """
    n = int(states.shape[0])
    eye = torch.eye(n, dtype=outputs.dtype, device=outputs.device)
    grads = torch.autograd.grad(
        outputs, states, grad_outputs=eye, is_grads_batched=True,
        create_graph=False, retain_graph=False)[0]
    idx = torch.arange(n, device=states.device)
    return grads[idx, idx]


def torch_local_obs(
    states: torch.Tensor,
    goals: torch.Tensor,
    lidar: torch.Tensor,
    *,
    max_neighbors: int,
    n_rays: int,
    comm_radius: float,
) -> torch.Tensor:
    """Rebuild the V5 local observation as a differentiable torch tensor.

    Parameters
    ----------
    states : [N,4]
        Agent states ``[x,y,vx,vy]``.  This tensor must be a leaf when a
        gradient w.r.t. agent states is required.
    goals : [N,4]
        Goal states (velocity entries are zero).
    lidar : [N*n_rays,4]
        Official LiDAR hit states.
    """
    if states.ndim != 2 or states.shape[-1] != 4:
        raise ValueError(f'states must be [N,4], got {tuple(states.shape)}')
    n = states.shape[0]
    if goals.shape != (n, 4):
        raise ValueError(f'goals must be [N,4], got {tuple(goals.shape)}')
    if lidar.shape != (n * int(n_rays), 4):
        raise ValueError(
            f'lidar must be [N*n_rays,4], got {tuple(lidar.shape)}')

    pos = states[:, :2]
    rel = states[None, :, :] - states[:, None, :]
    dist = torch.linalg.vector_norm(
        pos[None, :, :] - pos[:, None, :], dim=-1)
    # Topology is held fixed; the selected relative features remain
    # differentiable through both the receiver and neighbour states.
    order = torch.argsort(dist.detach(), dim=1)[:, :int(max_neighbors)]
    neighbor = torch.gather(
        rel, 1, order[..., None].expand(-1, -1, 4))

    lidar4 = lidar.reshape(n, int(n_rays), 4)
    lidar_rel = lidar4[..., :2] - pos[:, None, :]
    hit_dist = torch.linalg.vector_norm(lidar_rel.detach(), dim=-1)
    active = hit_dist < (float(comm_radius) - 1e-1)
    lidar_rel = torch.where(active[..., None], lidar_rel, torch.zeros_like(lidar_rel))

    return torch.cat([
        states,
        goals - states,
        neighbor.reshape(n, -1),
        lidar_rel.reshape(n, -1),
    ], dim=-1)
