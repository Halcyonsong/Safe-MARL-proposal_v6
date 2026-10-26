"""Differentiable graph safety score (an empirical certificate, not a proof)."""

import torch
from torch import nn


class MLP(nn.Sequential):
    def __init__(self, input_dim, hidden_dims, output_dim, beta=10):
        layers = []
        for width in hidden_dims:
            layers.extend([nn.Linear(input_dim, width), nn.Softplus(beta=beta)])
            input_dim = width
        layers.append(nn.Linear(input_dim, output_dim))
        super().__init__(*layers)
        for layer in self:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                nn.init.zeros_(layer.bias)


class CBFNetwork(nn.Module):
    """One output per agent; goal/obstacle nodes never become training targets."""

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        beta = config.get('softplus_beta', 10)
        encoder = config.get('encoder_dims', [128, 256, 256])
        attention = config.get('attention_dims', [128, 128])
        width = attention[-1]
        if config.get('edge_feature_dim', 5) != 5:
            raise ValueError('Current graph schema requires 5 edge features; retrain old checkpoints')
        self.encoder = MLP(11, encoder, width, beta)
        self.attention_gate = MLP(width, attention, 1, beta)
        self.attention_value = MLP(width, attention, width, beta)
        self.output_mlp = MLP(width, config.get('output_dims', [256, 256, 128]), 1, beta)

    def forward(self, node_features, edge_features, edge_index, batch=None):
        source, target = edge_index
        x = torch.cat([node_features[source], node_features[target], edge_features], dim=-1)
        encoded = self.encoder(x)
        logits = self.attention_gate(encoded)
        values = self.attention_value(encoded)
        agent_indices = torch.nonzero(node_features[:, 0] == 1, as_tuple=False).flatten()
        # Segment softmax over each receiving node. This is equivalent to the
        # old Python loop but keeps the large Stage 1 batch on the GPU.
        edge_logits = logits.squeeze(-1)
        num_nodes = node_features.shape[0]
        max_logits = torch.full_like(node_features[:, 0], -torch.inf)
        max_logits.scatter_reduce_(0, target, edge_logits, reduce='amax', include_self=True)
        exp_logits = torch.exp(edge_logits - max_logits[target])
        denom = torch.zeros_like(max_logits)
        denom.scatter_add_(0, target, exp_logits)
        weights = exp_logits / denom[target].clamp_min(torch.finfo(exp_logits.dtype).tiny)
        pooled = torch.zeros((num_nodes, values.shape[-1]), device=values.device, dtype=values.dtype)
        pooled.scatter_add_(0, target.unsqueeze(-1).expand_as(values), values * weights.unsqueeze(-1))
        return self.output_mlp(pooled[agent_indices])

    def compute_gradient(self, h, states, create_graph=None):
        """Diagonal blocks dh_i/ds_i, not the sum of graph-output derivatives.

        These are local policy features. Full multi-agent Lie derivatives also
        require dh_i/ds_j; this implementation makes no hard CBF claim.
        """
        if not states.requires_grad:
            raise ValueError('Set requires_grad before constructing graph features')
        if create_graph is None:
            create_graph = self.training
        gradients = []
        for i in range(states.shape[0]):
            full = torch.autograd.grad(h[i, 0], states, retain_graph=True,
                                       create_graph=create_graph)[0]
            gradients.append(full[i])
        return torch.stack(gradients)

    def forward_with_gradient(self, node_features, edge_features, edge_index, states):
        h = self(node_features, edge_features, edge_index)
        return h, self.compute_gradient(h, states)


def create_graph_data(states, goals, obstacles=None, sensing_radius=3.,
                      agent_radius=.3, area_size=20.):
    """Input states [N,4], goals [N,2], circles [M,3]; edges are j -> i.

    The last edge feature is the source radius. Four wall points per agent
    represent map boundaries. Self edges keep isolated agents differentiable.
    """
    if states.ndim != 2 or states.shape[1] != 4:
        raise ValueError('Graph states must have shape [N,4], not flattened global states')
    goals = torch.as_tensor(goals, device=states.device, dtype=states.dtype)
    if goals.shape != (states.shape[0], 2):
        raise ValueError('goals must have shape [N,2]')
    if obstacles is None:
        obstacles = states.new_empty((0, 3))
    obstacles = torch.as_tensor(obstacles, device=states.device, dtype=states.dtype).reshape(-1, 3)
    n, m = len(states), len(obstacles)
    # Wall points use the current tangential coordinate. Their relative
    # normal coordinate is the signed distance to the boundary.
    walls = []
    for position in states[:, :2]:
        for axis in range(2):
            for side in (-1, 1):
                wall = position.clone()
                wall[axis] = side * area_size / 2
                walls.append(wall)
    positions = torch.cat([states[:, :2], goals, obstacles[:, :2], torch.stack(walls)])
    velocities = torch.cat([states[:, 2:], states.new_zeros((n + m + 4*n, 2))])
    types = states.new_zeros((2*n + m + 4*n, 3))
    types[:n, 0], types[n:2*n, 1], types[2*n:, 2] = 1, 1, 1
    radii = torch.cat([states.new_full((n,), agent_radius), states.new_zeros(n),
                       obstacles[:, 2], states.new_zeros(4*n)])
    edges, features = [], []
    for i in range(n):
        for j in range(len(positions)):
            # Each agent sees its own goal and wall nodes only.
            if n <= j < 2*n and j != n+i:
                continue
            if j >= 2*n+m and (j - 2*n-m) // 4 != i:
                continue
            rel_pos = positions[j] - positions[i]
            if i != j and j != n+i and torch.linalg.vector_norm(rel_pos).detach() - radii[j] > sensing_radius:
                continue
            rel_vel = velocities[j] - velocities[i]
            if i == j:
                # Absolute position/velocity provide boundary context even
                # when all other nodes lie outside the sensing radius.
                rel_pos, rel_vel = states[i, :2], states[i, 2:]
            edges.append((j, i))
            features.append(torch.cat([rel_pos, rel_vel, radii[j:j+1]]))
    return types, torch.stack(features), torch.tensor(edges, device=states.device).T


def create_batched_graph_data(states, goals, obstacles=None, sensing_radius=3.,
                              agent_radius=.3, area_size=20.):
    """Build one disconnected graph for a batch of scenes.

    The graph is flattened into a single PyTorch graph with globally offset
    node indices. This removes the Python scene loop from Stage 1 while
    preserving the exact local-neighborhood semantics of ``create_graph_data``.
    """
    if states.ndim != 3 or states.shape[-1] != 4:
        raise ValueError('Batched graph states must have shape [B,N,4]')
    batch_size, num_agents = states.shape[:2]
    goals = torch.as_tensor(goals, device=states.device, dtype=states.dtype)
    if goals.shape != (batch_size, num_agents, 2):
        raise ValueError('Batched goals must have shape [B,N,2]')
    if obstacles is None:
        obstacles = states.new_empty((batch_size, 0, 3))
    obstacles = torch.as_tensor(obstacles, device=states.device, dtype=states.dtype)
    if obstacles.ndim != 3 or obstacles.shape[0] != batch_size or obstacles.shape[-1] != 3:
        raise ValueError('Batched obstacles must have shape [B,M,3]')
    num_obstacles = obstacles.shape[1]
    device = states.device
    dtype = states.dtype

    # Keep the same node ordering as the single-scene constructor.
    walls = states[:, :, :2].unsqueeze(2).expand(-1, -1, 4, -1).clone()
    wall_positions = torch.stack((
        torch.stack((-torch.ones_like(walls[..., 0]), torch.ones_like(walls[..., 0])), dim=-1),
        torch.stack((-torch.ones_like(walls[..., 0]), torch.ones_like(walls[..., 0])), dim=-1),
    ), dim=2)
    # Explicit construction makes each wall coordinate easy to audit:
    # [x=-R, x=+R, y=-R, y=+R].
    wall_positions = walls.clone()
    wall_positions[:, :, 0, 0] = -area_size / 2
    wall_positions[:, :, 1, 0] = area_size / 2
    wall_positions[:, :, 2, 1] = -area_size / 2
    wall_positions[:, :, 3, 1] = area_size / 2
    wall_positions = wall_positions.reshape(batch_size, 4 * num_agents, 2)

    positions = torch.cat((states[:, :, :2], goals, obstacles[:, :, :2], wall_positions), dim=1)
    velocities = torch.cat((
        states[:, :, 2:],
        torch.zeros((batch_size, num_agents + num_obstacles + 4 * num_agents, 2),
                    device=device, dtype=dtype),
    ), dim=1)
    node_types = torch.zeros((batch_size, 2 * num_agents + num_obstacles + 4 * num_agents, 3),
                             device=device, dtype=dtype)
    node_types[:, :num_agents, 0] = 1
    node_types[:, num_agents:2 * num_agents, 1] = 1
    node_types[:, 2 * num_agents:, 2] = 1
    radii = torch.cat((
        torch.full((batch_size, num_agents), agent_radius, device=device, dtype=dtype),
        torch.zeros((batch_size, num_agents), device=device, dtype=dtype),
        obstacles[:, :, 2],
        torch.zeros((batch_size, 4 * num_agents), device=device, dtype=dtype),
    ), dim=1)

    local_sources, local_targets, local_always = [], [], []
    for receiver in range(num_agents):
        # Agent nodes, the receiver's goal, all obstacles, and its four walls.
        sources = list(range(num_agents))
        sources.append(num_agents + receiver)
        sources.extend(range(2 * num_agents, 2 * num_agents + num_obstacles))
        wall_start = 2 * num_agents + num_obstacles + 4 * receiver
        sources.extend(range(wall_start, wall_start + 4))
        local_sources.extend(sources)
        local_targets.extend([receiver] * len(sources))
        local_always.extend([source == receiver or source == num_agents + receiver
                             for source in sources])
    local_sources = torch.tensor(local_sources, device=device, dtype=torch.long)
    local_targets = torch.tensor(local_targets, device=device, dtype=torch.long)
    local_always = torch.tensor(local_always, device=device, dtype=torch.bool)
    edges_per_scene = local_sources.numel()

    offsets = torch.arange(batch_size, device=device, dtype=torch.long)
    offsets = offsets * positions.shape[1]
    source = (local_sources.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
    target = (local_targets.unsqueeze(0) + offsets.unsqueeze(1)).reshape(-1)
    rel_pos = positions.reshape(-1, 2)[source] - positions.reshape(-1, 2)[target]
    rel_vel = velocities.reshape(-1, 2)[source] - velocities.reshape(-1, 2)[target]
    source_radius = radii.reshape(-1)[source].unsqueeze(1)
    # Match create_graph_data: self edges carry the receiver's absolute
    # position/velocity so an isolated agent remains input-dependent.
    self_edge = source == target
    flat_states = states.reshape(-1, 4)
    nodes_per_scene = positions.shape[1]
    agent_state_index = target[self_edge] % nodes_per_scene
    rel_pos[self_edge] = flat_states[agent_state_index, :2]
    rel_vel[self_edge] = flat_states[agent_state_index, 2:]
    distances = torch.linalg.vector_norm(rel_pos, dim=1) - source_radius[:, 0]
    keep = local_always.unsqueeze(0).expand(batch_size, -1).reshape(-1)
    keep |= distances <= sensing_radius
    edge_features = torch.cat((rel_pos, rel_vel, source_radius), dim=1)
    edge_index = torch.stack((source[keep], target[keep]), dim=0)
    edge_features = edge_features[keep]
    return node_types.reshape(-1, 3), edge_features, edge_index
