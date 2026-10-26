"""Alternating co-training for the V6 full-graph h on V6 rollouts.

The policy collector supplies the state distribution; h is updated from the
same actor transitions with safety-only labels and h_dot.  No QP labels,
expert actions, task rewards, or official h checkpoints are used here.  This is
an optional later stage; the main V6 experiment keeps h frozen.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from gcbfplus.utils.graph import GraphsTuple
from .official_h import OfficialLearnedH, save_official_h_checkpoint


GRAPH_KEYS = (
    'nodes', 'states', 'edges', 'receivers', 'senders', 'node_type',
    'n_node', 'n_edge',
)


def graph_to_arrays(graph) -> dict[str, np.ndarray]:
    """Convert one official GraphsTuple to a pickle-friendly record."""
    return {key: np.asarray(getattr(graph, key))
            for key in GRAPH_KEYS}


def _stack_graphs(records: list[dict], prefix: str = '') -> GraphsTuple:
    if not records:
        raise ValueError('empty graph batch')
    n_node = int(records[0][prefix + 'n_node'])
    n_edge = int(records[0][prefix + 'n_edge'])
    for row in records:
        if int(row[prefix + 'n_node']) != n_node:
            raise ValueError('inconsistent n_node in h co-training batch')
        if int(row[prefix + 'n_edge']) != n_edge:
            raise ValueError('inconsistent n_edge in h co-training batch')
    b = len(records)
    return GraphsTuple(
        n_node=jnp.full((b,), n_node, dtype=jnp.int32),
        n_edge=jnp.full((b,), n_edge, dtype=jnp.int32),
        nodes=jnp.asarray(np.stack(
            [row[prefix + 'nodes'] for row in records]), dtype=jnp.float32),
        edges=jnp.asarray(np.stack(
            [row[prefix + 'edges'] for row in records]), dtype=jnp.float32),
        states=jnp.asarray(np.stack(
            [row[prefix + 'states'] for row in records]), dtype=jnp.float32),
        receivers=jnp.asarray(np.stack(
            [row[prefix + 'receivers'] for row in records]), dtype=jnp.int32),
        senders=jnp.asarray(np.stack(
            [row[prefix + 'senders'] for row in records]), dtype=jnp.int32),
        node_type=jnp.asarray(np.stack(
            [row[prefix + 'node_type'] for row in records]), dtype=jnp.int32),
        env_states=None,
        connectivity=None,
    )


def h_safety_loss(
    params,
    net,
    graph: GraphsTuple,
    next_graph: GraphsTuple,
    labels: jnp.ndarray,
    hdot_mask: jnp.ndarray,
    *,
    n_agents: int,
    eps: float,
    unsafe_weight: float,
    hdot_coef: float,
    hdot_alpha: float,
    dt: float,
):
    """Official safety-only h loss on a batch of actor transitions."""
    def one_h(g):
        return net.apply({'params': params}, g, n_agents).squeeze(-1)

    h = jax.vmap(one_h)(graph)
    h_next = jax.vmap(one_h)(next_graph)

    safe = labels > 0
    unsafe = labels < 0
    safe_loss = jnp.where(
        safe, jax.nn.relu(eps - h), 0.0).sum() / \
        jnp.maximum(safe.sum(), 1.0)
    unsafe_loss = jnp.where(
        unsafe, jax.nn.relu(h + eps), 0.0).sum() / \
        jnp.maximum(unsafe.sum(), 1.0)
    # Keep the current h as the CBF target in the finite-difference h_dot;
    # otherwise the 1/dt term can reduce the loss by lowering h_t instead of
    # raising h_next, which collapses the barrier (see the h-mechanism run).
    h_dot = (h_next - jax.lax.stop_gradient(h)) / float(dt)
    hdot_raw = jax.nn.relu(-h_dot - float(hdot_alpha) * h + eps)
    # h_dot trains h only on labeled states; intermediate (0) states are
    # ignored, matching the official GCBF+ labeled_mask.
    active = hdot_mask[:, None] * (labels != 0)
    hdot_loss = (hdot_raw * active).sum() / jnp.maximum(active.sum(), 1.0)

    total = (safe_loss + unsafe_weight * unsafe_loss
             + hdot_coef * hdot_loss)
    info = dict(
        total=total,
        safe=safe_loss,
        unsafe=unsafe_loss,
        hdot=hdot_loss,
        h_mean=jnp.mean(h),
        h_neg_rate=jnp.mean(h < 0.0),
    )
    return total, info


class OfficialHCoTrainer:
    """Small JAX trainer that keeps h in sync with MAPPO/PPO rollouts."""

    def __init__(self, provider: OfficialLearnedH, config: dict, *,
                 seed: int = 0):
        self.provider = provider
        self.net = provider.net
        self.n_agents = int(provider.n_agents)
        self.params = provider.params
        hcfg = dict(config.get('h', {}).get('train', {}))
        self.lr = float(config.get('lr', hcfg.get('lr', 1e-4)))
        self.weight_decay = float(config.get(
            'weight_decay', hcfg.get('weight_decay', 1e-4)))
        self.eps = float(config.get('hdot_eps', hcfg.get('hdot_eps', 0.02)))
        self.unsafe_weight = float(config.get(
            'unsafe_weight', hcfg.get('unsafe_weight', 1.0)))
        self.hdot_coef = float(config.get(
            'hdot_coef', hcfg.get('hdot_coef', 0.2)))
        self.hdot_alpha = float(config.get(
            'hdot_alpha', hcfg.get('hdot_alpha', 1.0)))
        self.dt = float(config.get('dt', hcfg.get('dt', 0.03)))
        self.batch_size = int(config.get('batch_size', 128))
        self.epochs = int(config.get('update_epochs', 2))
        self.replay_size = int(config.get('replay_size', 2048))
        self.rng = np.random.default_rng(int(seed))
        self.buffer: list[dict] = []
        self.opt = optax.adamw(
            learning_rate=self.lr, weight_decay=self.weight_decay)
        self.opt_state = self.opt.init(self.params)
        self._step_fn = jax.jit(self._train_step)

    def _train_step(self, params, opt_state, batch):
        graph, next_graph, labels, mask = batch

        def objective(p):
            return h_safety_loss(
                p, self.net, graph, next_graph, labels, mask,
                n_agents=self.n_agents, eps=self.eps,
                unsafe_weight=self.unsafe_weight,
                hdot_coef=self.hdot_coef,
                hdot_alpha=self.hdot_alpha, dt=self.dt)

        (loss, info), grads = jax.value_and_grad(
            objective, has_aux=True)(params)
        updates, opt_state = self.opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, info

    def add_rollout(self, rollout) -> None:
        """Append the current actor rollout to the h replay buffer."""
        graphs = getattr(rollout, 'h_graph', None)
        next_graphs = getattr(rollout, 'h_next_graph', None)
        labels = getattr(rollout, 'h_labels', None)
        hdot_mask = getattr(rollout, 'h_hdot_mask', None)
        if graphs is None or next_graphs is None:
            return
        for graph, next_graph, label, mask in zip(
                graphs, next_graphs, labels, hdot_mask):
            self.buffer.append(dict(
                graph=graph, next_graph=next_graph,
                labels=np.asarray(label, dtype=np.float32),
                hdot_mask=np.float32(mask),
            ))
        if len(self.buffer) > self.replay_size:
            self.buffer = self.buffer[-self.replay_size:]

    def _sample(self) -> list[dict]:
        n = min(self.batch_size, len(self.buffer))
        if n == 0:
            return []
        idx = self.rng.choice(len(self.buffer), size=n, replace=False)
        return [self.buffer[int(i)] for i in idx]

    def update(self, rollout=None) -> dict[str, float]:
        if rollout is not None:
            self.add_rollout(rollout)
        logs = []
        for _ in range(max(1, self.epochs)):
            rows = self._sample()
            if not rows:
                break
            graph = _stack_graphs([r['graph'] for r in rows])
            next_graph = _stack_graphs([r['next_graph'] for r in rows])
            labels = jnp.asarray(np.stack([r['labels'] for r in rows]))
            mask = jnp.asarray(np.stack([r['hdot_mask'] for r in rows]))
            self.params, self.opt_state, info = self._step_fn(
                self.params, self.opt_state,
                (graph, next_graph, labels, mask))
            logs.append({key: float(value) for key, value in info.items()})
        if not logs:
            return {}
        self.provider.set_params(self.params)
        out = {key: float(np.mean([row[key] for row in logs]))
               for key in logs[0]}
        out['rows'] = len(self.buffer)
        return out

    def save(self, path: str | Path) -> Path:
        return save_official_h_checkpoint(
            path, self.params, n_agents=self.n_agents,
            gnn_layers=self.provider.gnn_layers,
            h_offset=self.provider.h_offset,
            config=dict(co_train=True, lr=self.lr,
                        batch_size=self.batch_size,
                        replay_size=self.replay_size,
                        epochs=self.epochs,
                        hdot_coef=self.hdot_coef),
        )


__all__ = [
    'GRAPH_KEYS', 'graph_to_arrays', 'h_safety_loss',
    'OfficialHCoTrainer',
]
