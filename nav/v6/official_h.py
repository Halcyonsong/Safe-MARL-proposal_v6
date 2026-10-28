"""V6-owned full-graph safety h(s) implementation.

The module filename is kept for compatibility with existing V6 loaders, but the
checkpoint produced by ``train_v6_official_h.py`` is **V6's own h**, trained
from V6-collected data.  It never loads official h/``cbf.pkl``, never receives
task reward, QP/u_ref labels, expert actions, or V5 data.

The graph schema and generic GNN/MLP building blocks are reused as design
references.  All parameters, thresholds, and calibration are V6-owned; no
official-h weight, value, gradient, or work point is imported.
"""

from __future__ import annotations

import functools as ft
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

import sys

_VENDOR = Path(__file__).resolve().parents[2] / 'third_party' / 'gcbfplus'
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from gcbfplus.nn.gnn import GNN  # noqa: E402
from gcbfplus.nn.mlp import MLP  # noqa: E402
from gcbfplus.nn.utils import default_nn_init  # noqa: E402


class OfficialCBFNet(nn.Module):
    """V6 full-graph safety h: GNN -> MLP head -> tanh scalar per agent.

    The class name is a historical compatibility label; parameters are trained
    from V6 data and no official h weights are loaded.
    """

    gnn_cls: type
    head_cls: type

    @nn.compact
    def __call__(self, obs, n_agents, *args, **kwargs):
        x = self.gnn_cls()(obs, node_type=0, n_type=n_agents)
        x = self.head_cls()(x)
        return nn.tanh(nn.Dense(1, kernel_init=default_nn_init())(x))


def build_official_net(gnn_layers: int = 1) -> OfficialCBFNet:
    gnn = ft.partial(
        GNN, msg_dim=128, hid_size_msg=(256, 256),
        hid_size_aggr=(128, 128), hid_size_update=(256, 256),
        out_dim=128, n_layers=int(gnn_layers))
    head = ft.partial(
        MLP, hid_sizes=(256, 256), act=nn.relu, act_final=False,
        name='CBFHead')
    return OfficialCBFNet(gnn_cls=gnn, head_cls=head)


class OfficialLearnedH:
    """Environment-facing provider for the V6-learned full-graph h."""

    kind = 'learned_official_cbf'

    def __init__(
        self,
        params,
        env,
        *,
        n_agents: int = 8,
        gnn_layers: int = 1,
        h_offset: float = 0.0,
        h_scale: float = 1.0,
        metadata: dict | None = None,
    ):
        self.env = env
        self.n_agents = int(n_agents)
        self.gnn_layers = int(gnn_layers)
        self.params = jax.tree_util.tree_map(jnp.asarray, params)
        self.net = build_official_net(self.gnn_layers)
        self.h_offset = float(h_offset)
        self.h_scale = float(h_scale)
        self.metadata = dict(metadata or {})
        self._value_fn = jax.jit(self._value_impl)
        self._value_and_grad_fn = jax.jit(self._value_and_grad_impl)
        # Warm the JIT cache and validate graph compatibility.
        _ = self._value_fn(env.get_graph())

    def set_params(self, params):
        """Replace h parameters (used by alternating h/actor co-training)."""
        self.params = jax.tree_util.tree_map(jnp.asarray, params)
        self._value_fn = jax.jit(self._value_impl)
        self._value_and_grad_fn = jax.jit(self._value_and_grad_impl)

    def _apply(self, graph):
        return self.net.apply(
            {'params': self.params}, graph, self.n_agents).squeeze(-1)

    def _value_impl(self, graph):
        return self._apply(graph)

    def _value_and_grad_impl(self, graph):
        agent_node_id = jnp.arange(self.n_agents)
        agent_state = graph.type_states(
            type_idx=0, n_type=self.n_agents)

        env_state = graph.env_states

        def h_aug(new_agent_state):
            # Rebuild the *full* environment graph from the modified
            # state.  add_edge_feats() overwrites the padded/inactive edge
            # rows (their original features are sentinel values), which makes
            # the value and its local gradient inconsistent with h training.
            new_env_state = env_state._replace(agent=new_agent_state)
            new_graph = self.env._env.get_graph(new_env_state)
            return self._apply(new_graph)

        h = h_aug(agent_state)
        jac = jax.jacobian(h_aug)(agent_state)  # [N,N,4]
        local = jnp.diagonal(jac, axis1=0, axis2=1).T
        return h, local

    def value(self, graph=None):
        graph = self.env.get_graph() if graph is None else graph
        h = np.asarray(self._value_fn(graph), dtype=np.float32)
        return self.h_scale * h + self.h_offset

    def value_and_local_grad(self, graph=None):
        graph = self.env.get_graph() if graph is None else graph
        h, grad = self._value_and_grad_fn(graph)
        return (self.h_scale * np.asarray(h, dtype=np.float32) + self.h_offset,
                self.h_scale * np.asarray(grad, dtype=np.float32))


def save_official_h_checkpoint(
    path: str | Path,
    params,
    *,
    n_agents: int = 8,
    gnn_layers: int = 1,
    h_offset: float = 0.0,
    metrics: dict | None = None,
    config: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        schema='v6-official-cbf-h',
        params=jax.tree_util.tree_map(np.asarray, params),
        n_agents=int(n_agents),
        gnn_layers=int(gnn_layers),
        h_offset=float(h_offset),
        metrics=dict(metrics or {}),
        config=dict(config or {}),
    )
    with path.open('wb') as stream:
        pickle.dump(payload, stream)
    return path


def load_official_h_checkpoint(path: str | Path) -> dict:
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def provider_from_checkpoint(
    path: str | Path,
    env,
    *,
    h_offset: float | None = None,
    h_scale: float | None = None,
    n_agents: int | None = None,
) -> OfficialLearnedH:
    ckpt = load_official_h_checkpoint(path)
    if ckpt.get('schema') != 'v6-official-cbf-h':
        raise ValueError(f'not a V6 safety h checkpoint: {path}')
    offset = float(ckpt.get('h_offset', 0.0))
    if h_offset is not None:
        offset = float(h_offset)
    scale = float(ckpt.get('h_scale', 1.0))
    if h_scale is not None:
        scale = float(h_scale)
    # ``n_agents`` is an evaluation-time graph-size override for the density
    # stress test.  The graph net is local/permutation-invariant, so the same
    # weights can be applied to a larger agent set; the checkpoint's metadata
    # remains the training-time size.  Default behaviour is unchanged.
    graph_agents = int(
        ckpt.get('n_agents', env.num_agents)
        if n_agents is None else n_agents)
    return OfficialLearnedH(
        ckpt['params'], env,
        n_agents=graph_agents,
        gnn_layers=int(ckpt.get('gnn_layers', 1)),
        h_offset=offset * scale,
        h_scale=scale,
        metadata=ckpt)


__all__ = [
    'OfficialCBFNet', 'OfficialLearnedH', 'build_official_net',
    'save_official_h_checkpoint', 'load_official_h_checkpoint',
    'provider_from_checkpoint',
]
