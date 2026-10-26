"""Fixed official GCBF+ h(s) bridge.

The h network is *not* trained here. This module loads the official
DoubleIntegrator cbf.pkl checkpoint and exposes h and dh/ds for V5 policy
features/losses. It deliberately does not import the official QP/actor stack.
"""

from __future__ import annotations

import functools as ft
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn

_VENDOR = Path(__file__).resolve().parents[3] / 'third_party' / 'gcbfplus'
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from gcbfplus.nn.gnn import GNN
from gcbfplus.nn.mlp import MLP
from gcbfplus.nn.utils import default_nn_init


class OfficialCBFNet(nn.Module):
    """Copy of the official GCBF+ CBFNet (no QP/actor dependency)."""

    gnn_cls: type
    head_cls: type

    @nn.compact
    def __call__(self, obs, n_agents, *args, **kwargs):
        x = self.gnn_cls()(obs, node_type=0, n_type=n_agents)
        x = self.head_cls()(x)
        return nn.tanh(nn.Dense(1, kernel_init=default_nn_init())(x))


class OfficialFixedH:
    """Fixed official DoubleIntegrator GCBF+ h(s)."""

    def __init__(self, checkpoint, env, n_agents=8, gnn_layers=1):
        self.env = env
        self.n_agents = int(n_agents)
        with open(checkpoint, 'rb') as stream:
            self.params = pickle.load(stream)['params']
        gnn = ft.partial(
            GNN, msg_dim=128, hid_size_msg=(256, 256),
            hid_size_aggr=(128, 128), hid_size_update=(256, 256),
            out_dim=128, n_layers=int(gnn_layers))
        head = ft.partial(
            MLP, hid_sizes=(256, 256), act=nn.relu, act_final=False,
            name='CBFHead')
        self.net = OfficialCBFNet(gnn_cls=gnn, head_cls=head)
        self._value_fn = jax.jit(self._value_impl)
        self._value_and_grad_fn = jax.jit(self._value_and_grad_impl)
        # Validate the checkpoint once (also warms the JIT cache).
        _ = self._value_fn(env.get_graph())

    def _value_impl(self, graph):
        return self.net.apply(
            {'params': self.params}, graph, self.n_agents).squeeze(-1)

    def _value_and_grad_impl(self, graph):
        agent_node_id = jnp.arange(self.n_agents)
        agent_state = graph.type_states(type_idx=0, n_type=self.n_agents)

        def h_aug(new_agent_state):
            new_state = graph.states.at[agent_node_id].set(new_agent_state)
            new_graph = self.env._env.add_edge_feats(graph, new_state)
            return self.net.apply(
                {'params': self.params}, new_graph, self.n_agents).squeeze(-1)

        h = h_aug(agent_state)
        jac = jax.jacobian(h_aug)(agent_state)  # [N, N, 4]
        local = jnp.diagonal(jac, axis1=0, axis2=1).T  # [N,4]
        return h, local

    def value(self, graph=None):
        graph = self.env.get_graph() if graph is None else graph
        return self._value_fn(graph)

    def value_and_local_grad(self, graph=None):
        """Return h [N] and d h_i / d s_i [N,4] for the current graph."""
        graph = self.env.get_graph() if graph is None else graph
        return self._value_and_grad_fn(graph)
