"""Smoke tests for the V6 from-scratch full-graph safety h provider."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

NAV = Path(__file__).resolve().parents[1]
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from envs.gcbfplus import OfficialDoubleIntegrator  # noqa: E402
from v6.official_h import (  # noqa: E402
    OfficialLearnedH, build_official_net)
from v6.co_train import graph_to_arrays, _stack_graphs  # noqa: E402


def test_official_cbf_random_provider_value_and_grad():
    env = OfficialDoubleIntegrator(
        num_agents=8, area_size=4.0, max_step=16,
        num_obstacles=8, n_rays=32, max_neighbors=5, seed=0)
    env.reset(seed=0)
    net = build_official_net(gnn_layers=1)
    params = net.init(
        jax.random.PRNGKey(0), env.get_graph(), env.num_agents)['params']
    provider = OfficialLearnedH(params, env, n_agents=env.num_agents,
                                gnn_layers=1)
    value = provider.value()
    h, grad = provider.value_and_local_grad()
    assert value.shape == (env.num_agents,)
    assert h.shape == (env.num_agents,)
    assert grad.shape == (env.num_agents, 4)
    assert np.isfinite(value).all()
    assert np.isfinite(h).all()
    assert np.isfinite(grad).all()
    assert (np.abs(value) <= 1.0 + 1e-6).all()
    # The local-gradient path must evaluate the same full graph, including
    # padded/inactive edge rows, not only the active edge features.
    np.testing.assert_allclose(h, value, atol=1e-5)


def test_h_metrics_ignore_intermediate_horizon_states():
    from scripts.train_v6_official_h import _calibrate, _metrics
    h = np.array([0.5, 0.4, -0.2, 0.1], dtype=np.float32)
    labels = np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32)
    metrics = _metrics(h, labels, threshold=0.0)
    # Only the two labeled states enter the classification metrics.
    assert metrics['unsafe_rate'] == 0.5
    assert metrics['balanced_accuracy'] == 1.0
    threshold, score = _calibrate(h, labels)
    assert score == 1.0


def test_co_train_graph_batch_matches_single_graph():
    env = OfficialDoubleIntegrator(
        num_agents=8, area_size=4.0, max_step=16,
        num_obstacles=8, n_rays=32, max_neighbors=5, seed=0)
    env.reset(seed=0)
    net = build_official_net(gnn_layers=1)
    params = net.init(
        jax.random.PRNGKey(0), env.get_graph(), env.num_agents)['params']
    graph = env.get_graph()
    record = graph_to_arrays(graph)
    batch = _stack_graphs([record, record])
    assert batch.n_node.shape == (2,)
    assert batch.nodes.shape[0] == 2
    h_single = net.apply(
        {'params': params}, graph, env.num_agents).squeeze(-1)
    h_batch = jax.vmap(lambda g: net.apply(
        {'params': params}, g, env.num_agents).squeeze(-1))(batch)
    np.testing.assert_allclose(h_batch[0], h_single, atol=1e-5)
    np.testing.assert_allclose(h_batch[1], h_single, atol=1e-5)
