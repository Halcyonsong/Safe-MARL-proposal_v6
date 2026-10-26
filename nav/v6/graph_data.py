"""Serialize environment GraphsTuple objects for the V6 safety h.

V6 h is state-only, but its network is a full-graph GNN over the
agent/goal/LiDAR nodes and their edge features.  These helpers keep the
serialized graph schema aligned with the environment graph; no official h
output, QP label, expert action, or task reward is stored here.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

GRAPH_KEYS = (
    'nodes', 'states', 'edges', 'receivers', 'senders', 'node_type',
)


def graph_to_numpy(graph) -> dict[str, np.ndarray]:
    """Convert one official GraphsTuple to fixed-shape NumPy arrays."""
    return dict(
        nodes=np.asarray(graph.nodes, dtype=np.float32),
        states=np.asarray(graph.states, dtype=np.float32),
        edges=np.asarray(graph.edges, dtype=np.float32),
        receivers=np.asarray(graph.receivers, dtype=np.int64),
        senders=np.asarray(graph.senders, dtype=np.int64),
        node_type=np.asarray(graph.node_type, dtype=np.int32),
        n_node=np.int32(np.asarray(graph.n_node).item()),
        n_edge=np.int32(np.asarray(graph.n_edge).item()),
    )


def add_graph_to_row(
    row: dict,
    graph,
    *,
    prefix: str = 'graph_',
) -> dict:
    """Add one graph's arrays to a row with a prefix."""
    payload = graph_to_numpy(graph)
    for key, value in payload.items():
        row[prefix + key] = value
    return row


def stack_graph_rows(
    rows,
    *,
    prefix: str = 'graph_',
) -> dict[str, np.ndarray]:
    """Stack graph arrays from rows.  Every graph must have the same shape."""
    out: dict[str, np.ndarray] = {}
    for key in GRAPH_KEYS:
        out[prefix + key] = np.stack([row[prefix + key] for row in rows])
    out[prefix + 'n_node'] = np.asarray(
        [row[prefix + 'n_node'] for row in rows], dtype=np.int32)
    out[prefix + 'n_edge'] = np.asarray(
        [row[prefix + 'n_edge'] for row in rows], dtype=np.int32)
    return out


def stack_graph_split(
    rows,
    mask,
    *,
    base_prefix: str = 'graph_',
    next_prefix: str = 'next_graph_',
) -> dict[str, np.ndarray]:
    """Stack current and next graph arrays for a split mask."""
    selected = [row for row, keep in zip(rows, mask) if keep]
    out = stack_graph_rows(selected, prefix=base_prefix)
    if selected and (next_prefix + 'states') in selected[0]:
        out.update(stack_graph_rows(selected, prefix=next_prefix))
    return out


def graph_keys(prefix: str = 'graph_') -> tuple[str, ...]:
    """Return the exact keys written by :func:`stack_graph_rows`."""
    return tuple(prefix + key for key in GRAPH_KEYS) + (
        prefix + 'n_node', prefix + 'n_edge')


__all__ = [
    'GRAPH_KEYS', 'add_graph_to_row', 'graph_to_numpy', 'stack_graph_rows',
    'stack_graph_split', 'graph_keys',
]
