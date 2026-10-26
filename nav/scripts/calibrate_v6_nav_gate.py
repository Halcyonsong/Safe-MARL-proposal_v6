"""Calibrate the V6 navigation task gate from a V6-owned h checkpoint.

Stage B pre-registers the gate as::

    low  = 0.0
    high = max(0.02, q10_safe_h_shifted)

where ``h_shifted = h_raw + h_offset`` is the value the actor sees.  The 10th
percentile is taken over horizon-safe validation states only (label > 0).  This
script computes that quantity and reports the resulting gate activation on the
validation graphs.  It is a diagnostic: it does not train, tune, or sweep.

The checkpoint and data are V6-owned.  No official h, QP/u_ref label, expert
action, or task reward is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

# Importing v6.official_h first adds the vendored graph code to sys.path.
from v6.official_h import (  # noqa: E402
    build_official_net, load_official_h_checkpoint)
from gcbfplus.utils.graph import GraphsTuple  # noqa: E402
from v6.provenance import assert_independent_path  # noqa: E402


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _make_graphs(data, idx, prefix: str = 'graph_') -> GraphsTuple:
    """Rebuild the exact padded graph schema used by h training."""
    nodes = np.asarray(data[prefix + 'nodes'][idx], dtype=np.float32)
    states = np.asarray(data[prefix + 'states'][idx], dtype=np.float32)
    edges = np.asarray(data[prefix + 'edges'][idx], dtype=np.float32)
    receivers = np.asarray(data[prefix + 'receivers'][idx], dtype=np.int64)
    senders = np.asarray(data[prefix + 'senders'][idx], dtype=np.int64)
    node_type = np.asarray(data[prefix + 'node_type'][idx], dtype=np.int32)
    b, n = nodes.shape[:2]
    e = edges.shape[1]
    return GraphsTuple(
        n_node=jnp.full(b, n, dtype=jnp.int32),
        n_edge=jnp.full(b, e, dtype=jnp.int32),
        nodes=jnp.asarray(nodes),
        edges=jnp.asarray(edges),
        states=jnp.asarray(states),
        receivers=jnp.asarray(receivers),
        senders=jnp.asarray(senders),
        node_type=jnp.asarray(node_type),
        env_states=None,
        connectivity=None,
    )


def _predict(params, net, data, prefix: str, n_agents: int,
             batch_size: int = 512) -> np.ndarray:
    def single(graph):
        return net.apply({'params': params}, graph, n_agents).squeeze(-1)

    out = []
    n = int(data[prefix + 'nodes'].shape[0])
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(start + batch_size, n))
        out.append(np.asarray(jax.vmap(single)(_make_graphs(data, idx, prefix))))
    return np.concatenate(out) if out else np.zeros((0, n_agents), np.float32)


def _shifted_quantiles(values: np.ndarray, mask: np.ndarray) -> dict:
    x = np.asarray(values)[mask]
    if x.size == 0:
        return dict(count=0, mean=0.0, q05=0.0, q10=0.0, q25=0.0,
                    q50=0.0, q75=0.0, q90=0.0)
    return dict(
        count=int(x.size),
        mean=float(np.mean(x)),
        q05=float(np.quantile(x, 0.05)),
        q10=float(np.quantile(x, 0.10)),
        q25=float(np.quantile(x, 0.25)),
        q50=float(np.quantile(x, 0.50)),
        q75=float(np.quantile(x, 0.75)),
        q90=float(np.quantile(x, 0.90)),
    )


def _gate_stats(h: np.ndarray, labels: np.ndarray, low: float,
                high: float) -> dict:
    if high <= low:
        gate = np.ones_like(h, dtype=np.float32)
    else:
        gate = np.clip((h - low) / (high - low), 0.0, 1.0)
    mask = np.isfinite(labels)

    def one(sel):
        g = gate[sel]
        return dict(
            count=int(g.size),
            mean=float(np.mean(g)) if g.size else 0.0,
            closed_frac=float(np.mean(g < 0.5)) if g.size else 0.0,
            fully_closed_frac=float(np.mean(g <= 0.0)) if g.size else 0.0,
            open_frac=float(np.mean(g >= 1.0)) if g.size else 0.0,
        )

    safe = mask & (labels > 0)
    ignored = mask & (labels == 0)
    unsafe = mask & (labels < 0)
    return dict(all=one(mask), safe=one(safe), ignored=one(ignored),
                unsafe=one(unsafe))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--data',
                    default='outputs/v6/independent/data_stageb/val_data.npz')
    ap.add_argument('--out',
                    default='outputs/v6/independent/h_stageb/'
                            'gate_calibration.json')
    args = ap.parse_args()

    ckpt_path = assert_independent_path(
        _resolve(args.checkpoint), kind='V6 nav h checkpoint')
    data_path = assert_independent_path(
        _resolve(args.data), kind='V6 nav gate calibration data')
    out = assert_independent_path(
        _resolve(args.out), kind='V6 nav gate calibration output')

    ckpt = load_official_h_checkpoint(ckpt_path)
    if ckpt.get('schema') != 'v6-official-cbf-h':
        raise ValueError(f'not a V6 h checkpoint: {ckpt_path}')
    n_agents = int(ckpt.get('n_agents', 8))
    gnn_layers = int(ckpt.get('gnn_layers', 1))
    h_offset = float(ckpt.get('h_offset', 0.0))
    threshold = -h_offset
    net = build_official_net(gnn_layers)

    with np.load(data_path) as data:
        labels = np.asarray(data['graph_labels'], dtype=np.float32)
        h_raw = _predict(ckpt['params'], net, data, 'graph_', n_agents)
        h_next_raw = _predict(
            ckpt['params'], net, data, 'next_graph_', n_agents)
        episode_ids = np.asarray(data['episode_ids'])
        steps = np.asarray(data['steps'])
        label_lookup = {
            (int(ep), int(step)): labels[i]
            for i, (ep, step) in enumerate(zip(episode_ids, steps))
        }
        next_labels = np.full_like(labels, np.nan, dtype=np.float32)
        for i, (ep, step) in enumerate(zip(episode_ids, steps)):
            nxt = label_lookup.get((int(ep), int(step) + 1))
            if nxt is not None:
                next_labels[i] = nxt

    h = h_raw + h_offset
    h_next = h_next_raw + h_offset
    safe_now = labels > 0
    safe_next = next_labels > 0
    q10_safe = float(np.quantile(
        h_next[safe_next], 0.10)) if safe_next.any() else float(
            np.quantile(h[safe_now], 0.10)) if safe_now.any() else 0.0
    low = 0.0
    high = max(0.02, q10_safe)

    result = dict(
        schema='v6.nav_gate_calibration.v1',
        checkpoint=str(ckpt_path),
        data=str(data_path),
        threshold_raw=float(threshold),
        h_offset=float(h_offset),
        h_shift_definition='h_raw + h_offset (the value the actor sees)',
        q10_safe_h=float(q10_safe),
        recommended_low=float(low),
        recommended_high=float(high),
        rule='low=0.0; high=max(0.02, q10_safe_h)',
        current={
            'safe': _shifted_quantiles(h, safe_now),
            'ignored': _shifted_quantiles(h, labels == 0),
            'unsafe': _shifted_quantiles(h, labels < 0),
        },
        next={
            'safe': _shifted_quantiles(h_next, safe_next),
            'ignored': _shifted_quantiles(h_next, next_labels == 0),
            'unsafe': _shifted_quantiles(h_next, next_labels < 0),
        },
        gate_current=_gate_stats(h, labels, low, high),
        gate_next=_gate_stats(h_next, next_labels, low, high),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

    # Keep the raw arrays available only for this process; no extra artifact.
    del h_raw, h_next_raw


if __name__ == '__main__':
    main()
