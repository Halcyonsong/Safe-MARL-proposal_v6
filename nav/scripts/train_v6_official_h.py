"""Train the standalone V6 safety h(s) from V6-collected data.

The network is a V6-owned full-graph safety network (GNN + MLP head + tanh).
Training uses only environment collision/unsafe labels and the h_dot barrier
consistency condition on V6 transitions.  It never loads official h/``cbf.pkl``,
never uses task reward, QP/u_ref labels, expert actions, or V5 data.

The module/file names under ``v6/official_h.py`` are retained for compatibility
with existing loaders, but the produced checkpoint is V6's own h, not the
official pretrained h.  All thresholds/offsets are calibrated on V6 validation
data only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from v6.official_h import (  # noqa: E402
    build_official_net, save_official_h_checkpoint)
from v6.provenance import assert_independent_path  # noqa: E402
from gcbfplus.utils.graph import GraphsTuple  # noqa: E402


GRAPH_KEYS = (
    'nodes', 'states', 'edges', 'receivers', 'senders', 'node_type',
    'n_node', 'n_edge',
)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _make_graphs(data, idx, prefix: str = 'graph_') -> GraphsTuple:
    nodes = np.asarray(data[prefix + 'nodes'][idx], dtype=np.float32)
    states = np.asarray(data[prefix + 'states'][idx], dtype=np.float32)
    edges = np.asarray(data[prefix + 'edges'][idx], dtype=np.float32)
    receivers = np.asarray(data[prefix + 'receivers'][idx], dtype=np.int64)
    senders = np.asarray(data[prefix + 'senders'][idx], dtype=np.int64)
    node_type = np.asarray(data[prefix + 'node_type'][idx], dtype=np.int32)
    n_node = np.asarray(data[prefix + 'n_node'][idx], dtype=np.int32)
    n_edge = np.asarray(data[prefix + 'n_edge'][idx], dtype=np.int32)
    b, n = nodes.shape[:2]
    e = edges.shape[1]
    # Keep a batch axis on every field.  jax.vmap maps the leading axis to
    # the full single-graph GNN, which requires scalar n_node/n_edge.
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



def _make_single_graph(data, row: int, prefix: str = 'graph_') -> GraphsTuple:
    return GraphsTuple(
        n_node=jnp.asarray(data[prefix + 'n_node'][row], dtype=jnp.int32),
        n_edge=jnp.asarray(data[prefix + 'n_edge'][row], dtype=jnp.int32),
        nodes=jnp.asarray(data[prefix + 'nodes'][row], dtype=jnp.float32),
        edges=jnp.asarray(data[prefix + 'edges'][row], dtype=jnp.float32),
        states=jnp.asarray(data[prefix + 'states'][row], dtype=jnp.float32),
        receivers=jnp.asarray(data[prefix + 'receivers'][row], dtype=jnp.int64),
        senders=jnp.asarray(data[prefix + 'senders'][row], dtype=jnp.int64),
        node_type=jnp.asarray(data[prefix + 'node_type'][row], dtype=jnp.int32),
        env_states=None,
        connectivity=None,
    )

def _batch(data, idx, *, need_next: bool, n_agents: int,
           label_key: str = 'graph_labels'):
    graph = _make_graphs(data, idx, 'graph_')
    next_graph = (_make_graphs(data, idx, 'next_graph_')
                  if need_next else None)
    if label_key not in data:
        label_key = 'labels'
    mask_key = 'graph_hdot_mask' if 'graph_hdot_mask' in data else 'hdot_mask'
    labels = jnp.asarray(np.asarray(data[label_key][idx]).reshape(len(idx), n_agents), dtype=jnp.float32)
    mask = jnp.asarray(np.asarray(data[mask_key][idx]).reshape(-1), dtype=jnp.float32)
    return graph, next_graph, labels, mask


def _target_batch(data, idx, n_agents: int, target_key: str | None):
    if not target_key or target_key not in data:
        return (jnp.zeros((len(idx), n_agents), dtype=jnp.float32),
                jnp.zeros((len(idx), n_agents), dtype=jnp.float32))
    target = np.asarray(data[target_key][idx]).reshape(len(idx), n_agents)
    if 'graph_target_mask' in data:
        mask = np.asarray(data['graph_target_mask'][idx]).reshape(
            len(idx), 1)
        mask = np.broadcast_to(mask, target.shape).astype(np.float32)
    else:
        mask = np.ones_like(target, dtype=np.float32)
    return (jnp.asarray(target, dtype=jnp.float32),
            jnp.asarray(mask, dtype=jnp.float32))


def _target_grad_batch(data, idx, n_agents: int):
    if 'graph_margin_grads' not in data:
        return jnp.zeros((len(idx), n_agents, 4), dtype=jnp.float32)
    grad = np.asarray(data['graph_margin_grads'][idx]).reshape(
        len(idx), n_agents, 4)
    return jnp.asarray(grad, dtype=jnp.float32)


def _valid_arrays(h: np.ndarray, labels: np.ndarray):
    h = np.asarray(h).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    valid = labels != 0
    return h[valid], labels[valid]


def _metrics(h: np.ndarray, labels: np.ndarray,
             threshold: float) -> dict:
    raw_h = np.asarray(h).reshape(-1)
    h, labels = _valid_arrays(h, labels)
    y = labels < 0
    pred = h < float(threshold)
    u = float(np.mean(pred[y])) if y.any() else 0.0
    s = float(np.mean(~pred[~y])) if (~y).any() else 0.0
    # AUC via positive/negative rank sum.
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos and n_neg:
        # Unsafe is the positive class: lower h means more unsafe.  Rank by
        # -h so AUROC is > 0.5 for a correctly oriented safety value.
        score = -h
        order = np.argsort(score, kind='mergesort')
        sorted_score = score[order]
        ranks = np.empty(len(score), dtype=np.float64)
        i = 0
        while i < len(score):
            j = i + 1
            while j < len(score) and sorted_score[j] == sorted_score[i]:
                j += 1
            ranks[order[i:j]] = 0.5 * (i + 1 + j)
            i = j
        auroc = (float(ranks[y].sum()) - n_pos * (n_pos + 1) / 2.0) \
            / (n_pos * n_neg)
    else:
        auroc = 0.5
    return dict(
        accuracy=float(np.mean(pred == y)) if len(y) else 0.0,
        balanced_accuracy=0.5 * (u + s),
        unsafe_recall=u,
        safe_recall=s,
        false_alarm=float(np.mean(pred & ~y)) if len(y) else 0.0,
        miss=float(np.mean(~pred & y)) if len(y) else 0.0,
        auroc=float(auroc),
        unsafe_rate=float(np.mean(y)) if len(y) else 0.0,
        h_mean=float(np.mean(h)) if len(h) else 0.0,
        h_std=float(np.std(h)) if len(h) else 0.0,
        h_mean_all=float(np.mean(raw_h)) if len(raw_h) else 0.0,
        h_neg_rate=float(np.mean(pred)) if len(pred) else 0.0,
        h_q05=float(np.quantile(h, 0.05)) if len(h) else 0.0,
        h_q10=float(np.quantile(h, 0.10)) if len(h) else 0.0,
        h_q25=float(np.quantile(h, 0.25)) if len(h) else 0.0,
        h_q50=float(np.quantile(h, 0.50)) if len(h) else 0.0,
        h_q75=float(np.quantile(h, 0.75)) if len(h) else 0.0,
        h_q90=float(np.quantile(h, 0.90)) if len(h) else 0.0,
        h_safe_q05=float(np.quantile(h[~y], 0.05)) if (~y).any() else 0.0,
        h_safe_q10=float(np.quantile(h[~y], 0.10)) if (~y).any() else 0.0,
        h_safe_q25=float(np.quantile(h[~y], 0.25)) if (~y).any() else 0.0,
        h_safe_q50=float(np.quantile(h[~y], 0.50)) if (~y).any() else 0.0,
    )


def _calibrate(h: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    h, labels = _valid_arrays(h, labels)
    if len(h) == 0 or not (labels > 0).any() or not (labels < 0).any():
        return 0.0, 0.0
    y = labels < 0
    candidates = np.unique(np.quantile(h, np.linspace(0.01, 0.99, 99)))
    best_t, best = 0.0, -1.0
    for t in candidates:
        pred = h < float(t)
        u = float(np.mean(pred[y])) if y.any() else 0.0
        s = float(np.mean(~pred[~y])) if (~y).any() else 0.0
        score = 0.5 * (u + s)
        if score > best:
            best, best_t = score, float(t)
    return best_t, best


def _forward(net, params, graph, n_agents: int):
    """Apply the V6 full-graph GNN to every graph in a batch."""
    def single(g):
        return net.apply({'params': params}, g, n_agents).squeeze(-1)
    if getattr(graph.n_node, 'ndim', 0) == 0:
        return single(graph)
    return jax.vmap(single)(graph)


def _graph_with_agent_state(graph, agent_state, n_agents: int,
                            comm_radius: float):
    """Rebuild the graph edge features from a differentiable agent state.

    The serialized graph stores ``edge = state[receiver] - state[sender]``
    with the official communication-radius normalization.  The h input has no
    separate absolute-position feature, so perturbing the agent state must
    also perturb every incident edge.  This mirrors the env's
    ``add_edge_feats`` and keeps the local h gradient consistent with the
    training graph.  Padded/inactive edges are left untouched.
    """
    states = graph.states.at[:n_agents].set(agent_state)
    diff = states[graph.receivers] - states[graph.senders]

    def scale_all(vec):
        norm = jnp.sqrt(1e-6 + jnp.sum(vec ** 2, axis=-1,
                                       keepdims=True))
        coef = jnp.where(
            norm > comm_radius,
            comm_radius / jnp.maximum(norm, comm_radius), 1.0)
        return vec * coef

    # ``get_graph`` stores the current-state graph before the extra
    # ``add_edge_feats`` call used for next-state transitions.  In that
    # convention agent-agent and agent-obstacle edges remain raw.  The
    # vendored goal-edge normalizer indexes ``.at[:, :2]`` on the receiver
    # axis, so only edges into agents 0 and 1 are scaled, and it scales all
    # four features (not just xy).  This is the actual graph the actor's
    # OfficialLearnedH provider rebuilds, so we reproduce it exactly.
    goal_edge = ((graph.node_type[graph.senders] == 1)
                 & (graph.receivers < 2))[:, None]
    diff = jnp.where(goal_edge, scale_all(diff), diff)
    active = (graph.receivers < n_agents)[:, None]
    edges = jnp.where(active, diff, graph.edges)
    return graph._replace(states=states, edges=edges)


def _local_grad_single(net, params, graph, n_agents: int,
                       comm_radius: float):
    """Diagonal ``d h_i / d state_i`` for one graph, shape ``[N, 4]``."""
    agent_state = graph.states[:n_agents]

    def h_of_state(state):
        rebuilt = _graph_with_agent_state(
            graph, state, n_agents, comm_radius)
        return net.apply({'params': params}, rebuilt, n_agents).squeeze(-1)

    jac = jax.jacfwd(h_of_state)(agent_state)
    return jnp.diagonal(jac, axis1=0, axis2=1).T


def _local_grads(net, params, graph, n_agents: int,
                 comm_radius: float):
    """Batched diagonal local state gradients, shape ``[B, N, 4]``."""
    return jax.vmap(
        lambda g: _local_grad_single(
            net, params, g, n_agents, comm_radius))(graph)


def _local_grad_single_fd(net, params, graph, n_agents: int,
                          comm_radius: float, eps: float):
    """Central-difference diagonal ``d h_i / d state_i`` for one graph.

    The analytic ``jax.jacfwd`` version is exact, but differentiating through
    it makes the gradient-alignment loss second-order.  With the masked GNN
    aggregation that second-order reverse pass produced NaN gradients on
    finite losses (observed at batch 16 of the repaired run).  Central
    differences keep the same alignment objective while making the loss
    first-order in the parameters, so ``value_and_grad`` is stable.
    """
    base = graph.states[:n_agents]

    def h_at(state):
        rebuilt = _graph_with_agent_state(
            graph, state, n_agents, comm_radius)
        return net.apply({'params': params}, rebuilt, n_agents).squeeze(-1)

    cols = []
    for dim in range(4):
        delta = jnp.zeros_like(base).at[:, dim].set(eps)
        hp = h_at(base + delta)
        hm = h_at(base - delta)
        cols.append((hp - hm) / (2.0 * eps))
    return jnp.stack(cols, axis=-1)


def _local_grads_fd(net, params, graph, n_agents: int,
                    comm_radius: float, eps: float):
    """Batched central-difference local gradients, shape ``[B, N, 4]``."""
    return jax.vmap(
        lambda g: _local_grad_single_fd(
            net, params, g, n_agents, comm_radius, eps))(graph)


def _target_metrics(h: np.ndarray, target: np.ndarray, labels: np.ndarray,
                    target_mask: np.ndarray | None = None) -> dict:
    """Regression/ranking diagnostics for the continuous safety target."""
    h = np.asarray(h, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    mask = np.isfinite(h) & np.isfinite(target)
    if target_mask is not None:
        mask &= np.asarray(target_mask).reshape(-1).astype(bool)
    out = dict(target_rows=int(mask.sum()))
    if not mask.any():
        return out
    hh, tt = h[mask], target[mask]
    if np.std(hh) > 1e-12 and np.std(tt) > 1e-12:
        corr = float(np.corrcoef(hh, tt)[0, 1])
    else:
        corr = 0.0
    safe = mask & (labels > 0)
    if safe.any() and np.std(h[safe]) > 1e-12 \
            and np.std(target[safe]) > 1e-12:
        safe_corr = float(np.corrcoef(h[safe], target[safe])[0, 1])
    else:
        safe_corr = 0.0
    out.update(
        target_corr=corr,
        target_corr_safe=safe_corr,
        target_mae=float(np.mean(np.abs(hh - tt))),
        target_rmse=float(np.sqrt(np.mean((hh - tt) ** 2))),
        target_mean_pred=float(np.mean(hh)),
        target_mean_true=float(np.mean(tt)),
    )
    return out


def _selection_score(metrics: dict) -> float:
    """Balance future-target quality with safety classification/alignment.

    The previous Stage-B checkpoint was selected on balanced accuracy alone;
    it classified unsafe states well but did not learn the predictive target.
    Target correlation therefore carries the largest weight here, followed by
    safe-row correlation and gradient alignment.
    """
    return (0.50 * float(metrics.get('target_corr', 0.0))
            + 0.20 * float(metrics.get('target_corr_safe', 0.0))
            + 0.20 * float(metrics.get('grad_cosine', 0.0))
            + 0.10 * float(metrics.get('balanced_accuracy', 0.0)))


def _grad_eval_indices(data, label_key: str, n_agents: int,
                       max_rows: int, seed: int) -> np.ndarray:
    """Choose a bounded, risk-enriched subset for gradient diagnostics."""
    labels = np.asarray(data[label_key]).reshape(-1, n_agents)
    n = len(labels)
    if max_rows <= 0 or n <= max_rows:
        return np.arange(n)
    risky = np.where(labels.min(axis=1) <= 0)[0]
    safe = np.where(labels.min(axis=1) > 0)[0]
    rng = np.random.default_rng(int(seed) + 7919)
    n_risky = min(len(risky), max_rows // 2)
    picked = (rng.choice(risky, size=n_risky, replace=False)
              if n_risky else np.empty(0, dtype=np.int64))
    n_safe = max_rows - n_risky
    if n_safe > 0:
        picked = np.concatenate([
            picked,
            rng.choice(safe, size=min(n_safe, len(safe)), replace=False)])
    return np.sort(picked.astype(np.int64))


def _gradient_cosine(params, net, data, idx: np.ndarray, n_agents: int,
                     comm_radius: float, label_key: str) -> dict:
    """Mean cosine between learned and target local state gradients.

    Diagnostics use the exact forward-mode Jacobian.  This is not
    differentiated, so it does not hit the second-order NaN that forced the
    training loss to use central differences.
    """
    graph = _make_graphs(data, idx)
    pred = np.asarray(_local_grads(
        net, params, graph, n_agents, comm_radius), dtype=np.float64)
    target = np.asarray(data['graph_margin_grads'][idx], dtype=np.float64)
    labels = np.asarray(data[label_key][idx]).reshape(len(idx), n_agents)
    pred_norm = np.linalg.norm(pred, axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)
    active = (labels != 0) & (pred_norm > 1e-8) & (target_norm > 1e-8)
    if not active.any():
        return dict(grad_cosine=0.0, grad_rows=0,
                    grad_cosine_safe=0.0, grad_rows_safe=0)
    cos = np.sum(pred * target, axis=-1) / (
        pred_norm * target_norm + 1e-12)
    safe = active & (labels > 0)
    return dict(
        grad_cosine=float(np.mean(cos[active])),
        grad_rows=int(active.sum()),
        grad_cosine_safe=(float(np.mean(cos[safe])) if safe.any() else 0.0),
        grad_rows_safe=int(safe.sum()),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/v6/v6_nav_independent.yaml')
    ap.add_argument('--train-data')
    ap.add_argument('--val-data')
    ap.add_argument('--test-data')
    ap.add_argument('--out', default='outputs/v6/independent/h')
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--batch-size', type=int)
    ap.add_argument('--lr', type=float)
    ap.add_argument('--gnn-layers', type=int)
    ap.add_argument('--unsafe-weight', type=float)
    ap.add_argument('--hdot-coef', type=float)
    ap.add_argument('--hdot-alpha', type=float)
    ap.add_argument('--hdot-eps', type=float)
    ap.add_argument('--safe-margin', type=float,
                    help='h target margin; default is hdot-eps')
    ap.add_argument('--target-key', default=None,
                    help='optional continuous target array, e.g. '
                         'graph_margin_targets from future-min-margin data')
    ap.add_argument('--target-coef', type=float, default=None,
                    help='weight for the continuous target regression loss')
    ap.add_argument('--target-on-all', action=argparse.BooleanOptionalAction,
                    default=None,
                    help='regress the target on ignored rows too; default '
                         'only horizon-safe rows')
    ap.add_argument('--target-loss-delta', type=float, default=None,
                    help='SmoothL1 transition point for the target loss; '
                         '1.0 matches the Ant predictive-h objective')
    ap.add_argument('--grad-coef', type=float, default=None,
                    help='weight for diagonal state-gradient alignment')
    ap.add_argument('--grad-on-all', action=argparse.BooleanOptionalAction,
                    default=None,
                    help='align gradients on unsafe/ignored rows too')
    ap.add_argument('--grad-eval-rows', type=int, default=None,
                    help='validation rows used for the gradient-cosine '
                         'checkpoint diagnostic (0 = all)')
    ap.add_argument('--grad-max-rows', type=int, default=None,
                    help='graphs per training batch used for the gradient '
                         'alignment loss (0 = whole batch)')
    ap.add_argument('--max-grad-norm', type=float, default=None,
                    help='global gradient-norm clip; 1.0 matches the Ant '
                         'predictive-h trainer; <=0 disables clipping')
    ap.add_argument('--grad-fd-eps', type=float, default=None,
                    help='central-difference step for the local state '
                         'gradient alignment (default 1e-3)')
    ap.add_argument('--comm-radius', type=float, default=None,
                    help='communication radius used when rebuilding edges '
                         'for the local state gradient')
    ap.add_argument('--max-rows', type=int)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--label-key', default='graph_labels',
                    help='graph_labels (horizon) or graph_labels_instant')
    ap.add_argument('--hdot-safe-only', action=argparse.BooleanOptionalAction,
                    default=None,
                    help='apply h_dot only to horizon-safe states; useful when '
                         'the behavior policy itself is unsafe')
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = NAV / cfg_path
    config = yaml.safe_load(cfg_path.read_text())
    hcfg = dict(config.get('h', {}).get('train', {}))
    env_cfg = config['env']
    n_agents = int(env_cfg['num_agents'])
    data_dir = _resolve(hcfg.get('data', 'outputs/v6/independent/data'))
    train_path = assert_independent_path(
        _resolve(args.train_data or hcfg.get(
            'train_data', data_dir / 'train_merged.npz')),
        kind='V6 h train data')
    val_path = assert_independent_path(
        _resolve(args.val_data or data_dir / 'val_data.npz'),
        kind='V6 h val data')
    test_path = assert_independent_path(
        _resolve(args.test_data or data_dir / 'test_data.npz'),
        kind='V6 h test data')
    out = assert_independent_path(
        _resolve(args.out or hcfg.get('out', 'outputs/v6/independent/h')),
        kind='V6 h output')
    train = np.load(train_path)
    val = np.load(val_path)
    test = np.load(test_path)
    label_key = args.label_key
    if label_key not in train:
        label_key = 'labels'
    n_graphs = len(train[label_key])
    n_train = n_graphs if args.max_rows is None \
        else min(int(args.max_rows), n_graphs)
    print(json.dumps(dict(train=str(train_path), val=str(val_path),
                          test=str(test_path), rows=n_train)), flush=True)

    seed = int(args.seed if args.seed is not None else config.get('seed', 0))
    rng = np.random.default_rng(seed)
    net = build_official_net(
        int(args.gnn_layers if args.gnn_layers is not None
            else hcfg.get('gnn_layers', 1)))
    # Initialize on one graph with the exact padded environment schema.
    sample_graph = _make_single_graph(train, 0)
    key = jax.random.PRNGKey(seed)
    params = net.init(key, sample_graph, n_agents)['params']
    max_grad_norm = float(
        args.max_grad_norm if args.max_grad_norm is not None
        else hcfg.get('max_grad_norm', 1.0))
    grad_fd_eps = float(
        args.grad_fd_eps if args.grad_fd_eps is not None
        else hcfg.get('grad_fd_eps', 1e-3))
    base_opt = optax.adamw(
        learning_rate=float(args.lr if args.lr is not None
                            else hcfg.get('lr', 1e-4)),
        weight_decay=float(hcfg.get('weight_decay', 1e-4)))
    # The Ant predictive-h trainer clips the global gradient norm to 1.0.
    # Without it the second-order gradient-alignment term can explode at
    # lr=1e-3; the repaired h diverged to NaN in epoch 1 before this was
    # matched.
    opt = (optax.chain(optax.clip_by_global_norm(max_grad_norm), base_opt)
           if max_grad_norm > 0 else base_opt)
    opt_state = opt.init(params)

    unsafe_weight = float(
        args.unsafe_weight if args.unsafe_weight is not None
        else hcfg.get('unsafe_weight', 1.0))
    hdot_coef = float(
        args.hdot_coef if args.hdot_coef is not None
        else hcfg.get('hdot_coef', 0.2))
    hdot_alpha = float(
        args.hdot_alpha if args.hdot_alpha is not None
        else hcfg.get('hdot_alpha', 1.0))
    hdot_eps = float(
        args.hdot_eps if args.hdot_eps is not None
        else hcfg.get('hdot_eps', 0.02))
    safe_margin = float(
        args.safe_margin if args.safe_margin is not None
        else hcfg.get('safe_margin', hdot_eps))
    dt = float(hcfg.get('dt', 0.03))
    batch_size = int(args.batch_size or hcfg.get('batch_size', 128))
    epochs = int(args.epochs or hcfg.get('epochs', 15))
    hdot_safe_only = bool(
        args.hdot_safe_only if args.hdot_safe_only is not None
        else hcfg.get('hdot_safe_only', False))
    target_key = args.target_key or hcfg.get('target_key')
    target_coef = float(
        args.target_coef if args.target_coef is not None
        else hcfg.get('target_coef', 0.0))
    target_on_all = bool(
        args.target_on_all if args.target_on_all is not None
        else hcfg.get('target_on_all', False))
    target_loss_delta = float(
        args.target_loss_delta if args.target_loss_delta is not None
        else hcfg.get('target_loss_delta', 1.0))
    grad_coef = float(
        args.grad_coef if args.grad_coef is not None
        else hcfg.get('grad_coef', 0.0))
    grad_on_all = bool(
        args.grad_on_all if args.grad_on_all is not None
        else hcfg.get('grad_on_all', False))
    grad_eval_rows = int(
        args.grad_eval_rows if args.grad_eval_rows is not None
        else hcfg.get('grad_eval_rows', 512))
    grad_max_rows = int(
        args.grad_max_rows if args.grad_max_rows is not None
        else hcfg.get('grad_max_rows', 16))
    comm_radius = float(
        args.comm_radius if args.comm_radius is not None
        else hcfg.get('comm_radius', 0.5))
    if target_key and target_key not in train:
        raise ValueError(
            f'target_key={target_key!r} is missing from {train_path}')
    if target_key and target_coef <= 0.0:
        raise ValueError('target_key requires target_coef > 0')
    if grad_coef > 0.0 and 'graph_margin_grads' not in train:
        raise ValueError(
            'grad_coef > 0 requires graph_margin_grads in the training data; '
            'run rebuild_v6_h_targets.py first')
    print(json.dumps(dict(
        target_key=target_key, target_coef=target_coef,
        target_on_all=target_on_all, target_loss_delta=target_loss_delta,
        grad_coef=grad_coef, grad_on_all=grad_on_all,
        grad_eval_rows=grad_eval_rows, grad_max_rows=grad_max_rows,
        max_grad_norm=max_grad_norm, grad_fd_eps=grad_fd_eps,
        comm_radius=comm_radius)), flush=True)

    def loss_fn(params, graph, next_graph, labels, mask, target,
                target_mask, target_grad):
        h = _forward(net, params, graph, n_agents)
        safe = labels > 0
        unsafe = labels < 0
        safe_loss = jnp.where(
            safe, jax.nn.relu(safe_margin - h), 0.0).sum() / \
            jnp.maximum(safe.sum(), 1.0)
        unsafe_loss = jnp.where(
            unsafe, jax.nn.relu(h + safe_margin), 0.0).sum() / \
            jnp.maximum(unsafe.sum(), 1.0)
        loss = safe_loss + unsafe_weight * unsafe_loss
        target_loss = jnp.asarray(0.0)
        if target_key and target_coef > 0.0:
            # ``target_on_all`` must include ignored rows (unsafe somewhere in
            # the forward horizon but not now).  Those are exactly the rows
            # where the future-min target carries predictive information; the
            # binary hinge terms intentionally ignore them.
            active = target_mask.astype(bool)
            if not target_on_all:
                active = active & safe
            element = optax.huber_loss(
                h, target, delta=target_loss_delta)
            target_loss = jnp.where(
                active, element, 0.0).sum() / \
                jnp.maximum(active.sum(), 1.0)
            loss = loss + target_coef * target_loss
        else:
            active = None
        grad_loss = jnp.asarray(0.0)
        if grad_coef > 0.0:
            # Align d h_i / d state_i with the analytic geometric safety
            # direction.  This is the navigation analogue of the Ant
            # predictive-h gradient target.  ``grad_on_all`` includes
            # unsafe/ignored rows, which are the ones that carry the
            # predictive warning signal.
            if grad_max_rows > 0:
                k = min(int(grad_max_rows), int(graph.n_node.shape[0]))
                grad_graph = jax.tree_util.tree_map(
                    lambda x: x[:k], graph)
                grad_labels = labels[:k]
                grad_target = target_grad[:k]
            else:
                grad_graph = graph
                grad_labels = labels
                grad_target = target_grad
            pred_grad = _local_grads_fd(
                net, params, grad_graph, n_agents, comm_radius, grad_fd_eps)
            # Avoid ``jnp.linalg.norm`` here: its derivative is 0/0 for an
            # exactly-zero gradient, which can produce a NaN parameter
            # gradient even when that row is masked out of the cosine loss.
            # The epsilon inside the sqrt keeps the denominator differentiable
            # at zero; the mask below excludes those rows from the objective.
            pred_sq = jnp.sum(pred_grad * pred_grad, axis=-1)
            target_sq = jnp.sum(grad_target * grad_target, axis=-1)
            grad_mask = ((grad_labels != 0) if grad_on_all
                         else (grad_labels > 0))
            grad_mask = (grad_mask & (pred_sq > 1e-16)
                         & (target_sq > 1e-16))
            cos = jnp.sum(pred_grad * grad_target, axis=-1) / jnp.sqrt(
                pred_sq * target_sq + 1e-12)
            grad_loss = jnp.where(
                grad_mask, 1.0 - cos, 0.0).sum() / \
                jnp.maximum(grad_mask.sum(), 1.0)
            loss = loss + grad_coef * grad_loss
        if next_graph is not None:
            h_next = _forward(net, params, next_graph, n_agents)
            # The finite-difference h_dot must not be allowed to satisfy the
            # condition by lowering h_t: with dt=0.03, d/dh_t of
            # -(h_{t+1}-h_t)/dt - alpha*h_t is 1/dt-alpha > 0, so the raw
            # expression would push h_t down.  Keep h_t as the CBF target and
            # train only h_next (the -alpha*h_t term still pushes h_t up).
            hdot = (h_next - jax.lax.stop_gradient(h)) / dt
            viol = jax.nn.relu(-hdot - hdot_alpha * h + hdot_eps)
            # The h_dot term is applied only on valid labeled transitions.
            # When the behavior policy is itself unsafe, unsafe transitions
            # can push h upward (to satisfy h_dot) and collapse the barrier;
            # `--hdot-safe-only` restricts this term to horizon-safe states.
            if hdot_safe_only:
                active = mask[:, None] * (labels > 0)
            else:
                active = mask[:, None] * (labels != 0)
            hdot_loss = (viol * active).sum() / jnp.maximum(active.sum(), 1.0)
            loss = loss + hdot_coef * hdot_loss
        else:
            hdot_loss = jnp.asarray(0.0)
        return loss, dict(safe=safe_loss, unsafe=unsafe_loss,
                          target=target_loss, grad=grad_loss,
                          hdot=hdot_loss, total=loss)

    @jax.jit
    def train_step(params, opt_state, graph, next_graph, labels, mask,
                   target, target_mask, target_grad):
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(
                params, graph, next_graph, labels, mask, target,
                target_mask, target_grad)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, metrics

    @jax.jit
    def predict(params, graph):
        return _forward(net, params, graph, n_agents)

    def eval_split(data, eval_params=None, fixed_threshold=None):
        eval_params = params if eval_params is None else eval_params
        out_h, out_y, out_t, out_m = [], [], [], []
        key = label_key if label_key in data else (
            'graph_labels' if 'graph_labels' in data else 'labels')
        for start in range(0, len(data[key]), 512):
            idx = np.arange(start, min(start + 512, len(data[key])))
            graph = _make_graphs(data, idx)
            out_h.append(np.asarray(predict(eval_params, graph)))
            out_y.append(np.asarray(
                data[key][idx]).reshape(len(idx), n_agents))
            if target_key and target_key in data:
                out_t.append(np.asarray(data[target_key][idx]).reshape(
                    len(idx), n_agents))
                if 'graph_target_mask' in data:
                    mask = np.asarray(
                        data['graph_target_mask'][idx]).reshape(len(idx), 1)
                    out_m.append(np.broadcast_to(
                        mask, (len(idx), n_agents)))
        h = np.concatenate(out_h)
        y = np.concatenate(out_y)
        if fixed_threshold is None:
            threshold, _ = _calibrate(h, y)
        else:
            threshold = float(fixed_threshold)
        metrics = _metrics(h, y, threshold)
        if out_t:
            target = np.concatenate(out_t)
            target_mask = (np.concatenate(out_m) if out_m else None)
            metrics.update(_target_metrics(h, target, y, target_mask))
            grad_idx = _grad_eval_indices(data, key, n_agents,
                                          grad_eval_rows, seed)
            if len(grad_idx):
                metrics.update(_gradient_cosine(
                    eval_params, net, data, grad_idx, n_agents,
                    comm_radius, key))
            else:
                metrics.update(dict(grad_cosine=0.0, grad_rows=0,
                                    grad_cosine_safe=0.0,
                                    grad_rows_safe=0))
        return threshold, metrics

    best_score = -float('inf')
    best_params = None
    best_epoch = 0
    history = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n_train)
        rows = []
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            graph, next_graph, labels, mask = _batch(
                train, idx, need_next=bool(hdot_coef > 0),
                n_agents=n_agents, label_key=label_key)
            target, target_mask = _target_batch(
                train, idx, n_agents, target_key)
            target_grad = _target_grad_batch(train, idx, n_agents)
            params, opt_state, metrics = train_step(
                params, opt_state, graph, next_graph, labels, mask, target,
                target_mask, target_grad)
            if not np.isfinite(float(metrics['total'])):
                raise FloatingPointError(
                    f'non-finite training loss at epoch {epoch}, '
                    f'batch start={start}')
            rows.append({k: float(v) for k, v in metrics.items()})
        th, val_metrics = eval_split(val)
        row = dict(epoch=epoch,
                   train={k: float(np.mean([r[k] for r in rows]))
                          for k in rows[0]},
                   val=val_metrics, threshold=float(th),
                   selection_score=_selection_score(val_metrics))
        history.append(row)
        print(json.dumps(row), flush=True)
        score = float(row['selection_score'])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_params = jax.tree_util.tree_map(
                lambda x: np.asarray(x).copy(), params)

    if best_params is None:
        best_params = params
    # Calibrate on validation and report held-out test metrics using the
    # validation-selected best parameters.
    threshold, val_metrics = eval_split(val, best_params)
    _, test_metrics = eval_split(test, best_params, fixed_threshold=threshold)
    out.mkdir(parents=True, exist_ok=True)
    tag = str(args.tag or hcfg.get('tag', 'v6_h'))
    path = save_official_h_checkpoint(
        out / f'{tag}.pkl', best_params,
        n_agents=n_agents, gnn_layers=int(hcfg.get('gnn_layers', 1)),
        h_offset=-float(threshold),
        metrics=dict(val=val_metrics, test=test_metrics),
        config=dict(epochs=epochs, batch_size=batch_size,
                    lr=float(args.lr if args.lr is not None
                             else hcfg.get('lr', 1e-4)),
                    unsafe_weight=unsafe_weight, hdot_coef=hdot_coef,
                    hdot_alpha=hdot_alpha, hdot_eps=hdot_eps,
                    safe_margin=float(safe_margin), dt=dt,
                    label_key=label_key, hdot_safe_only=hdot_safe_only,
                    target_key=target_key, target_coef=target_coef,
                    target_on_all=target_on_all,
                    target_loss_delta=float(target_loss_delta),
                    grad_coef=float(grad_coef), grad_on_all=bool(grad_on_all),
                    grad_eval_rows=int(grad_eval_rows),
                    grad_max_rows=int(grad_max_rows),
                    max_grad_norm=float(max_grad_norm),
                    grad_fd_eps=float(grad_fd_eps),
                    comm_radius=float(comm_radius), train_rows=int(n_train)))
    val_key = label_key if label_key in val else (
        'graph_labels' if 'graph_labels' in val else 'labels')
    test_key = label_key if label_key in test else (
        'graph_labels' if 'graph_labels' in test else 'labels')
    summary = dict(
        schema='v6.safety_h.train.v1', checkpoint=str(path),
        train_data=str(train_path), val_data=str(val_path),
        test_data=str(test_path), seed=seed, tag=tag,
        best_epoch=int(best_epoch), best_selection_score=float(best_score),
        threshold=float(threshold),
        h_offset=float(-threshold), safe_margin=float(safe_margin),
        label_key=label_key, target_key=target_key,
        target_coef=float(target_coef), target_on_all=bool(target_on_all),
        target_loss_delta=float(target_loss_delta),
        grad_coef=float(grad_coef), grad_on_all=bool(grad_on_all),
        val=val_metrics, test=test_metrics,
        history=history, config=dict(
            train_rows=int(n_train), val_rows=int(len(val[val_key])),
            test_rows=int(len(test[test_key])), n_agents=n_agents,
            label_key=label_key, hdot_coef=float(hdot_coef),
            hdot_safe_only=bool(hdot_safe_only),
            target_key=target_key, target_coef=float(target_coef),
            target_on_all=bool(target_on_all),
            target_loss_delta=float(target_loss_delta),
            grad_coef=float(grad_coef), grad_on_all=bool(grad_on_all),
            grad_eval_rows=int(grad_eval_rows),
            grad_max_rows=int(grad_max_rows),
            max_grad_norm=float(max_grad_norm),
            grad_fd_eps=float(grad_fd_eps),
            comm_radius=float(comm_radius)))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(
        checkpoint=str(path), best_epoch=best_epoch,
        threshold=float(threshold), val=val_metrics,
        test=test_metrics), indent=2), flush=True)


if __name__ == '__main__':
    main()
