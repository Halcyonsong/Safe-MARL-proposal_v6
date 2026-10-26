"""Collect safety-only data for the standalone V6 navigation h.

Labels come from the navigation environment's collision/unsafe masks.  The
default behavior is a random policy, so V6 can collect its own data without
V5 weights, V5 data, official h, QP/u_ref labels, or expert actions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from envs.gcbfplus import OfficialDoubleIntegrator  # noqa: E402
from v6.graph_data import graph_to_numpy  # noqa: E402
from v6.provenance import assert_independent_path  # noqa: E402
from v6.safety import (  # noqa: E402
    dense_margin_state_gradients, dense_margin_values,
    forward_margin_sequence, horizon_margin_targets)
from v6.safety_labels import horizon_labels, instant_labels  # noqa: E402


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/v6/v6_nav_independent.yaml')
    ap.add_argument('--out', default='outputs/v6/independent/data')
    ap.add_argument('--episodes', type=int, default=30)
    ap.add_argument('--steps', type=int, default=256)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--test-ratio', type=float, default=0.1)
    ap.add_argument('--policy-checkpoint', action='append', default=None,
                    help='optional V6-owned behavior policy; may be repeated '
                         'to mix behavior sources. V5/V4 checkpoints are '
                         'forbidden. Default: random actions.')
    ap.add_argument('--h-checkpoint',
                    help='optional V6-owned h provider for the behavior policy; '
                         'official h is forbidden')
    ap.add_argument('--policy-device', default='cpu')
    ap.add_argument('--stochastic-actions', action='store_true',
                    help='sample behavior actions instead of using the mean '
                         'action; useful for on-policy h data coverage')
    ap.add_argument('--horizon', type=int, default=32,
                    help='forward-safe window for the main h label and '
                         'future-min margin target')
    ap.add_argument('--margin-scale', type=float, default=0.2,
                    help='tanh scale for the future-min geometric margin '
                         'target; this is a V6 nav safety target, not the '
                         'Ant 0.5 scale')
    ap.add_argument('--action-noise', type=float, default=0.0,
                    help='Gaussian exploration noise on the behavior action')
    args = ap.parse_args()

    out_dir = assert_independent_path(
        _resolve(args.out), kind='V6 h data output')
    for policy_path in (args.policy_checkpoint or []):
        assert_independent_path(
            policy_path, kind='V6 behavior policy')
    if args.h_checkpoint:
        assert_independent_path(
            args.h_checkpoint, kind='V6 behavior h')

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = NAV / cfg_path
    config = yaml.safe_load(cfg_path.read_text())
    env_cfg = config['env']
    env = OfficialDoubleIntegrator(
        num_agents=env_cfg['num_agents'],
        area_size=env_cfg['area_size'],
        max_step=env_cfg['max_step'],
        num_obstacles=env_cfg['num_obstacles'],
        n_rays=env_cfg['n_rays'],
        max_neighbors=env_cfg['max_neighbors'],
        seed=args.seed,
    )
    env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    policies = []
    hnet = None
    if args.policy_checkpoint:
        from v6.official_h import provider_from_checkpoint
        from v6.safety import LocalSafetyProvider
        from v6.training import load_v6_checkpoint
        for policy_path in args.policy_checkpoint:
            policy, _, _, _ = load_v6_checkpoint(
                _resolve(policy_path), device=args.policy_device)
            policy.eval()
            policies.append(policy)
        if args.h_checkpoint:
            h_path = _resolve(args.h_checkpoint)
            if h_path.suffix.lower() in ('.pkl', '.pickle'):
                hnet = provider_from_checkpoint(h_path, env)
            else:
                hnet = LocalSafetyProvider.from_checkpoint(
                    h_path, env, device=args.policy_device,
                    max_neighbors=env_cfg['max_neighbors'],
                    n_rays=env_cfg['n_rays'],
                    comm_radius=env.comm_radius)
    rows = []
    for episode in range(int(args.episodes)):
        env.reset(seed=args.seed + episode)
        for step in range(int(args.steps)):
            graph = env.get_graph()
            # The environment unsafe semantics include collision and the
            # heading-dependent warning zone.  Keep it as the instantaneous
            # negative mask; the horizon-safe label is computed later.
            unsafe = np.asarray(env._env.unsafe_mask(graph), dtype=bool)
            obs = env.observation().astype(np.float32)
            margin = dense_margin_values(
                obs, max_neighbors=int(env_cfg['max_neighbors']),
                n_rays=int(env_cfg['n_rays']),
                area_size=float(env_cfg['area_size']),
                comm_radius=float(env.comm_radius), include_wall=False)
            if not policies:
                action = rng.uniform(
                    -1.0, 1.0, (env.num_agents, 2)).astype(np.float32)
            else:
                policy = policies[episode % len(policies)]
                if hnet is None:
                    h = np.zeros(env.num_agents, dtype=np.float32)
                    grad = np.zeros((env.num_agents, 4), dtype=np.float32)
                else:
                    h, grad = hnet.value_and_local_grad()
                with torch.no_grad():
                    obs_t = torch.tensor(
                        obs, dtype=torch.float32,
                        device=args.policy_device)
                    h_t = torch.tensor(
                        h, dtype=torch.float32,
                        device=args.policy_device).unsqueeze(-1)
                    grad_t = torch.tensor(
                        grad, dtype=torch.float32,
                        device=args.policy_device)
                    extra = torch.zeros(
                        env.num_agents, int(getattr(policy, 'extra_dim', 0)),
                        dtype=torch.float32, device=args.policy_device)
                    if args.stochastic_actions:
                        out = policy.sample(obs_t, h_t, grad_t, extra=extra)
                        action_key = 'env_action'
                    else:
                        out = policy(obs_t, h_t, grad_t, extra=extra)
                        action_key = 'action'
                action = out[action_key].detach().cpu().numpy().astype(
                    np.float32)
            if args.action_noise > 0:
                action = np.clip(
                    action + rng.normal(
                        0.0, float(args.action_noise), action.shape
                    ).astype(np.float32), -1.0, 1.0)
            _, _, terminated, truncated, _ = env.step(action)
            next_obs = env.observation().astype(np.float32)
            next_margin = dense_margin_values(
                next_obs, max_neighbors=int(env_cfg['max_neighbors']),
                n_rays=int(env_cfg['n_rays']),
                area_size=float(env_cfg['area_size']),
                comm_radius=float(env.comm_radius), include_wall=False)
            row = dict(
                obs=obs, next_obs=next_obs, unsafe=unsafe,
                margin=margin, next_margin=next_margin,
                episode=np.int32(episode), step=np.int32(step),
                terminal=np.bool_(terminated or truncated),
                terminal_unsafe=np.zeros(env.num_agents, dtype=bool),
            )
            for key, value in graph_to_numpy(graph).items():
                row['graph_' + key] = value
            for key, value in graph_to_numpy(env.get_graph()).items():
                row['next_graph_' + key] = value
            rows.append(row)
            if terminated or truncated:
                # The state after the final action can contain the terminal
                # unsafe event; it is not a training row, but it must close
                # the forward-safe window of the preceding 32 states.
                rows[-1]['terminal_unsafe'] = np.asarray(
                    env._env.unsafe_mask(env.get_graph()), dtype=bool)
                break
        print(f'episode {episode + 1}/{args.episodes}: rows={len(rows)}', flush=True)

    obs = np.stack([r['obs'] for r in rows])
    next_obs = np.stack([r['next_obs'] for r in rows])
    unsafe = np.stack([r['unsafe'] for r in rows])
    margins = np.stack([r['margin'] for r in rows])
    next_margins = np.stack([r['next_margin'] for r in rows])
    margin_grads = dense_margin_state_gradients(
        obs, max_neighbors=int(env_cfg['max_neighbors']),
        n_rays=int(env_cfg['n_rays']), area_size=float(env_cfg['area_size']),
        comm_radius=float(env.comm_radius), scale=float(args.margin_scale),
        include_wall=False)
    target_mask = np.ones(len(rows), dtype=np.float32)
    episode_ids = np.asarray([r['episode'] for r in rows], dtype=np.int32)
    labels_horizon = np.zeros_like(unsafe, dtype=np.float32)
    labels_instant = np.zeros_like(unsafe, dtype=np.float32)
    safe_horizon = np.zeros_like(unsafe, dtype=bool)
    future_margin = np.zeros_like(margins, dtype=np.float32)
    margin_targets = np.zeros_like(margins, dtype=np.float32)
    hdot_mask = np.zeros(len(rows), dtype=np.float32)
    for ep in np.unique(episode_ids):
        idx = np.where(episode_ids == ep)[0]
        seq = np.concatenate([
            unsafe[idx],
            rows[int(idx[-1])]['terminal_unsafe'][None, :],
        ], axis=0)
        lab, safe = horizon_labels(seq, horizon=int(args.horizon))
        labels_horizon[idx] = lab
        safe_horizon[idx] = safe
        labels_instant[idx] = instant_labels(seq)
        # Future-minimum geometric margin over the same forward horizon.  The
        # terminal next state is appended so a collision on the final action
        # still closes the preceding window.  Official warning-zone labels
        # can be heading-dependent and may not be captured by pure geometry,
        # so states flagged unsafe within the horizon are forced to a
        # negative target.
        margin_seq = forward_margin_sequence(
            margins[idx], next_margins[idx[-1]])
        target_seq, risk_seq = horizon_margin_targets(
            margin_seq, scale=float(args.margin_scale),
            horizon=int(args.horizon))
        tgt = target_seq[:len(idx)].copy()
        risk = risk_seq[:len(idx)]
        tgt[~safe] = np.minimum(tgt[~safe], -0.1)
        margin_targets[idx] = tgt
        future_margin[idx] = risk
        # The final row's next graph is the terminal state.  Keep it out of
        # the h_dot loss; all other actor transitions are valid.
        hdot_mask[idx[:-1]] = 1.0
    labels = labels_horizon
    graph = {}
    next_graph = {}
    for key in ('nodes', 'states', 'edges', 'receivers', 'senders', 'node_type'):
        graph[key] = np.stack([r['graph_' + key] for r in rows])
        next_graph[key] = np.stack([r['next_graph_' + key] for r in rows])
    graph['n_node'] = np.asarray([r['graph_n_node'] for r in rows], dtype=np.int32)
    graph['n_edge'] = np.asarray([r['graph_n_edge'] for r in rows], dtype=np.int32)
    next_graph['n_node'] = np.asarray(
        [r['next_graph_n_node'] for r in rows], dtype=np.int32)
    next_graph['n_edge'] = np.asarray(
        [r['next_graph_n_edge'] for r in rows], dtype=np.int32)
    episode_ids = np.asarray([r['episode'] for r in rows], dtype=np.int32)
    episodes = np.unique(episode_ids)
    rng.shuffle(episodes)
    n_val = max(1, int(len(episodes) * float(args.val_ratio)))
    n_test = max(1, int(len(episodes) * float(args.test_ratio)))
    val_eps = episodes[:n_val]
    test_eps = episodes[n_val:n_val + n_test]
    train_eps = episodes[n_val + n_test:]

    out = out_dir
    out.mkdir(parents=True, exist_ok=True)
    splits = {'train': train_eps, 'val': val_eps, 'test': test_eps}
    summary = dict(
        schema='v6.h_data.v1',
        behavior=('random' if not policies
                  else [str(p) for p in args.policy_checkpoint]),
        behavior_h=(None if args.h_checkpoint is None
                    else str(args.h_checkpoint)),
        stochastic_actions=bool(args.stochastic_actions),
        seed=int(args.seed),
        episodes=int(args.episodes), rows=len(rows),
        horizon=int(args.horizon),
        margin_scale=float(args.margin_scale),
        unsafe_rate=float(np.mean(labels < 0)),
        horizon_safe_rate=float(np.mean(labels > 0)),
        ignored_rate=float(np.mean(labels == 0)),
        instant_unsafe_rate=float(np.mean(labels_instant < 0)),
        target_mean=float(np.mean(margin_targets)),
        target_std=float(np.std(margin_targets)),
        target_negative_rate=float(np.mean(margin_targets < 0)),
        future_margin_mean=float(np.mean(future_margin)),
        future_margin_min=float(np.min(future_margin)),
        future_margin_max=float(np.max(future_margin)),
        hdot_rows=int(np.sum(hdot_mask > 0)),
        splits={})
    for name, eps in splits.items():
        mask = np.isin(episode_ids, eps)
        payload = dict(
            obs=obs[mask], next_obs=next_obs[mask],
            labels=labels[mask],
            graph_labels=labels[mask],
            graph_labels_horizon=labels[mask],
            graph_labels_instant=labels_instant[mask],
            graph_margin_values=margins[mask],
            graph_future_margin=future_margin[mask],
            graph_margin_targets=margin_targets[mask],
            graph_margin_grads=margin_grads[mask],
            graph_target_mask=target_mask[mask],
            graph_hdot_mask=hdot_mask[mask],
            unsafe=unsafe[mask],
            safe_horizon=safe_horizon[mask],
            episode_ids=episode_ids[mask],
            steps=np.asarray([r['step'] for r in rows], dtype=np.int32)[mask],
        )
        for key, value in graph.items():
            payload['graph_' + key] = value[mask]
        for key, value in next_graph.items():
            payload['next_graph_' + key] = value[mask]
        np.savez_compressed(out / f'{name}_data.npz', **payload)
        split_labels = labels[mask]
        split_target = margin_targets[mask]
        summary['splits'][name] = dict(
            episodes=int(len(eps)), rows=int(mask.sum()),
            unsafe_rate=float(np.mean(split_labels < 0)),
            horizon_safe_rate=float(np.mean(split_labels > 0)),
            ignored_rate=float(np.mean(split_labels == 0)),
            target_mean=float(np.mean(split_target)),
            target_negative_rate=float(np.mean(split_target < 0)))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'saved: {out}', flush=True)


if __name__ == '__main__':
    main()
