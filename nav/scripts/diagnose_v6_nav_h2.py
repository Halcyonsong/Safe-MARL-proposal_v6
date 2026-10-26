#!/usr/bin/env python3
"""Diagnose whether a second navigation h/actor iteration is justified.

This is a read-only evaluation of an existing promoted actor (actor1) and its
frozen V6 h (h1).  It does not train, gate-sweep, or load official h.

For every agent-step it records:

- h1 and the hypothetical predictive gate at the current/next state;
- official collision/unsafe/reach events;
- the geometric future-min margin target from the actor1 state distribution;
- h1 behavior in the steps leading up to the first collision of each agent.

The output is intended to answer one question: are collisions occurring in
states that h1 calls safe, or in states h1 already flags as unsafe?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from envs.gcbfplus import OfficialDoubleIntegrator  # noqa: E402
from v6.official_h import provider_from_checkpoint as official_h_provider  # noqa: E402
from v6.provenance import assert_independent_path  # noqa: E402
from v6.safety import (  # noqa: E402
    LocalSafetyProvider,
    dense_margin_values,
    horizon_min_margin,
)
from v6.safety_labels import horizon_labels  # noqa: E402
from v6.training import load_v6_checkpoint  # noqa: E402


def _cwd_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def _device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _quantiles(values: np.ndarray, prefix: str = '') -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'count': 0}
    out = {
        'count': int(values.size),
        'mean': float(values.mean()),
        'std': float(values.std()),
        'min': float(values.min()),
        'max': float(values.max()),
    }
    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        out[f'{prefix}q{q:02d}'] = float(np.percentile(values, q))
    return out


def _auc(score: np.ndarray, positive: np.ndarray) -> float | None:
    """Rank AUROC with average ranks; higher score means more positive."""
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    n1 = int(positive.sum())
    n0 = int((~positive).sum())
    if n1 == 0 or n0 == 0:
        return None
    ranks = rankdata(score, method='average')
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _binary_metrics(h: np.ndarray, unsafe: np.ndarray,
                    threshold: float = 0.0) -> dict:
    h = np.asarray(h, dtype=np.float64)
    unsafe = np.asarray(unsafe, dtype=bool)
    pred_unsafe = h <= threshold
    tp = int((pred_unsafe & unsafe).sum())
    fn = int(((~pred_unsafe) & unsafe).sum())
    fp = int((pred_unsafe & (~unsafe)).sum())
    tn = int(((~pred_unsafe) & (~unsafe)).sum())
    unsafe_recall = tp / (tp + fn) if tp + fn else None
    safe_recall = tn / (tn + fp) if tn + fp else None
    false_alarm = fp / (fp + tn) if fp + tn else None
    ba = (None if unsafe_recall is None or safe_recall is None
          else 0.5 * (unsafe_recall + safe_recall))
    return dict(threshold=float(threshold), tp=tp, fn=fn, fp=fp, tn=tn,
                unsafe_recall=unsafe_recall, safe_recall=safe_recall,
                false_alarm=false_alarm, balanced_accuracy=ba)


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _gate(h_next: np.ndarray, low: float, high: float) -> np.ndarray:
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.ones_like(h_next, dtype=np.float32)
    return np.clip((h_next - low) / (high - low), 0.0, 1.0).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True,
                    help='actor1 policy_final.pth')
    ap.add_argument('--h-checkpoint', required=True,
                    help='frozen V6 h1 checkpoint')
    ap.add_argument('--num-episodes', type=int, default=100)
    ap.add_argument('--seed', type=int, default=10000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--reference-eval',
                    help='optional existing eval_100ep_final.json to verify')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    device = _device(args.device)
    ckpt_path = assert_independent_path(
        _cwd_path(args.checkpoint), kind='V6 policy checkpoint')
    model, _task_net, _ema, ckpt = load_v6_checkpoint(
        ckpt_path, device=device)
    config = ckpt['config']
    env_cfg = config['env']
    env = OfficialDoubleIntegrator(
        num_agents=env_cfg['num_agents'],
        area_size=env_cfg['area_size'],
        max_step=env_cfg['max_step'],
        num_obstacles=env_cfg['num_obstacles'],
        n_rays=env_cfg['n_rays'],
        max_neighbors=env_cfg['max_neighbors'],
        seed=config['seed'],
    )
    env.reset(seed=int(config['seed']))
    h_path = assert_independent_path(
        _cwd_path(args.h_checkpoint), kind='V6 audit h')
    h_offset = config['h'].get('h_offset')
    if h_path.suffix.lower() in ('.pkl', '.pickle'):
        hnet = official_h_provider(
            h_path, env,
            h_offset=None if h_offset is None else float(h_offset),
            h_scale=config['h'].get('h_scale'))
    else:
        hnet = LocalSafetyProvider.from_checkpoint(
            h_path, env, device=device,
            max_neighbors=env_cfg['max_neighbors'],
            n_rays=env_cfg['n_rays'],
            comm_radius=env.comm_radius)
        if h_offset is not None:
            hnet.h_offset = float(h_offset)

    training = config['training']
    reward_scale = float(training['reward_scale'])
    reach_bonus = float(training.get('reach_bonus', 0.0))
    h_scale = float(config['h'].get('feature_scale', 1.0))
    use_h_features = bool(config['h'].get('use_features', True))
    low = float(training.get('task_gate_low', 0.0))
    high = float(training.get('task_gate_high', 0.0))
    extra_dim = int(ckpt.get('extra_dim', 5))
    n_agents = int(env.num_agents)

    all_h = []
    all_gate = []
    all_h_next = []
    all_target = []
    all_target_raw = []
    all_margin = []
    all_future_collision = []
    all_future_unsafe = []
    all_current_unsafe = []
    all_horizon_safe = []
    all_episode = []
    all_step = []
    all_agent = []
    collision_h = []
    collision_h_next = []
    collision_gate = []
    collision_h_cur_above_high = []
    collision_h_cur_above_zero = []
    collision_h_cur_in_gate = []
    collision_h_cur_below_zero = []
    collision_h_next_above_high = []
    collision_h_next_above_zero = []
    collision_h_next_below_zero = []
    collision_gate_open = []
    collision_gate_half = []
    collision_gate_closed = []
    pre_windows = {k: [] for k in range(6)}
    pre_window_high = {k: [] for k in range(6)}
    pre_window_zero = {k: [] for k in range(6)}
    episodes = []
    total_collision_steps = 0
    total_agent_steps = 0
    total_collision_events = 0

    for ep in range(args.num_episodes):
        seed = int(args.seed + ep)
        obs = env.reset(seed=seed)
        unsafe_init = np.asarray(
            env._env.unsafe_mask(env.get_graph()), dtype=bool)
        h, grad_h = hnet.value_and_local_grad()
        margin_seq = [dense_margin_values(
            obs, max_neighbors=env_cfg['max_neighbors'],
            n_rays=env_cfg['n_rays'], area_size=env_cfg['area_size'],
            comm_radius=float(env.comm_radius), include_wall=False)]
        h_seq = []
        h_next_seq = []
        gate_seq = []
        collision_seq = []
        unsafe_seq = []
        reached_seq = []
        reward_seq = []
        action_abs = []
        reached_any = np.zeros(n_agents, dtype=bool)
        collision_any = np.zeros(n_agents, dtype=bool)

        for t in range(env.max_step):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            h_policy = (h * h_scale if use_h_features else np.zeros_like(h))
            grad_policy = (grad_h * h_scale if use_h_features
                           else np.zeros_like(grad_h))
            h_t = torch.as_tensor(h_policy, dtype=torch.float32,
                                  device=device).unsqueeze(-1)
            grad_t = torch.as_tensor(grad_policy, dtype=torch.float32,
                                     device=device)
            extra = torch.zeros((n_agents, extra_dim), dtype=torch.float32,
                                device=device)
            with torch.no_grad():
                out = model(obs_t, h_t, grad_t, extra=extra)
                action = out['action'].detach().cpu().numpy()

            h_cur = np.asarray(h, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            collision = np.asarray(info['collision'], dtype=bool)
            unsafe = np.asarray(info['unsafe'], dtype=bool)
            reached = np.asarray(info['reached'], dtype=bool)
            h_next, grad_next = hnet.value_and_local_grad()
            gate_next = _gate(np.asarray(h_next), low, high)
            margin_next = dense_margin_values(
                obs, max_neighbors=env_cfg['max_neighbors'],
                n_rays=env_cfg['n_rays'], area_size=env_cfg['area_size'],
                comm_radius=float(env.comm_radius), include_wall=False)
            margin_seq.append(margin_next)

            h_seq.append(h_cur)
            h_next_seq.append(np.asarray(h_next, dtype=np.float32))
            gate_seq.append(gate_next)
            collision_seq.append(collision)
            unsafe_seq.append(unsafe)
            reached_seq.append(reached)
            reward_seq.append(np.asarray(reward, dtype=np.float32))
            action_abs.append(float(np.abs(action).mean()))
            collision_any |= collision
            reached_any |= reached
            total_agent_steps += n_agents
            total_collision_steps += int(collision.sum())

            if terminated or truncated:
                break
            h, grad_h = h_next, grad_next

        # Close the future-min window with the terminal next-state margin.
        margins = np.asarray(margin_seq, dtype=np.float32)  # [T+1,N]
        risk = horizon_min_margin(margins, horizon=32)      # [T+1,N]
        target_raw = np.tanh(risk / 0.2).astype(np.float32)
        h_seq = np.asarray(h_seq, dtype=np.float32)         # [T,N]
        h_next_seq = np.asarray(h_next_seq, dtype=np.float32)
        gate_seq = np.asarray(gate_seq, dtype=np.float32)
        collision_seq = np.asarray(collision_seq, dtype=bool)
        unsafe_seq = np.asarray(unsafe_seq, dtype=bool)
        reached_seq = np.asarray(reached_seq, dtype=bool)
        T = h_seq.shape[0]
        risk_t = risk[:T]
        target_t = target_raw[:T].copy()

        # Match the h training target exactly: the collector forces every
        # horizon-unsafe row to target <= -0.1.  The unsafe series runs over
        # states 0..T, including the reset state and the terminal next state.
        unsafe_series = np.concatenate(
            [unsafe_init[None, :], unsafe_seq], axis=0)      # [T+1,N]
        _label_seq, horizon_safe_seq = horizon_labels(
            unsafe_series, horizon=32)
        horizon_safe = horizon_safe_seq[:T]
        current_unsafe = unsafe_series[:T]
        target_t[~horizon_safe] = np.minimum(target_t[~horizon_safe], -0.1)

        # Future collision/unsafe in the next 32 steps, excluding the current
        # state.  collision_seq[t] is the event at state t+1.
        future_collision = np.zeros((T, n_agents), dtype=bool)
        future_unsafe = np.zeros((T, n_agents), dtype=bool)
        for t in range(T):
            end_c = min(T, t + 32)
            end_u = min(T + 1, t + 33)
            if end_c > t:
                future_collision[t] = collision_seq[t:end_c].any(axis=0)
            if end_u > t + 1:
                future_unsafe[t] = unsafe_series[t + 1:end_u].any(axis=0)

        tr, ag = np.meshgrid(np.arange(T), np.arange(n_agents),
                             indexing='ij')
        all_episode.append(np.full(T * n_agents, ep, dtype=np.int32))
        all_step.append(tr.reshape(-1).astype(np.int32))
        all_agent.append(ag.reshape(-1).astype(np.int32))
        all_h.append(h_seq.reshape(-1))
        all_h_next.append(h_next_seq.reshape(-1))
        all_gate.append(gate_seq.reshape(-1))
        all_target.append(target_t.reshape(-1))
        all_target_raw.append(target_raw[:T].reshape(-1))
        all_margin.append(risk_t.reshape(-1))
        all_future_collision.append(future_collision.reshape(-1))
        all_future_unsafe.append(future_unsafe.reshape(-1))
        all_current_unsafe.append(current_unsafe.reshape(-1))
        all_horizon_safe.append(horizon_safe.reshape(-1))

        # First collision event per agent avoids counting a persistent contact
        # as multiple independent events.
        prev = np.zeros(n_agents, dtype=bool)
        for t in range(T):
            new = collision_seq[t] & ~prev
            for agent in np.where(new)[0]:
                total_collision_events += 1
                collision_h.append(h_seq[t, agent])
                collision_h_next.append(h_next_seq[t, agent])
                collision_gate.append(gate_seq[t, agent])
                collision_h_cur_above_high.append(h_seq[t, agent] >= high)
                collision_h_cur_above_zero.append(h_seq[t, agent] > 0.0)
                collision_h_cur_in_gate.append(
                    (h_seq[t, agent] >= 0.0) and (h_seq[t, agent] < high))
                collision_h_cur_below_zero.append(h_seq[t, agent] <= 0.0)
                collision_h_next_above_high.append(h_next_seq[t, agent] >= high)
                collision_h_next_above_zero.append(h_next_seq[t, agent] > 0.0)
                collision_h_next_below_zero.append(h_next_seq[t, agent] <= 0.0)
                collision_gate_open.append(gate_seq[t, agent] >= 0.999)
                collision_gate_half.append(gate_seq[t, agent] >= 0.5)
                collision_gate_closed.append(gate_seq[t, agent] <= 1e-6)
                for lag in range(6):
                    tt = t - lag
                    if tt >= 0:
                        value = float(h_seq[tt, agent])
                        pre_windows[lag].append(value)
                        pre_window_high[lag].append(value >= high)
                        pre_window_zero[lag].append(value > 0.0)
            prev = collision_seq[t]

        # Match evaluate_v6 exactly: reward is a per-agent vector, and
        # reach_bonus is paid per step to every agent whose persistent
        # ``reached`` flag is true.  Accumulate per agent, then average.
        reward_sum = np.asarray(
            reward_seq, dtype=np.float64).sum(axis=0)          # [N]
        reached_steps = np.asarray(
            reached_seq, dtype=np.float64).sum(axis=0)         # [N]
        ep_return = float(np.mean(reward_sum * reward_scale
                                  + reach_bonus * reached_steps))
        episodes.append(dict(
            episode=ep, seed=seed, steps=int(T),
            episode_return=ep_return,
            safe_episode=float(not collision_any.any()),
            collision_episode=float(collision_any.any()),
            official_safety=float(np.mean(~collision_any)),
            official_reach=float(reached_any.mean()),
            official_success=float(np.mean((~collision_any) & reached_any)),
            h_min=float(h_seq.min()), h_mean=float(h_seq.mean()),
            action_abs=float(np.mean(action_abs)),
        ))

    all_h = np.concatenate(all_h).astype(np.float64)
    all_h_next = np.concatenate(all_h_next).astype(np.float64)
    all_gate = np.concatenate(all_gate).astype(np.float64)
    all_target = np.concatenate(all_target).astype(np.float64)
    all_target_raw = np.concatenate(all_target_raw).astype(np.float64)
    all_margin = np.concatenate(all_margin).astype(np.float64)
    all_future_collision = np.concatenate(all_future_collision).astype(bool)
    all_future_unsafe = np.concatenate(all_future_unsafe).astype(bool)
    all_current_unsafe = np.concatenate(all_current_unsafe).astype(bool)
    all_horizon_safe = np.concatenate(all_horizon_safe).astype(bool)
    all_episode = np.concatenate(all_episode)
    all_step = np.concatenate(all_step)
    all_agent = np.concatenate(all_agent)

    # Offline label semantics: unsafe-now vs horizon-safe, ignoring rows that
    # are safe now but unsafe within the 32-step horizon.
    graded = all_current_unsafe | all_horizon_safe
    unsafe_label = all_current_unsafe[graded]
    ignored = ~all_current_unsafe & ~all_horizon_safe
    safe_rows = all_horizon_safe
    unsafe_rows = all_current_unsafe
    calibration = dict(
        target_corr=_pearson(all_h, all_target),
        target_corr_raw=_pearson(all_h, all_target_raw),
        safe_target_corr=_pearson(all_h[safe_rows], all_target[safe_rows]),
        unsafe_target_corr=_pearson(
            all_h[unsafe_rows], all_target[unsafe_rows]),
        auroc_unsafe=_auc(-all_h[graded], unsafe_label),
        binary_at_zero=_binary_metrics(all_h[graded], unsafe_label, 0.0),
        label_rates=dict(
            current_unsafe=float(all_current_unsafe.mean()),
            horizon_safe=float(all_horizon_safe.mean()),
            ignored=float(ignored.mean()),
        ),
        h_by_label=dict(
            horizon_safe=_quantiles(all_h[all_horizon_safe]),
            current_unsafe=_quantiles(all_h[all_current_unsafe]),
            ignored=_quantiles(all_h[ignored]),
        ),
        target=_quantiles(all_target),
        target_raw=_quantiles(all_target_raw),
        margin=_quantiles(all_margin),
        future_collision_auroc=_auc(-all_h, all_future_collision),
        future_collision_count=int(all_future_collision.sum()),
        future_unsafe_auroc=_auc(-all_h, all_future_unsafe),
        future_unsafe_count=int(all_future_unsafe.sum()),
    )

    collision_count = len(collision_h)
    collision_summary = dict(
        events=collision_count,
        collision_steps=int(total_collision_steps),
        h_cur=_quantiles(np.asarray(collision_h)),
        h_next=_quantiles(np.asarray(collision_h_next)),
        gate_next=_quantiles(np.asarray(collision_gate)),
        frac_h_cur_above_high=(float(np.mean(collision_h_cur_above_high))
                               if collision_count else None),
        frac_h_cur_above_zero=(float(np.mean(collision_h_cur_above_zero))
                               if collision_count else None),
        frac_h_cur_in_gate=(float(np.mean(collision_h_cur_in_gate))
                            if collision_count else None),
        frac_h_cur_below_zero=(float(np.mean(collision_h_cur_below_zero))
                               if collision_count else None),
        frac_h_next_above_high=(float(np.mean(collision_h_next_above_high))
                                if collision_count else None),
        frac_h_next_above_zero=(float(np.mean(collision_h_next_above_zero))
                                if collision_count else None),
        frac_h_next_below_zero=(float(np.mean(collision_h_next_below_zero))
                                if collision_count else None),
        frac_gate_open=(float(np.mean(collision_gate_open))
                        if collision_count else None),
        frac_gate_half=(float(np.mean(collision_gate_half))
                        if collision_count else None),
        frac_gate_closed=(float(np.mean(collision_gate_closed))
                          if collision_count else None),
    )

    pre = {}
    for lag in range(6):
        vals = np.asarray(pre_windows[lag], dtype=np.float64)
        pre[f'lag_{lag}'] = dict(
            count=int(vals.size),
            mean_h=float(vals.mean()) if vals.size else None,
            median_h=float(np.median(vals)) if vals.size else None,
            frac_above_high=(float(np.mean(pre_window_high[lag]))
                             if vals.size else None),
            frac_above_zero=(float(np.mean(pre_window_zero[lag]))
                             if vals.size else None),
        )

    verification = None
    if args.reference_eval:
        ref = json.load(open(args.reference_eval))
        ref_m = ref.get('metrics', ref)
        verification = {
            'reference': str(args.reference_eval),
            'reference_safe_episode': ref_m.get('safe_episode'),
            'audit_safe_episode': float(np.mean(
                [e['safe_episode'] for e in episodes])),
            'reference_official_safety': ref_m.get('official_safety'),
            'audit_official_safety': float(np.mean(
                [e['official_safety'] for e in episodes])),
            'reference_official_success': ref_m.get('official_success'),
            'audit_official_success': float(np.mean(
                [e['official_success'] for e in episodes])),
            'reference_episode_return': ref_m.get('episode_return'),
            'audit_episode_return': float(np.mean(
                [e['episode_return'] for e in episodes])),
        }

    payload = dict(
        schema='v6.nav.h2_audit.v1',
        tag=args.tag,
        checkpoint=str(args.checkpoint),
        h_checkpoint=str(h_path),
        training_seed=config.get('seed'),
        eval_seed=int(args.seed),
        episodes=int(args.num_episodes),
        device=str(device),
        protocol=dict(
            steps=int(env.max_step),
            horizon=32,
            gate_low=low,
            gate_high=high,
            gate_scope=str(training.get('task_gate_scope', '')),
            h_offset=float(getattr(hnet, 'h_offset', 0.0)),
            target=('tanh(future_min_margin/0.2), wall-free, horizon 32; '
                    'horizon-unsafe rows forced <= -0.1 (training target)'),
        ),
        aggregate=dict(
            agent_steps=int(total_agent_steps),
            collision_steps=int(total_collision_steps),
            collision_events=int(total_collision_events),
            collision_episodes=int(sum(e['collision_episode']
                                       for e in episodes)),
            safe_episode=float(np.mean([e['safe_episode']
                                        for e in episodes])),
            official_safety=float(np.mean([e['official_safety']
                                           for e in episodes])),
            official_success=float(np.mean([e['official_success']
                                            for e in episodes])),
            official_reach=float(np.mean([e['official_reach']
                                          for e in episodes])),
            episode_return=float(np.mean([e['episode_return']
                                          for e in episodes])),
        ),
        h_all=_quantiles(all_h),
        h_next_all=_quantiles(all_h_next),
        gate_all=dict(
            quantiles=_quantiles(all_gate),
            mean=float(all_gate.mean()),
            closed_frac=float(np.mean(all_gate < 0.5)),
            fully_closed_frac=float(np.mean(all_gate <= 1e-6)),
            open_frac=float(np.mean(all_gate >= 0.999)),
        ),
        collision=collision_summary,
        pre_collision=pre,
        calibration=calibration,
        verification=verification,
        episode_rows=episodes,
    )
    out = _cwd_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out.with_suffix('.npz')
    np.savez_compressed(
        npz_path,
        h=all_h.astype(np.float32),
        h_next=all_h_next.astype(np.float32),
        gate_next=all_gate.astype(np.float32),
        target=all_target.astype(np.float32),
        target_raw=all_target_raw.astype(np.float32),
        margin=all_margin.astype(np.float32),
        current_unsafe=all_current_unsafe,
        horizon_safe=all_horizon_safe,
        future_collision=all_future_collision,
        future_unsafe=all_future_unsafe,
        episode=all_episode,
        step=all_step,
        agent=all_agent,
    )
    payload['data_npz'] = str(npz_path)
    out.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({k: payload[k] for k in (
        'aggregate', 'h_all', 'gate_all', 'collision', 'calibration',
        'verification')}, indent=2))
    print(f'saved: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
