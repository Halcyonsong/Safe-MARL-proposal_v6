"""Collect an offline safety dataset for official V5 learned h.

The dataset uses only official Ant cost thresholds and episode outcome labels.
It does not store or use task return, QP targets, or expert u_ref actions.

Behavior sources:
  --actor-dir DIR             upstream MAPPO / MAPPO-Lag actor directory
  --policy-checkpoint FILE    existing V5 QP-free policy checkpoint

Example:
  python scripts/collect_ant_safety_data.py \
      --actor-dir outputs/v5/ant/baselines/lag_ft_2m_models \
      --policy-checkpoint outputs/v5/ant/safety_only_pure_r005_g015_u80/policy_best.pth \
      --episodes 80 --noise-std 0.05 \
      --out outputs/v5/official/ant/learned_h/safety_data.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from algos.safe_policy.qp_free_ant import QpFreeAntEnsemble  # noqa: E402
from envs.mamujoco import Ant2x4Env  # noqa: E402
from scripts.train_ant_qp_free import (  # noqa: E402
    base_action,
    load_base_actors,
)


def _resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _load_policy(path, env, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = (ckpt.get('config') or {}).get('config', {})
    model = QpFreeAntEnsemble(
        env.num_agents,
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        h_dim=4,
        grad_dim=9,
        hidden=128,
        correction_radius=cfg.get('correction_radius', 0.03),
        gate_margin=cfg.get('gate_margin', 0.1),
        gate_tau=cfg.get('gate_tau', 0.05),
        gate_on_wall=cfg.get('gate_on_wall', True),
        initial_log_std=cfg.get('initial_log_std', -3.0),
    )
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    model.eval()
    return model, dict(path=str(path), kind='v5_policy', config=cfg)


def _actor_spec(path, env, device):
    return dict(
        kind='actor',
        path=str(path),
        actors=load_base_actors(env, device, path),
    )


def _choose_action(spec, obs, h, grad, state, env, device, noise_std, rng):
    if spec['kind'] == 'random':
        return rng.uniform(-1.0, 1.0, size=(env.num_agents, env.action_dim)).astype(np.float32)
    if spec['kind'] == 'actor':
        action, state['rnn'] = base_action(
            spec['actors'], obs, state['rnn'], state['masks'], device)
    else:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        h_t = torch.as_tensor(h, dtype=torch.float32, device=device)
        grad_t = torch.as_tensor(grad, dtype=torch.float32, device=device)
        with torch.no_grad():
            out = spec['model'](obs_t, h_t, grad_t)
        action = out['action'].cpu().numpy()
    if noise_std:
        action = action + rng.normal(0.0, noise_std, size=action.shape)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def collect_source(
    env,
    spec,
    episodes,
    seed,
    device,
    noise_std,
    rng,
):
    states = []
    next_states = []
    costs = []
    terminals = []
    episode_rows = []
    lengths = []
    cost_rates = []
    survived = []

    for ep in range(episodes):
        ep_seed = int(seed + ep)
        obs, h, grad, hv, hs, g = env.reset(seed=ep_seed)
        geom = env.get_geometry()
        if spec['kind'] == 'actor':
            spec['state'] = dict(
                rnn=[torch.zeros(1, 1, 64, device=device)
                     for _ in range(env.num_agents)],
                masks=torch.ones(1, 1, device=device),
            )
        else:
            spec['state'] = {}
        rows = []
        ep_cost = []
        ep_term = False
        for _ in range(env.max_step):
            action = _choose_action(
                spec, obs, h, grad, spec['state'], env, device, noise_std, rng)
            # Current-state labels are threshold-aligned official margins.
            label = (np.asarray(hv, dtype=np.float64) < 0.0).astype(np.float32)
            rows.append(dict(
                state=geom.copy(),
                cost_labels=label,
                next_state=None,
            ))
            obs, h, grad, hv_next, hs_next, g, term, trunc, info = env.step(action)
            geom_next = env.get_geometry()
            rows[-1]['next_state'] = geom_next.copy()
            ep_cost.append(float(info['cost']))
            ep_term = bool(term)
            # Add the post-step state when it is a height/upright violation or
            # an early termination.  Initial/current states are usually safe,
            # so without this the floor/ceil/upright heads see no positives.
            if term or bool(np.any(np.asarray(hv_next)[1:] < 0.0)):
                rows.append(dict(
                    state=geom_next.copy(),
                    cost_labels=(np.asarray(hv_next, dtype=np.float64) < 0.0).astype(np.float32),
                    next_state=geom_next.copy(),
                ))
            hv, hs = hv_next, hs_next
            geom = geom_next
            if term or trunc:
                break
        # Trajectory-level survival label: every transition in a failed
        # episode gets terminal=1, so h_surv learns to predict eventual failure.
        terminal = float(ep_term)
        for row in rows:
            states.append(row['state'])
            next_states.append(row['next_state'])
            costs.append(row['cost_labels'])
            terminals.append(terminal)
        lengths.append(len(rows))
        cost_rates.append(float(np.mean(ep_cost)) if ep_cost else 0.0)
        survived.append(float(not ep_term))
        episode_rows.append(dict(
            seed=ep_seed,
            length=len(rows),
            cost_rate=cost_rates[-1],
            terminal=terminal,
            survived=survived[-1],
        ))

    return dict(
        states=np.asarray(states, dtype=np.float32),
        next_states=np.asarray(next_states, dtype=np.float32),
        cost_labels=np.asarray(costs, dtype=np.float32),
        terminal=np.asarray(terminals, dtype=np.float32),
        episodes=episode_rows,
        lengths=lengths,
        cost_rates=cost_rates,
        survived=survived,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--episodes', type=int, default=80,
                    help='total episodes, split across behavior sources')
    ap.add_argument('--seed', type=int, default=10000)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--noise-std', type=float, default=0.0)
    ap.add_argument('--actor-dir', action='append', default=[])
    ap.add_argument('--policy-checkpoint', action='append', default=[])
    ap.add_argument('--random-episodes', type=int, default=0,
                    help='extra uniform-random exploration episodes')
    args = ap.parse_args()

    if not args.actor_dir and not args.policy_checkpoint and args.random_episodes <= 0:
        raise SystemExit('at least one behavior source is required')
    device = torch.device(args.device)
    env = Ant2x4Env(seed=args.seed, episode_limit=1000)
    specs = []
    for path in args.actor_dir:
        specs.append(_actor_spec(_resolve(path), env, device))
    for path in args.policy_checkpoint:
        model, meta = _load_policy(_resolve(path), env, device)
        specs.append(dict(kind='policy', path=meta['path'], model=model,
                          config=meta['config']))
    if args.random_episodes > 0:
        specs.append(dict(kind='random', path='uniform_random'))
    n = len(specs)
    base = args.episodes // n
    extra = args.episodes % n
    counts = [base + (1 if i < extra else 0) for i in range(n)]
    if args.random_episodes > 0:
        counts[-1] = int(args.random_episodes)
    rng = np.random.default_rng(args.seed)

    states = []
    next_states = []
    costs = []
    terminals = []
    metadata = []
    for spec, count in zip(specs, counts):
        out = collect_source(
            env, spec, count, args.seed + len(metadata) * 1000,
            device, args.noise_std, rng)
        states.append(out['states'])
        next_states.append(out['next_states'])
        costs.append(out['cost_labels'])
        terminals.append(out['terminal'])
        metadata.append(dict(
            path=spec['path'],
            kind=spec['kind'],
            episodes=count,
            transitions=int(len(out['states'])),
            cost_rate=float(np.mean(out['cost_rates'])),
            survived=float(np.mean(out['survived'])),
            source_cost_positive_rate=float(np.mean(out['cost_labels'])),
            source_terminal_rate=float(np.mean(out['terminal'])),
        ))
        print(json.dumps(metadata[-1]), flush=True)
    env.close()

    data = dict(
        states=np.concatenate(states, axis=0),
        next_states=np.concatenate(next_states, axis=0),
        cost_labels=np.concatenate(costs, axis=0),
        terminal=np.concatenate(terminals, axis=0),
    )
    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    summary = dict(
        schema='v5.ant.safety_data.v1',
        out=str(out_path),
        transitions=int(len(data['states'])),
        cost_positive_rate=float(data['cost_labels'].mean()),
        terminal_rate=float(data['terminal'].mean()),
        cost_component_rates=dict(zip(
            ['wall', 'floor', 'ceil', 'upright'],
            [float(v) for v in data['cost_labels'].mean(axis=0)])),
        sources=metadata,
        note='official cost thresholds + trajectory outcome only; no return/QP/u_ref',
    )
    (out_path.parent / 'safety_data_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
