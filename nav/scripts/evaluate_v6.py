"""Evaluate a V6 checkpoint with its V6-learned h and optional V_T."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from envs.gcbfplus import OfficialDoubleIntegrator  # noqa: E402
from v6.official_h import provider_from_checkpoint as official_h_provider  # noqa: E402
from v6.provenance import assert_independent_path  # noqa: E402
from v6.safety import LocalSafetyProvider  # noqa: E402
from v6.training import evaluate_v6, load_v6_checkpoint  # noqa: E402


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _cwd_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def _device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--h-checkpoint',
                    help='V6 self-trained h checkpoint; official h is rejected')
    ap.add_argument('--num-episodes', type=int, default=100)
    ap.add_argument('--seed', type=int, default=10000)
    ap.add_argument('--out')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--stochastic', action='store_true',
                    help='sample actions for Ant-style robustness evaluation')
    args = ap.parse_args()

    device = _device(args.device)
    ckpt_path = assert_independent_path(
        _cwd_path(args.checkpoint), kind='V6 policy checkpoint')
    model, task_net, ema, ckpt = load_v6_checkpoint(
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
    # The h provider warms up on a real graph, so initialize the env first.
    env.reset(seed=int(config['seed']))
    if args.h_checkpoint:
        h_path = _cwd_path(args.h_checkpoint)
    else:
        # Co-trained policies must be evaluated with their paired h.  Prefer
        # the h saved next to the policy instead of silently using h0.
        names = (['h_best.pkl', 'h_final.pkl']
                 if 'best' in ckpt_path.name.lower()
                 else ['h_final.pkl', 'h_best.pkl'])
        sibling = next((ckpt_path.parent / name for name in names
                        if (ckpt_path.parent / name).exists()), None)
        h_path = sibling if sibling is not None else _resolve(
            config['h']['checkpoint'])
    h_path = assert_independent_path(h_path, kind='V6 evaluation h')
    h_offset = config['h'].get('h_offset')
    if h_path.suffix.lower() in ('.pkl', '.pickle'):
        hnet = official_h_provider(
            h_path, env, h_offset=None if h_offset is None else float(h_offset),
            h_scale=config['h'].get('h_scale'))
    else:
        hnet = LocalSafetyProvider.from_checkpoint(
            h_path, env, device=device,
            max_neighbors=env_cfg['max_neighbors'],
            n_rays=env_cfg['n_rays'],
            comm_radius=env.comm_radius)
        if h_offset is not None:
            hnet.h_offset = float(h_offset)
    result = evaluate_v6(
        env, hnet, task_net, ema, model,
        seeds=[args.seed + i for i in range(args.num_episodes)],
        device=device,
        reward_scale=float(config['training']['reward_scale']),
        reach_bonus=float(config['training'].get('reach_bonus', 0.0)),
        use_h_features=bool(config['h'].get('use_features', True)),
        h_feature_scale=float(config['h'].get('feature_scale', 1.0)),
        max_neighbors=env_cfg['max_neighbors'],
        n_rays=env_cfg['n_rays'],
        task_grad_scale=float(config['training'].get('task_grad_scale', 1.0)),
        use_task_scalar=bool(
            config['training'].get('use_task_scalar', True)),
        use_task_grad=bool(config['training'].get('use_task_grad', True)),
        cbf_alpha=float(config['training'].get('cbf_alpha', 1.0)),
        cbf_eps=float(config['training'].get('cbf_eps', 0.02)),
        mass=float(config['training'].get('mass', 0.1)),
        stochastic=bool(args.stochastic))
    payload = dict(
        checkpoint=str(args.checkpoint), h_checkpoint=str(h_path),
        steps=ckpt['steps'], episodes=args.num_episodes,
        training_seed=config.get('seed'), eval_seed=args.seed,
        stochastic=bool(args.stochastic), metrics=result)
    print(json.dumps(payload, indent=2))
    if args.out:
        out = _cwd_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + '\n')
        print(f'saved: {out}')


if __name__ == '__main__':
    main()
