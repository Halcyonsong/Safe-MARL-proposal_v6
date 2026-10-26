"""Deterministic evaluation for V5 QP-free checkpoints on official DoubleIntegrator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

import torch

from algos.safe_policy import OfficialFixedH
from v6.official_h import (
    load_official_h_checkpoint, provider_from_checkpoint)
from v6.safety import LocalSafetyProvider
from algos.safe_policy.training import evaluate, load_checkpoint
from envs.gcbfplus import OfficialDoubleIntegrator


def _resolve(path):
    """Resolve repo-relative assets such as the h checkpoint."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _cwd_path(path):
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--out')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--h-checkpoint',
                        help='override h: official cbf.pkl, learned .pkl, or .pt')
    parser.add_argument('--h-offset', type=float,
                        help='override provider h offset')
    parser.add_argument('--h-scale', type=float,
                        help='diagnostic affine h scale; default 1')
    parser.add_argument('--cbf-alpha', type=float, default=1.0)
    parser.add_argument('--cbf-eps', type=float, default=0.02)
    args = parser.parse_args()

    model, checkpoint = load_checkpoint(
        _cwd_path(args.checkpoint), device=args.device)
    config = checkpoint['config']
    env_config = config['env']
    env = OfficialDoubleIntegrator(
        num_agents=env_config['num_agents'],
        area_size=env_config['area_size'],
        max_step=env_config['max_step'],
        num_obstacles=env_config['num_obstacles'],
        n_rays=env_config['n_rays'],
        max_neighbors=env_config['max_neighbors'],
        seed=config['seed'],
    )
    env.reset(seed=config['seed'])
    h_path = (_cwd_path(args.h_checkpoint) if args.h_checkpoint
              else _resolve(config['h']['checkpoint']))
    h_offset = args.h_offset
    if h_path.suffix.lower() in ('.pkl', '.pickle'):
        try:
            h_ckpt = load_official_h_checkpoint(h_path)
        except Exception:
            h_ckpt = {}
        if h_ckpt.get('schema') == 'v6-official-cbf-h':
            hnet = provider_from_checkpoint(
                h_path, env, h_offset=h_offset, h_scale=args.h_scale)
        else:
            hnet = OfficialFixedH(
                h_path, env, n_agents=env_config['num_agents'],
                gnn_layers=config['h'].get('gnn_layers', 1))
            if h_offset is not None:
                hnet.h_offset = float(h_offset)
    else:
        hnet = LocalSafetyProvider.from_checkpoint(
            h_path, env, device=args.device,
            max_neighbors=env_config['max_neighbors'],
            n_rays=env_config['n_rays'],
            comm_radius=env.comm_radius)
        if h_offset is not None:
            hnet.h_offset = float(h_offset)
    result = evaluate(
        env, hnet, model,
        seeds=[args.seed + i for i in range(args.num_episodes)],
        device=torch.device(args.device),
        reward_scale=float(config['training']['reward_scale']),
        reach_bonus=float(config['training'].get('reach_bonus', 0.0)),
        use_h_features=bool(config.get('h', {}).get(
            'use_features', True)),
        cbf_alpha=float(args.cbf_alpha),
        cbf_eps=float(args.cbf_eps))
    payload = dict(
        checkpoint=str(args.checkpoint), h_checkpoint=str(h_path),
        steps=checkpoint['steps'],
        episodes=args.num_episodes,
        training_seed=config.get('seed'),
        eval_seed=args.seed,
        # Kept for compatibility with earlier eval files.
        seed=args.seed,
        metrics=result)
    print(json.dumps(payload, indent=2))
    if args.out:
        out = _cwd_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + '\n')
        print(f'saved: {out}')


if __name__ == '__main__':
    main()
