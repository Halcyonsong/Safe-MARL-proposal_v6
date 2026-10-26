"""Deterministic evaluation for MAPPO-Lagrangian checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NAV = Path(__file__).resolve().parents[1]
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

import torch

from algos.baselines.mappo_lag_official import evaluate, load_mappo_lag_checkpoint
from envs.gcbfplus import OfficialDoubleIntegrator


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
    args = parser.parse_args()

    model, checkpoint = load_mappo_lag_checkpoint(
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
        seed=config['seed'])
    env.reset(seed=config['seed'])
    result = evaluate(
        env, model, seeds=[args.seed + i for i in range(args.num_episodes)],
        device=torch.device(args.device),
        reward_scale=float(config['training']['reward_scale']),
        reach_bonus=float(config['training']['reach_bonus']))
    payload = dict(
        checkpoint=str(args.checkpoint),
        steps=checkpoint['steps'],
        episodes=args.num_episodes,
        training_seed=config.get('seed'),
        eval_seed=args.seed,
        seed=args.seed,
        multiplier=checkpoint.get('multiplier'),
        metrics=result,
    )
    print(json.dumps(payload, indent=2))
    if args.out:
        out = _cwd_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + '\n')
        print(f'saved: {out}')


if __name__ == '__main__':
    main()
