"""Evaluate an official-MAPPO base checkpoint with fixed scene seeds."""

import argparse
import json
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from algos.stage2.train_official_mappo import load_official_policy
from utils.evaluation import evaluate_policy
from utils.ppo import Features
from utils.runtime import setup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/navigation_double_integrator.yaml')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out')
    args = parser.parse_args()

    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    device = setup(config, args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = load_official_policy(checkpoint, device)
    metrics = evaluate_policy(
        model, Features(config, device), config['env'],
        args.num_episodes, args.seed)
    result = dict(
        checkpoint=str(args.checkpoint),
        episodes=args.num_episodes,
        seed=args.seed,
        env_config=config['env'],
        metrics=metrics,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + '\n')
        print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
