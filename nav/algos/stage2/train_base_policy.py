"""Shared decentralized actor and agent-conditioned centralized MAPPO critic."""

import argparse
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
from algos.stage2.train_official_mappo import train_official_mappo


def train_base_policy(config_path, output_dir, device='cuda', resume_from=None):
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    if resume_from is not None:
        raise ValueError('Official MAPPO checkpoints use a different format; start a new run')
    return train_official_mappo(config, output_dir, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/navigation_double_integrator.yaml')
    parser.add_argument('--output', default='outputs/stage2')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume')
    args = parser.parse_args()
    train_base_policy(args.config, args.output, args.device, args.resume)
