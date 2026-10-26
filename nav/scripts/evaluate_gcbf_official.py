"""Evaluate official GCBF/GCBF+ checkpoints on fixed scene keys.

The scene keys are ``jax.random.PRNGKey(seed)`` for seeds 10000..10099, which is
the same reset convention used by the V5 official-environment adapter. This is
intentionally not the split-key convention inside the upstream ``test.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from algos.baselines.gcbf_official import evaluate_official, load_official_model


def _cwd_path(path):
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', required=True,
                        help='official model directory containing config.yaml')
    parser.add_argument('--step', type=int, default=1000)
    parser.add_argument('--num-agents', type=int, default=8)
    parser.add_argument('--num-obstacles', type=int, default=8)
    parser.add_argument('--n-rays', type=int, default=32)
    parser.add_argument('--max-step', type=int, default=256)
    parser.add_argument('--area-size', type=float, default=4.0)
    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--out')
    args = parser.parse_args()

    env, algo, config, step = load_official_model(
        _resolve(args.path), step=args.step,
        num_agents=args.num_agents, num_obstacles=args.num_obstacles,
        n_rays=args.n_rays, max_step=args.max_step,
        area_size=args.area_size)
    result = evaluate_official(
        env, algo, seeds=[args.seed + i for i in range(args.num_episodes)],
        max_step=args.max_step)
    payload = dict(
        path=str(args.path),
        algo=getattr(config, 'algo', None),
        step=int(step),
        episodes=args.num_episodes,
        eval_seed=args.seed,
        seed=args.seed,
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
