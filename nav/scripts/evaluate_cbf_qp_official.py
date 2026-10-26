"""Evaluate official centralized CBF-QP on fixed scenes (OSQP solver fallback)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from algos.baselines.cbf_qp_official import build_qp_matrices, evaluate
from envs.gcbfplus import OfficialDoubleIntegrator


def _cwd_path(path):
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-agents', type=int, default=8)
    parser.add_argument('--num-obstacles', type=int, default=8)
    parser.add_argument('--n-rays', type=int, default=32)
    parser.add_argument('--max-step', type=int, default=256)
    parser.add_argument('--area-size', type=float, default=4.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=10000)
    parser.add_argument('--out')
    args = parser.parse_args()

    env = OfficialDoubleIntegrator(
        num_agents=args.num_agents, area_size=args.area_size,
        max_step=args.max_step, num_obstacles=args.num_obstacles,
        n_rays=args.n_rays, max_neighbors=5, seed=args.seed)
    env.reset(seed=args.seed)
    matrices = build_qp_matrices(env, k=args.k)
    result = evaluate(
        env, matrices, seeds=[args.seed + i for i in range(args.num_episodes)],
        alpha=args.alpha)
    payload = dict(
        method='centralized_cbf_qp',
        solver='osqp',
        alpha=args.alpha,
        k=args.k,
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
