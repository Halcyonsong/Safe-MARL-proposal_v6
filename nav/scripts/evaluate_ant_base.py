"""Deterministic fixed-seed evaluation of an upstream Ant MAPPO/Lag actor."""

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

from envs.mamujoco import Ant2x4Env  # noqa: E402
from scripts.train_ant_qp_free import load_base_actors  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--num-episodes', type=int, default=10)
    ap.add_argument('--seed', type=int, default=10000)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out')
    ap.add_argument('--shield-alpha', type=float, default=1.0,
                    help='multiply actions by this when h is below margin')
    ap.add_argument('--shield-margin', type=float, default=-1e9)
    args = ap.parse_args()
    device = torch.device(args.device)
    env = Ant2x4Env(seed=args.seed, episode_limit=1000)
    actors = load_base_actors(env, device, Path(args.ckpt))
    rows = []
    rnn_template = [torch.zeros(1, 1, 64, device=device)
                    for _ in range(env.num_agents)]
    for seed in range(args.seed, args.seed + args.num_episodes):
        obs, h, grad, hv, hs, g = env.reset(seed=seed)
        h_scalar = float(np.min(h))
        rnn = [r.clone() for r in rnn_template]
        masks = torch.ones(1, 1, device=device)
        costs, walls, topples, hvals = [], [], [], []
        ret = 0.0
        x_max = float(env.get_torso_xyz()[0])
        x_final = x_max
        length = 0
        while True:
            actions = []
            new_rnn = []
            for i, actor in enumerate(actors):
                o = torch.as_tensor(obs[i][None], dtype=torch.float32, device=device)
                with torch.no_grad():
                    a, _, r2 = actor(o, rnn[i], masks, None, True)
                actions.append(a.cpu().numpy()[0])
                new_rnn.append(r2)
            actions = np.stack(actions, axis=0)
            if h_scalar < args.shield_margin:
                actions = actions * args.shield_alpha
            obs, h, grad, hv, hs, g, term, trunc, info = env.step(actions)
            h_scalar = float(np.min(h))
            rnn = new_rnn
            ret += float(info['reward'])
            costs.append(float(info['cost']))
            walls.append(float(info['wall']))
            topples.append(float(info['topple']))
            hvals.append(float(info['h']))
            x_final = float(info['x'])
            x_max = max(x_max, x_final)
            length += 1
            if term or trunc:
                break
        rows.append(dict(
            return_=ret,
            cost_rate=float(np.mean(costs)),
            wall_rate=float(np.mean(walls)),
            topple_rate=float(np.mean(topples)),
            safe_episode=float(np.max(costs) == 0),
            wall_episode=float(np.max(walls) > 0),
            topple_episode=float(np.max(topples) > 0),
            survived=float(length >= env.max_step),
            length=length,
            h_mean=float(np.mean(hvals)),
            h_min=float(np.min(hvals)),
            h_neg_frac=float(np.mean(np.asarray(hvals) < 0)),
            x_final=x_final,
            x_max=x_max,
        ))
    metrics = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    metrics_std = {k: float(np.std([r[k] for r in rows])) for k in rows[0]}
    payload = dict(ckpt=args.ckpt, episodes=args.num_episodes,
                   eval_seed=args.seed, metrics=metrics,
                   metrics_std=metrics_std, episodes_data=rows)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + '\n')
    env.close()


if __name__ == '__main__':
    main()
