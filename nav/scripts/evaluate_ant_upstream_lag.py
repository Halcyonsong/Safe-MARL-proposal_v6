"""V6-compatible evaluation of an upstream MAPPO-Lagrangian Ant actor.

The upstream ``mappo_lag`` trainer saves one actor per agent under
``.../models/actor_agent{i}.pt``.  This adapter loads those actors without
importing any V5/V4 safety module, runs the V6 Ant-2x4 environment, and reports
the same metrics as ``evaluate_v6_ant_mappo.py``.

It is an **external baseline adapter**: the actors are trained by the upstream
MAPPO-Lagrangian code, not by V6, and the result is not a V6 main claim.

Example:
    PYTHONPATH=nav LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin \
    /home/xiang/miniconda3/envs/mappo_lag/bin/python \
      nav/scripts/evaluate_ant_upstream_lag.py \
      --models-dir <dir with actor_agent0.pt, actor_agent1.pt> \
      --num-episodes 100 --seed 20000 --stochastic \
      --out outputs/.../eval_100ep_stochastic.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from v6.ant_env import Ant2x4Env  # noqa: E402

V2 = Path(os.environ.get(
    'SAFE_MAMUJOCO_ROOT',
    str(ROOT.parent / 'Safe-MARL-proposal_v2'))).expanduser()
PKG = V2 / 'third_party/mappo_lag/MAPPO-Lagrangian'
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _actor_args():
    """Architecture used by the upstream Ant 2x4 MAPPO-Lag recipe."""
    return SimpleNamespace(
        hidden_size=64, layer_N=1, use_ReLU=True, use_orthogonal=True,
        use_feature_normalization=True, use_naive_recurrent_policy=False,
        use_recurrent_policy=False, recurrent_N=1, gain=0.01,
        stacked_frames=1, use_policy_active_masks=True,
        std_x_coef=1.0, std_y_coef=0.5)


def load_upstream_actors(env, device, models_dir):
    from mappo_lagrangian.algorithms.r_mappo.algorithm.r_actor_critic import (
        R_Actor,
    )
    models_dir = Path(models_dir)
    actors = []
    for i in range(env.num_agents):
        actor_path = models_dir / f'actor_agent{i}.pt'
        if not actor_path.exists():
            raise FileNotFoundError(actor_path)
        actor = R_Actor(
            _actor_args(), env.env.observation_space[i],
            env.env.action_space[i], device)
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        actor.eval()
        for param in actor.parameters():
            param.requires_grad = False
        actors.append(actor)
    return actors


def evaluate(env, actors, *, episodes, seed, device, deterministic):
    rows = []
    for episode in range(int(episodes)):
        obs, h, grad, _hvec, _hs, _geom, _state = env.reset(
            seed=int(seed) + episode)
        rnn = [torch.zeros(1, 1, 64, device=device)
               for _ in range(env.num_agents)]
        masks = torch.ones(1, 1, device=device)

        ep_return = 0.0
        ep_forward = 0.0
        ep_ctrl = 0.0
        ep_survive = 0.0
        ep_cost = 0.0
        ep_wall = 0.0
        ep_topple = 0.0
        ep_cost_steps = 0
        ep_wall_steps = 0
        ep_topple_steps = 0
        ep_len = 0
        h_values = []
        x_values = []
        action_abs = []
        truncated = False

        for _ in range(int(env.max_step)):
            actions = []
            new_rnn = []
            for i, actor in enumerate(actors):
                o = torch.as_tensor(
                    obs[i][None], dtype=torch.float32, device=device)
                with torch.no_grad():
                    a, _, r = actor(
                        o, rnn[i], masks, None, bool(deterministic))
                actions.append(a.cpu().numpy()[0])
                new_rnn.append(r)
            actions = np.stack(actions, axis=0)
            rnn = new_rnn

            (obs, h, grad, _hvec, _hs, _geom, _state,
             reward, cost, terminated, truncated, info) = env.step(actions)
            ep_return += float(reward)
            ep_forward += float(info.get('forward_reward', 0.0))
            ep_ctrl += float(info.get('ctrl_reward', 0.0))
            ep_survive += float(info.get('survive_reward', 0.0))
            ep_cost = max(ep_cost, float(cost))
            ep_wall = max(ep_wall, float(info.get('wall', 0.0)))
            ep_topple = max(ep_topple, float(info.get('topple', 0.0)))
            ep_cost_steps += int(float(cost) > 0.0)
            ep_wall_steps += int(float(info.get('wall', 0.0)) > 0.0)
            ep_topple_steps += int(float(info.get('topple', 0.0)) > 0.0)
            ep_len += 1
            h_values.append(float(info.get('h', 0.0)))
            x_values.append(float(info.get('x', 0.0)))
            action_abs.append(float(np.mean(np.abs(actions))))
            if terminated or truncated:
                break

        rows.append(dict(
            safe_episode=1.0 if ep_cost <= 0.0 else 0.0,
            unsafe_episode=1.0 if ep_cost > 0.0 else 0.0,
            wall_episode=1.0 if ep_wall > 0.0 else 0.0,
            topple_episode=1.0 if ep_topple > 0.0 else 0.0,
            episode_return=float(ep_return),
            forward_return=float(ep_forward),
            ctrl_return=float(ep_ctrl),
            survive_return=float(ep_survive),
            episode_length=float(ep_len),
            survived=1.0 if truncated else 0.0,
            cost_rate=float(ep_cost),
            cost_fraction=float(ep_cost_steps / max(ep_len, 1)),
            wall_fraction=float(ep_wall_steps / max(ep_len, 1)),
            topple_fraction=float(ep_topple_steps / max(ep_len, 1)),
            h_mean=float(np.mean(h_values)) if h_values else 0.0,
            h_min=float(np.min(h_values)) if h_values else 0.0,
            h_neg_frac=float(np.mean(np.asarray(h_values) < 0.0))
            if h_values else 0.0,
            final_x=float(x_values[-1]) if x_values else 0.0,
            mean_x=float(np.mean(x_values)) if x_values else 0.0,
            action_abs=float(np.mean(action_abs)) if action_abs else 0.0,
        ))

    summary = {key: float(np.mean([row[key] for row in rows]))
               for key in rows[0]}
    summary['episodes'] = int(episodes)
    summary['eval_seed_start'] = int(seed)
    summary['deterministic'] = 1.0 if deterministic else 0.0
    return summary, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models-dir', required=True,
                    help='directory with actor_agent0.pt, actor_agent1.pt')
    ap.add_argument('--num-episodes', type=int, default=100)
    ap.add_argument('--seed', type=int, default=20000)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--stochastic', action='store_true')
    ap.add_argument('--training-seed', type=int, default=None)
    ap.add_argument('--episode-limit', type=int, default=1000)
    ap.add_argument('--out')
    args = ap.parse_args()

    models_dir = _resolve(args.models_dir)
    device = _device(args.device)
    env = Ant2x4Env(seed=int(args.seed),
                    episode_limit=int(args.episode_limit), agent_obsk=1)
    actors = load_upstream_actors(env, device, models_dir)
    metrics, rows = evaluate(
        env, actors, episodes=int(args.num_episodes), seed=int(args.seed),
        device=device, deterministic=not bool(args.stochastic))
    result = dict(
        schema='v6.ant_upstream_lag.eval.v1',
        source='upstream mappo_lagr',
        models_dir=str(models_dir),
        training_seed=args.training_seed,
        stochastic=bool(args.stochastic),
        episodes=int(args.num_episodes),
        seed=int(args.seed),
        env=env.get_env_info(),
        metrics=metrics,
        episodes_data=rows,
    )
    text = json.dumps(result, indent=2)
    if args.out:
        out = _resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + '\n')
    env.close()
    print(text)


if __name__ == '__main__':
    main()
