#!/usr/bin/env python3
"""Collect Ant rollouts for offline V_T training.

The saved data contains only the official task reward (no h gate, no safety
cost).  V_T is trained later by ``train_v6_ant_task_value.py``; this script
never updates the actor or h.
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

from v6.ant_env import Ant2x4Env  # noqa: E402
from v6.ant_h import AntSafetyProvider  # noqa: E402
from v6.ant_mappo import load_ant_mappo_model  # noqa: E402
from v6.ant_safety import NullAntSafetyProvider  # noqa: E402
from v6.ant_task_value import (  # noqa: E402
    load_ant_task_value, make_task_feature_fn)
from v6.provenance import assert_independent_path  # noqa: E402


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--h-checkpoint')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--seed', type=int, default=30000)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = _device(args.device)
    ckpt_path = _resolve(args.checkpoint)
    assert_independent_path(ckpt_path, kind='V6 Ant collector checkpoint')
    model, ckpt = load_ant_mappo_model(ckpt_path, device=device)
    cfg = dict(ckpt.get('config', {}))
    env_cfg = dict(cfg.get('env_cfg', {}))
    h_path = args.h_checkpoint or ckpt.get('h_checkpoint')
    if model.use_h_features:
        if not h_path:
            raise ValueError('checkpoint uses h features but no h checkpoint')
        h_path = _resolve(h_path)
        assert_independent_path(h_path, kind='V6 Ant h checkpoint')
        provider = AntSafetyProvider.from_checkpoint(h_path, device=device)
    else:
        provider = NullAntSafetyProvider()
        h_path = None
    task_value_path = (cfg.get('task_value') or {}).get('checkpoint')
    task_feature_fn = None
    if model.use_task_value:
        if not task_value_path:
            raise ValueError('V_T checkpoint missing from the policy config')
        task_value = load_ant_task_value(
            _resolve(task_value_path), device=device)
        task_feature_fn = make_task_feature_fn(task_value, device=device)

    env = Ant2x4Env(
        seed=int(args.seed),
        episode_limit=int(env_cfg.get('episode_limit', 1000)),
        agent_obsk=int(env_cfg.get('agent_obsk', 1)),
        h_kappa=float(env_cfg.get('h_kappa', 0.0)),
        h_kappa_z=float(env_cfg.get('h_kappa_z', 0.0)),
        upright_scale=float(env_cfg.get('upright_scale', 0.2)),
        safety_provider=provider,
        h_scale=float(env_cfg.get('h_scale', 0.5)),
    )
    max_step = int(env.max_step)
    n = int(env.num_agents)
    obs_dim = int(env.observation_dim)
    obs_data = np.zeros(
        (args.episodes, max_step, n, obs_dim), dtype=np.float32)
    reward_data = np.zeros(
        (args.episodes, max_step, n), dtype=np.float32)
    cost_data = np.zeros((args.episodes, max_step), dtype=np.float32)
    valid_data = np.zeros((args.episodes, max_step), dtype=np.float32)
    seeds = np.zeros((args.episodes,), dtype=np.int64)
    lengths = np.zeros((args.episodes,), dtype=np.int64)

    for episode in range(int(args.episodes)):
        ep_seed = int(args.seed) + episode
        seeds[episode] = ep_seed
        obs, h, grad, _hvec, _hs, _geom, _state = env.reset(seed=ep_seed)
        ep_len = 0
        for t in range(max_step):
            task = None
            if task_feature_fn is not None:
                task = task_feature_fn(obs)
            actions, _ = model.act_single(
                obs, h, grad, task,
                deterministic=bool(args.deterministic), device=device)
            obs_data[episode, t] = obs
            (obs, h, grad, _hvec, _hs, _geom, _state,
             reward, cost, terminated, truncated, _info) = env.step(actions)
            reward_data[episode, t, :] = float(reward)
            cost_data[episode, t] = float(cost)
            valid_data[episode, t] = 1.0
            ep_len += 1
            if terminated or truncated:
                break
        lengths[episode] = ep_len
        if (episode + 1) % 10 == 0 or episode == 0:
            print(f'collected {episode + 1}/{args.episodes} episodes',
                  flush=True)

    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=obs_data,
        task_reward=reward_data,
        cost=cost_data,
        valid=valid_data,
        seeds=seeds,
        episode_lengths=lengths,
    )
    meta = dict(
        schema='v6.ant_task_value.data.v1',
        checkpoint=str(ckpt_path),
        h_checkpoint=str(h_path) if h_path else None,
        episodes=int(args.episodes),
        seed=int(args.seed),
        num_agents=n,
        obs_dim=obs_dim,
        max_step=max_step,
        stochastic=not bool(args.deterministic),
        reward='official Ant task reward; ungated; no cost',
        out=str(out),
        mean_episode_length=float(lengths.mean()),
    )
    (out.parent / 'collect_summary.json').write_text(
        json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2), flush=True)
    env.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
