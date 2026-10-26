"""Collect actor1 rollouts for the V6 finite-horizon task value V_T.

This is data collection only: no actor update, no h update, no task-channel
update, no gate/cost change.  The saved task reward is the *ungated* task
reward, so V_T does not depend on h or the safety cost.

Output arrays (episode-major):
  obs            [E, T, N, 92]  base observation before each action
  task_reward    [E, T, N]      ungated task reward
  done           [E, T]         episode boundary (timeout)
  reached        [E, T, N]
  collision      [E, T, N]
  agent_states   [E, T, N, 4]
  goals          [E, T, N, 4]
  lidar          [E, T, N, 32, 2]
  seeds          [E]
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

from envs.gcbfplus import OfficialDoubleIntegrator  # noqa: E402
from v6.official_h import provider_from_checkpoint as official_h_provider  # noqa: E402
from v6.provenance import assert_independent_path  # noqa: E402
from v6.safety import LocalSafetyProvider  # noqa: E402
from v6.training import load_v6_checkpoint  # noqa: E402


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True,
                    help='fixed actor1 checkpoint')
    ap.add_argument('--h-checkpoint',
                    help='frozen h1 checkpoint; defaults to checkpoint config')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--seed', type=int, default=30000)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    device = _device(args.device)
    ckpt_path = assert_independent_path(
        _cwd_path(args.checkpoint), kind='V6 policy checkpoint')
    model, _task_net, _ema, ckpt = load_v6_checkpoint(ckpt_path, device=device)
    config = ckpt['config']
    env_cfg = config['env']
    env = OfficialDoubleIntegrator(
        num_agents=env_cfg['num_agents'],
        area_size=env_cfg['area_size'],
        max_step=env_cfg['max_step'],
        num_obstacles=env_cfg['num_obstacles'],
        n_rays=env_cfg['n_rays'],
        max_neighbors=env_cfg['max_neighbors'],
        seed=int(config.get('seed', 0)),
    )
    env.reset(seed=int(config.get('seed', 0)))

    h_path = _cwd_path(args.h_checkpoint or config['h']['checkpoint'])
    h_path = assert_independent_path(h_path, kind='V6 evaluation h')
    h_offset = config['h'].get('h_offset')
    if h_path.suffix.lower() in ('.pkl', '.pickle'):
        hnet = official_h_provider(
            h_path, env,
            h_offset=None if h_offset is None else float(h_offset),
            h_scale=config['h'].get('h_scale'))
    else:
        hnet = LocalSafetyProvider.from_checkpoint(
            h_path, env, device=device,
            max_neighbors=env_cfg['max_neighbors'],
            n_rays=env_cfg['n_rays'],
            comm_radius=env.comm_radius)
        if h_offset is not None:
            hnet.h_offset = float(h_offset)

    reward_scale = float(config['training']['reward_scale'])
    reach_bonus = float(config['training'].get('reach_bonus', 0.0))
    terminal_reach_bonus = float(
        config['training'].get('terminal_reach_bonus', 0.0))

    obs_rows, reward_rows, done_rows = [], [], []
    reached_rows, collision_rows = [], []
    state_rows, goal_rows, lidar_rows = [], [], []
    seeds = []
    model.eval()
    n = env.num_agents
    extra = torch.zeros(n, 5, device=device)

    for ep in range(int(args.episodes)):
        seed = int(args.seed) + ep
        seeds.append(seed)
        obs = env.reset(seed=seed)
        h, gh = hnet.value_and_local_grad()
        ep_obs, ep_reward, ep_done = [], [], []
        ep_reached, ep_collision = [], []
        ep_state, ep_goal, ep_lidar = [], [], []
        for _ in range(env.max_step):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            h_t = torch.as_tensor(h, dtype=torch.float32,
                                  device=device).unsqueeze(-1)
            gh_t = torch.as_tensor(gh, dtype=torch.float32, device=device)
            with torch.no_grad():
                out = model.sample(obs_t, h_t, gh_t, extra=extra)
                action = out['env_action'].detach().cpu().numpy()
            ep_obs.append(np.asarray(obs, dtype=np.float32).copy())
            ep_state.append(env.get_agent_states().copy())
            ep_goal.append(env.get_goal_states().copy())
            ep_lidar.append(env.get_lidar().copy())

            obs, progress, terminated, truncated, info = env.step(action)
            reach = np.asarray(info['reached'], dtype=np.float32)
            collision = np.asarray(info['collision'], dtype=np.float32)
            task_reward = progress.astype(np.float32) * reward_scale
            task_reward = task_reward + reach_bonus * reach
            if terminated:
                task_reward = task_reward + terminal_reach_bonus * reach
            ep_reward.append(task_reward.astype(np.float32))
            ep_reached.append(reach)
            ep_collision.append(collision)
            ep_done.append(bool(terminated or truncated))
            h, gh = hnet.value_and_local_grad()
            if terminated or truncated:
                break
        obs_rows.append(np.stack(ep_obs))
        reward_rows.append(np.stack(ep_reward))
        done_rows.append(np.asarray(ep_done, dtype=bool))
        reached_rows.append(np.stack(ep_reached))
        collision_rows.append(np.stack(ep_collision))
        state_rows.append(np.stack(ep_state))
        goal_rows.append(np.stack(ep_goal))
        lidar_rows.append(np.stack(ep_lidar))
        if (ep + 1) % 10 == 0 or ep == 0 or ep + 1 == int(args.episodes):
            print(f'collected {ep + 1}/{args.episodes} episodes', flush=True)

    out = _cwd_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=np.stack(obs_rows),
        task_reward=np.stack(reward_rows),
        done=np.stack(done_rows),
        reached=np.stack(reached_rows),
        collision=np.stack(collision_rows),
        agent_states=np.stack(state_rows),
        goals=np.stack(goal_rows),
        lidar=np.stack(lidar_rows),
        seeds=np.asarray(seeds, dtype=np.int64),
    )
    manifest = dict(
        checkpoint=str(ckpt_path),
        h_checkpoint=str(h_path),
        episodes=int(args.episodes),
        seed=int(args.seed),
        num_agents=int(n),
        max_step=int(env.max_step),
        reward_scale=reward_scale,
        reach_bonus=reach_bonus,
        terminal_reach_bonus=terminal_reach_bonus,
        stochastic=True,
        task_reward='ungated: reward_scale*progress + reach_bonus*reached + terminal_reach_bonus*reached_at_timeout',
        out=str(out),
    )
    (out.parent / 'manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n')
    print(f'saved: {out}')
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
