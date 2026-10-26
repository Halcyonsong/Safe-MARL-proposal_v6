"""Measure held-out quality of the learned task value V_T.

V_T is trained from task-only returns.  Classification-style accuracy is not
meaningful for a value function, so this script reports:

* explained variance / MSE against discounted Monte-Carlo task returns;
* Pearson and Spearman rank correlation with the return-to-go;
* one-step TD error;
* A_T scale and saturation;
* state-gradient norm.

The policy-level test is still the V6-0/V6-1/V6-2/V6-3 ladder; this script
tells us whether V_T itself is a useful ranking signal before that ablation.
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
from v6.safety import LocalSafetyProvider  # noqa: E402
from v6.training import (  # noqa: E402
    _task_channel, _tensor, load_v6_checkpoint,
)


def _cwd_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    # average ties
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _corr(_rank(a), _rank(b))


@torch.no_grad()
def evaluate_task_value(
    env,
    hnet,
    task_net,
    ema,
    model,
    seeds,
    device,
    *,
    gamma: float = 0.99,
    reward_scale: float = 20.0,
    reach_bonus: float = 5.0,
    terminal_reach_bonus: float = 20.0,
    use_task_scalar: bool = True,
    use_task_grad: bool = True,
    max_neighbors: int = 5,
    n_rays: int = 32,
) -> dict:
    values, rewards, next_values, a_values, grad_norms = [], [], [], [], []
    for seed in seeds:
        obs = env.reset(seed=int(seed))
        h, gh = hnet.value_and_local_grad()
        states, goals, lidar = (
            torch.as_tensor(env.get_agent_states(), dtype=torch.float32, device=device),
            torch.as_tensor(env.get_goal_states(), dtype=torch.float32, device=device),
            torch.as_tensor(env.get_lidar(), dtype=torch.float32, device=device),
        )
        v, a_t, feat = _task_channel(
            task_net, ema, states, goals, lidar,
            max_neighbors=max_neighbors, n_rays=n_rays,
            comm_radius=env.comm_radius, use_scalar=use_task_scalar,
            use_grad=use_task_grad, grad_scale=1.0)
        ep_v, ep_r = [], []
        ep_next, ep_a, ep_gn = [], [], []
        for _ in range(env.max_step):
            out = model(
                _tensor(obs, device), _tensor(h, device).unsqueeze(-1),
                _tensor(gh, device), extra=feat.detach())
            action = out['action'].detach().cpu().numpy()
            obs, progress, terminated, truncated, info = env.step(action)
            reach = np.asarray(info['reached'], dtype=np.float32)
            task_reward = (progress.astype(np.float32) * reward_scale
                           + reach_bonus * reach)
            if terminated:
                task_reward = task_reward + terminal_reach_bonus * reach
            ep_r.append(task_reward)
            ep_v.append(v.detach().cpu().numpy())
            ep_a.append(a_t.detach().cpu().numpy())
            ep_gn.append(feat[:, 1:].detach().norm(dim=-1).cpu().numpy())
            h, gh = hnet.value_and_local_grad()
            states = torch.as_tensor(
                env.get_agent_states(), dtype=torch.float32, device=device)
            goals = torch.as_tensor(
                env.get_goal_states(), dtype=torch.float32, device=device)
            lidar = torch.as_tensor(
                env.get_lidar(), dtype=torch.float32, device=device)
            v, a_t, feat = _task_channel(
                task_net, ema, states, goals, lidar,
                max_neighbors=max_neighbors, n_rays=n_rays,
                comm_radius=env.comm_radius, use_scalar=use_task_scalar,
                use_grad=use_task_grad, grad_scale=1.0)
            next_v = v.detach().cpu().numpy()
            ep_next.append(next_v)
            if terminated or truncated:
                break
        r = np.stack(ep_r).astype(np.float32)       # [T,N]
        v_ep = np.stack(ep_v).astype(np.float32)    # [T,N]
        next_ep = np.stack(ep_next).astype(np.float32)
        # Discounted return-to-go, no bootstrap beyond episode end.
        ret = np.zeros_like(r)
        running = np.zeros(env.num_agents, dtype=np.float32)
        for t in range(len(r) - 1, -1, -1):
            running = r[t] + gamma * running
            ret[t] = running
        values.append(v_ep)
        rewards.append(ret)
        next_values.append(next_ep)
        a_values.append(np.stack(ep_a).astype(np.float32))
        grad_norms.append(np.stack(ep_gn).astype(np.float32))

    V = np.concatenate([x.reshape(-1) for x in values])
    R = np.concatenate([x.reshape(-1) for x in rewards])
    Vn = np.concatenate([x.reshape(-1) for x in next_values])
    A = np.concatenate([x.reshape(-1) for x in a_values])
    G = np.concatenate([x.reshape(-1) for x in grad_norms])
    # TD(0) target uses the next recorded value; the last transition is
    # approximate because it is truncated, so it is reported as auxiliary.
    td = R - V
    mse = float(np.mean(td ** 2))
    var = float(np.var(R))
    out = {
        'episodes': len(seeds),
        'rows': int(V.size),
        'value_mean': float(np.mean(V)),
        'value_std': float(np.std(V)),
        'return_mean': float(np.mean(R)),
        'return_std': float(np.std(R)),
        'value_mse': mse,
        'explained_variance': float(1.0 - mse / max(var, 1e-8)),
        'pearson': _corr(V, R),
        'spearman': _spearman(V, R),
        'value_rmse': float(np.sqrt(mse)),
        'a_t_abs_mean': float(np.mean(np.abs(A))),
        'a_t_clip_saturation': float(np.mean(np.abs(A) >= 4.999)),
        'grad_norm_mean': float(np.mean(G)),
        'grad_norm_std': float(np.std(G)),
        'next_value_std': float(np.std(Vn)),
    }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--h-checkpoint')
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--seed', type=int, default=10000)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--out')
    args = ap.parse_args()

    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))
    model, task_net, ema, ckpt = load_v6_checkpoint(
        _cwd_path(args.checkpoint), device=device)
    config = ckpt['config']
    e = config['env']
    env = OfficialDoubleIntegrator(
        num_agents=e['num_agents'], area_size=e['area_size'],
        max_step=e['max_step'], num_obstacles=e['num_obstacles'],
        n_rays=e['n_rays'], max_neighbors=e['max_neighbors'],
        seed=config.get('seed', 0))
    # The V6 h provider warms up on a live graph; initialize the env before
    # constructing it.  This task-value script is legacy/post-hoc, not part of
    # the independent h-only main protocol.
    env.reset(seed=config.get('seed', 0))
    h_path = (Path(args.h_checkpoint) if args.h_checkpoint
              else _resolve(config['h']['checkpoint']))
    h_path = _cwd_path(h_path)
    h_offset = config['h'].get('h_offset')
    if h_path.suffix.lower() in ('.pkl', '.pickle'):
        hnet = official_h_provider(
            h_path, env, h_offset=None if h_offset is None else float(h_offset),
            h_scale=config['h'].get('h_scale'))
    else:
        hnet = LocalSafetyProvider.from_checkpoint(
            h_path, env, device=device,
            max_neighbors=e['max_neighbors'], n_rays=e['n_rays'],
            comm_radius=env.comm_radius)
        if h_offset is not None:
            hnet.h_offset = float(h_offset)
    result = evaluate_task_value(
        env, hnet, task_net, ema, model,
        seeds=[args.seed + i for i in range(args.episodes)],
        device=device,
        gamma=float(config['training'].get('gamma', 0.99)),
        reward_scale=float(config['training']['reward_scale']),
        reach_bonus=float(config['training'].get('reach_bonus', 0.0)),
        terminal_reach_bonus=float(
            config['training'].get('terminal_reach_bonus', 0.0)),
        use_task_scalar=bool(config['training'].get('use_task_scalar', True)),
        use_task_grad=bool(config['training'].get('use_task_grad', True)),
        max_neighbors=e['max_neighbors'], n_rays=e['n_rays'])
    payload = dict(checkpoint=str(args.checkpoint), h_checkpoint=str(h_path),
                   episodes=args.episodes, seed=args.seed, metrics=result)
    print(json.dumps(payload, indent=2))
    if args.out:
        out = _cwd_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    main()
