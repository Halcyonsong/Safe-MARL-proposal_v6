"""Train the V6 finite-horizon task value V_T from actor1 rollouts.

V_T target: K-step ungated task return.  No h gate, no safety cost, no PPO
critic role.  The saved checkpoint is consumed by the actor warm-start as a
frozen 5-D task feature [A_T, grad_s V_T].

This script is offline-only: it never updates the actor or h.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from v6.task_value import (  # noqa: E402
    TaskAdvantageEMA, build_task_value_net)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _device(name: str) -> torch.device:
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    err = pred - target
    mse = float(np.mean(err ** 2))
    var = float(np.var(target))
    return dict(
        rows=int(pred.size),
        mse=mse,
        rmse=float(math.sqrt(mse)),
        target_mean=float(np.mean(target)),
        target_std=float(np.std(target)),
        pred_mean=float(np.mean(pred)),
        pred_std=float(np.std(pred)),
        explained_variance=float(1.0 - mse / max(var, 1e-12)),
        pearson=_corr(pred, target),
        spearman=_corr(_rank(pred), _rank(target)),
        constant_baseline_mse=float(np.mean((target - np.mean(target)) ** 2)),
    )


def _finite_horizon_targets(
    rewards: np.ndarray, horizon: int, gamma: float,
) -> np.ndarray:
    """Return [E,T,N] K-step discounted returns.

    The environment only times out at max_step, so every episode has the same
    length.  Windows near the end are naturally shorter (return to timeout).
    """
    e, t, n = rewards.shape
    target = np.zeros((e, t, n), dtype=np.float32)
    for k in range(min(int(horizon), t)):
        target[:, :t - k, :] += (gamma ** k) * rewards[:, k:, :]
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', required=True)
    ap.add_argument('--config',
                    default='nav/configs/v6/v6_nav_stageb_repair_16k.yaml')
    ap.add_argument('--out', required=True)
    ap.add_argument('--horizon', type=int, default=64)
    ap.add_argument('--gamma', type=float, default=0.99)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--max-grad-norm', type=float, default=1.0)
    ap.add_argument('--train-frac', type=float, default=0.8)
    ap.add_argument('--val-frac', type=float, default=0.1)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = _device(args.device)

    data = np.load(_resolve(args.data))
    obs = np.asarray(data['obs'], dtype=np.float32)
    rewards = np.asarray(data['task_reward'], dtype=np.float32)
    seeds = np.asarray(data['seeds'])
    e, t, n, obs_dim = obs.shape
    if rewards.shape != (e, t, n):
        raise ValueError(f'reward shape {rewards.shape} != {(e, t, n)}')
    target = _finite_horizon_targets(rewards, args.horizon, args.gamma)

    if e < 3:
        raise ValueError('need at least 3 episodes for train/val/test')
    n_train = min(max(1, int(round(e * args.train_frac))), e - 2)
    n_val = min(max(1, int(round(e * args.val_frac))), e - n_train - 1)
    n_test = e - n_train - n_val
    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, e)

    def _flat(sl):
        return (obs[sl].reshape(-1, obs_dim),
                target[sl].reshape(-1),
                sl)

    obs_train, target_train, _ = _flat(train_slice)
    obs_val, target_val, _ = _flat(val_slice)
    obs_test, target_test, _ = _flat(test_slice)

    config = {}
    cfg_path = _resolve(args.config)
    if cfg_path.exists():
        import yaml
        full = yaml.safe_load(cfg_path.read_text())
        config = dict(full.get('task_value', {}))
    config.setdefault('arch', 'graph')
    config.setdefault('graph_hidden', 128)
    config.setdefault('graph_layers', 2)
    task_net = build_task_value_net(
        config, obs_dim,
        max_neighbors=int(config.get('max_neighbors', 5)),
        n_rays=int(config.get('n_rays', 32))).to(device)
    optimizer = torch.optim.AdamW(
        task_net.parameters(), lr=float(args.lr),
        weight_decay=float(args.weight_decay))

    train_ds = TensorDataset(
        torch.from_numpy(obs_train).float(),
        torch.from_numpy(target_train).float())
    train_loader = DataLoader(
        train_ds, batch_size=int(args.batch_size), shuffle=True,
        drop_last=False)
    val_obs_t = torch.from_numpy(obs_val).float().to(device)
    val_target_t = torch.from_numpy(target_val).float().to(device)

    def _eval_mse() -> float:
        task_net.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, val_obs_t.shape[0], int(args.batch_size)):
                xb = val_obs_t[start:start + int(args.batch_size)]
                yb = val_target_t[start:start + int(args.batch_size)]
                pred = task_net(xb)
                total += float(((pred - yb) ** 2).sum())
                count += int(yb.numel())
        task_net.train()
        return total / max(count, 1)

    best_val = float('inf')
    best_state = None
    best_epoch = -1
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        task_net.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = task_net(xb)
            loss = 0.5 * ((pred - yb) ** 2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                task_net.parameters(), float(args.max_grad_norm))
            optimizer.step()
            total += float(((pred - yb) ** 2).sum())
            count += int(yb.numel())
        train_mse = total / max(count, 1)
        val_mse = _eval_mse()
        history.append(dict(epoch=epoch, train_mse=train_mse,
                            val_mse=val_mse))
        print(f'epoch={epoch} train_mse={train_mse:.6f} '
              f'val_mse={val_mse:.6f}', flush=True)
        if val_mse < best_val - 1e-9:
            best_val = val_mse
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in task_net.state_dict().items()}
        elif epoch - best_epoch >= int(args.patience):
            print(f'early stop at epoch {epoch}; best={best_epoch}',
                  flush=True)
            break

    if best_state is not None:
        task_net.load_state_dict(best_state)
    task_net.eval()

    def _predict(obs_arr):
        out = []
        with torch.no_grad():
            for start in range(0, obs_arr.shape[0], 1024):
                xb = torch.from_numpy(obs_arr[start:start + 1024]).float().to(device)
                out.append(task_net(xb).detach().cpu().numpy())
        return np.concatenate(out)

    pred_train = _predict(obs_train)
    pred_val = _predict(obs_val)
    pred_test = _predict(obs_test)
    metrics = dict(
        train=_metrics(pred_train, target_train),
        val=_metrics(pred_val, target_val),
        test=_metrics(pred_test, target_test),
        best_epoch=best_epoch,
        best_val_mse=best_val,
        history=history,
    )

    # Per-agent EMA from train+val predictions.  The actor warm-start freezes
    # this baseline together with V_T.
    obs_ema = np.concatenate([obs_train, obs_val], axis=0)
    pred_ema = _predict(obs_ema)
    pred_ema_3d = pred_ema.reshape(-1, n)
    ema = TaskAdvantageEMA.zeros(
        n, warmup_count=1024.0, clip=5.0, decay=0.99, var_floor=1.0)
    ema.mean.copy_(torch.from_numpy(pred_ema_3d.mean(axis=0)).float())
    ema.var.copy_(torch.from_numpy(
        pred_ema_3d.var(axis=0)).float().clamp_min(1.0))
    ema.count = 1.0e6

    out = _resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        schema='v6.task_value.finite_horizon.v1',
        task_net=task_net.state_dict(),
        task_ema=ema.state_dict(),
        config=config,
        obs_dim=int(obs_dim),
        extra_dim=5,
        target=dict(
            kind='finite_horizon_task_return',
            horizon=int(args.horizon),
            gamma=float(args.gamma),
            reward='ungated_task: reward_scale*progress + reach_bonus*reached + terminal_reach_bonus*reached_at_timeout',
            uses_h_gate=False,
            uses_safety_cost=False,
        ),
        metrics=metrics,
        data=dict(
            path=str(_resolve(args.data)),
            episodes=int(e),
            steps_per_episode=int(t),
            num_agents=int(n),
            seeds=[int(x) for x in seeds],
            train_episodes=[int(x) for x in seeds[train_slice]],
            val_episodes=[int(x) for x in seeds[val_slice]],
            test_episodes=[int(x) for x in seeds[test_slice]],
        ),
        seed=int(args.seed),
    )
    torch.save(payload, out)
    ema_state = ema.state_dict()
    ema_json = {
        k: (v.tolist() if hasattr(v, 'tolist') else v)
        for k, v in ema_state.items()
    }
    summary = dict(
        out=str(out),
        schema=payload['schema'],
        target=payload['target'],
        metrics=metrics,
        data=payload['data'],
        ema=ema_json,
    )
    (out.parent / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    print(f'saved: {out}')
    print(json.dumps(dict(
        val=metrics['val'], test=metrics['test'],
        best_epoch=best_epoch), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
