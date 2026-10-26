"""Evaluate a V6 from-scratch Ant MAPPO checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from v6.ant_env import Ant2x4Env  # noqa: E402
from v6.ant_h import AntSafetyProvider  # noqa: E402
from v6.ant_mappo import (  # noqa: E402
    evaluate_ant_mappo, load_ant_mappo_model)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--h-checkpoint')
    ap.add_argument('--out')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--seed', type=int, default=20000)
    ap.add_argument('--episode-limit', type=int)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--stochastic', action='store_true',
                    help='sample from the policy instead of using its mean')
    ap.add_argument('--task-value-checkpoint',
                    help='frozen Ant V_T checkpoint for a V_T actor')
    args = ap.parse_args()

    ckpt_path = _resolve(args.checkpoint)
    assert_independent_path(ckpt_path, kind='V6 Ant MAPPO checkpoint')
    device = _device(args.device)
    model, ckpt = load_ant_mappo_model(ckpt_path, device=device)
    cfg = dict(ckpt.get('config', {}))
    env_cfg = dict(cfg.get('env_cfg', {}))
    use_h = bool(model.use_h_features)
    task_value_path = (
        args.task_value_checkpoint
        or (cfg.get('task_value') or {}).get('checkpoint'))
    task_feature_fn = None
    if model.use_task_value:
        if not task_value_path:
            raise ValueError(
                'checkpoint uses a V_T actor input but no task-value '
                'checkpoint was supplied')
        task_value_path = _resolve(task_value_path)
        assert_independent_path(
            task_value_path, kind='V6 Ant task-value checkpoint')
        task_value = load_ant_task_value(task_value_path, device=device)
        task_feature_fn = make_task_feature_fn(task_value, device=device)
    elif task_value_path:
        # A checkpoint without a V_T input must not silently ignore the flag.
        raise ValueError(
            '--task-value-checkpoint was supplied for a checkpoint whose '
            'actor has no task-value input')
    else:
        task_value_path = None
    h_path = args.h_checkpoint or cfg.get('h_checkpoint')
    if use_h:
        if not h_path:
            raise ValueError('checkpoint uses h features but no h checkpoint '
                             'was supplied')
        h_path = _resolve(h_path)
        assert_independent_path(h_path, kind='V6 Ant h checkpoint')
        provider = AntSafetyProvider.from_checkpoint(h_path, device=device)
    else:
        provider = NullAntSafetyProvider()
        h_path = None

    env = Ant2x4Env(
        seed=int(args.seed),
        episode_limit=int(
            args.episode_limit if args.episode_limit is not None
            else env_cfg.get('episode_limit', 1000)),
        agent_obsk=int(env_cfg.get('agent_obsk', 1)),
        h_kappa=float(env_cfg.get('h_kappa', 0.0)),
        h_kappa_z=float(env_cfg.get('h_kappa_z', 0.0)),
        upright_scale=float(env_cfg.get('upright_scale', 0.2)),
        safety_provider=provider,
        h_scale=float(env_cfg.get('h_scale', 0.5)),
    )
    metrics = evaluate_ant_mappo(
        env, model, episodes=int(args.episodes), seed=int(args.seed),
        device=device, deterministic=not bool(args.stochastic),
        task_feature_fn=task_feature_fn)
    result = dict(
        schema='v6.ant_mappo.eval.v1',
        checkpoint=str(ckpt_path),
        h_checkpoint=str(h_path) if h_path else None,
        task_value_checkpoint=(
            str(task_value_path) if task_value_path else None),
        use_h_features=bool(use_h),
        use_task_value=bool(model.use_task_value),
        stochastic=bool(args.stochastic),
        episodes=int(args.episodes),
        seed=int(args.seed),
        env=env.get_env_info(),
        metrics=metrics,
    )
    if args.out:
        out = _resolve(args.out)
    else:
        mode = 'stochastic' if args.stochastic else 'deterministic'
        out = ckpt_path.parent / (
            f'eval_{args.episodes}ep_{mode}_seed{args.seed}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    env.close()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
