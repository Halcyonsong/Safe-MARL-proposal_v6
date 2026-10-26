"""LEGACY V5 entry point: train the V5 QP-free policy.

This script is retained as historical reference and is not part of the
standalone V6 main path.  It uses the official h(s) frozen and must not be used
to produce V6 h/actor artifacts.  V6 uses ``scripts/train_v6.py`` with a
V6-trained h and no official h/QP/u_ref.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``algos``/``envs`` importable when this file is executed directly.
NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

import numpy as np
import torch
import yaml

from algos.safe_policy import OfficialFixedH, QpFreePolicy
from algos.safe_policy.training import (
    add_gae, collect_rollout, evaluate, ppo_update, save_checkpoint)
from envs.gcbfplus import OfficialDoubleIntegrator


def _resolve(path):
    """Resolve repo-relative assets such as the h checkpoint."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _cwd_path(path):
    """Resolve user CLI paths relative to the current working directory."""
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def _device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/v5/qp_free_double_integrator.yaml')
    parser.add_argument('--out', default=str(ROOT / 'outputs/v5/qp_free_double_integrator'))
    parser.add_argument('--total-steps', type=int)
    parser.add_argument('--rollout-steps', type=int)
    parser.add_argument('--eval-interval', type=int)
    parser.add_argument('--eval-episodes', type=int)
    parser.add_argument('--save-interval', type=int)
    parser.add_argument('--skip-final-eval', action='store_true')
    parser.add_argument('--resume')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = NAV / config_path
    with open(config_path) as stream:
        config = yaml.safe_load(stream)
    if args.total_steps is not None:
        config['training']['total_steps'] = args.total_steps
    if args.rollout_steps is not None:
        config['training']['rollout_steps'] = args.rollout_steps
    if args.eval_interval is not None:
        config['training']['eval_interval'] = args.eval_interval
    if args.eval_episodes is not None:
        config['training']['eval_episodes'] = args.eval_episodes
    if args.save_interval is not None:
        config['training']['save_interval'] = args.save_interval
    if args.seed is not None:
        config['seed'] = args.seed
    if args.device is not None:
        config['training']['device'] = args.device
    seed = int(config['seed'])
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device(config['training']['device'])
    out = _cwd_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))

    env = OfficialDoubleIntegrator(
        num_agents=config['env']['num_agents'],
        area_size=config['env']['area_size'],
        max_step=config['env']['max_step'],
        num_obstacles=config['env']['num_obstacles'],
        n_rays=config['env']['n_rays'],
        max_neighbors=config['env']['max_neighbors'],
        seed=seed,
    )
    obs_dim = env.observation_dim
    env.reset(seed=seed)
    hnet = OfficialFixedH(
        _resolve(config['h']['checkpoint']), env,
        n_agents=config['env']['num_agents'],
        gnn_layers=config['h']['gnn_layers'])
    model = QpFreePolicy(
        obs_dim, hidden=config['policy']['hidden'],
        correction_radius=config['policy']['correction_radius'],
        initial_log_std=config['policy']['initial_log_std']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    total_steps = int(config['training']['total_steps'])
    rollout_steps = int(config['training']['rollout_steps'])
    eval_interval = int(config['training']['eval_interval'])
    save_interval = int(config['training']['save_interval'])
    eval_episodes = int(config['training']['eval_episodes'])
    reward_scale = float(config['training']['reward_scale'])
    steps = 0
    best_score = -float('inf')
    best_step = None
    if args.resume:
        resume = torch.load(_cwd_path(args.resume), map_location='cpu',
                            weights_only=False)
        model.load_state_dict(resume['model'])
        steps = int(resume.get('steps', 0))
        print(f'resumed from {args.resume} at steps={steps}', flush=True)
    history = []
    eval_history = []
    print(f'V5 QP-free training: device={device} obs_dim={obs_dim} '
          f'total_steps={total_steps} rollout_steps={rollout_steps}', flush=True)

    while steps < total_steps:
        rollout = collect_rollout(
            env, hnet, model, rollout_steps, seed + steps, device, config)
        add_gae(rollout, config['training']['gamma'], config['training']['gae_lambda'])
        update = ppo_update(model, optimizer, rollout, config, device)
        steps += rollout_steps

        recent = rollout.completed[-eval_episodes:] if rollout.completed else []
        completed = {key: float(np.mean([row[key] for row in recent]))
                     for key in recent[0]} if recent else {}
        row = dict(steps=steps, update=update, completed=completed,
                   reward_mean=float(rollout.reward.mean()),
                   unsafe_mean=float(rollout.cost_target.mean()),
                   h_mean=float(rollout.h.mean()),
                   h_neg_rate=float((rollout.h < 0).float().mean()))
        history.append(row)
        with open(out / 'train.jsonl', 'a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(f'steps={steps} {row}', flush=True)

        if eval_interval and steps % eval_interval == 0:
            result = evaluate(
                env, hnet, model,
                seeds=[seed + 10000 + i for i in range(eval_episodes)],
                device=device, reward_scale=reward_scale,
                reach_bonus=float(config['training'].get('reach_bonus', 0.0)),
                use_h_features=bool(config.get('h', {}).get(
                    'use_features', True)))
            eval_history.append(dict(steps=steps, **result))
            with open(out / 'eval.jsonl', 'a') as stream:
                stream.write(json.dumps(dict(steps=steps, **result)) + '\n')
            # Safety-first model selection; task reach/return break ties.
            score = (result['safety']
                     + 0.1 * result.get('reach_final', 0.0)
                     + 0.001 * result['episode_return'])
            if score > best_score:
                best_score = score
                best_step = steps
                save_checkpoint(out / 'policy_best.pth', model, config, steps, obs_dim)
            print(f'eval steps={steps} {result}', flush=True)
        if save_interval and steps % save_interval == 0:
            save_checkpoint(out / f'policy_step_{steps}.pth', model, config, steps, obs_dim)

    save_checkpoint(out / 'policy_final.pth', model, config, steps, obs_dim)
    if args.skip_final_eval:
        result = {}
    else:
        result = evaluate(
            env, hnet, model,
            seeds=[seed + 10000 + i for i in range(eval_episodes)],
            device=device, reward_scale=reward_scale,
            reach_bonus=float(config['training'].get('reach_bonus', 0.0)),
            use_h_features=bool(config.get('h', {}).get(
                'use_features', True)))
        eval_history.append(dict(steps=steps, **result))
        score = (result['safety']
                 + 0.1 * result.get('reach_final', 0.0)
                 + 0.001 * result['episode_return'])
        if score > best_score:
            best_score = score
            best_step = steps
            save_checkpoint(out / 'policy_best.pth', model, config, steps, obs_dim)
        print(f'final eval: {result}', flush=True)
    (out / 'summary.json').write_text(json.dumps(dict(
        steps=steps, seed=seed, obs_dim=obs_dim, eval=result,
        best_step=best_step, best_score=best_score if best_step else None,
        history=history, eval_history=eval_history), indent=2) + '\n')
    print(f'saved: {out}', flush=True)


if __name__ == '__main__':
    main()
