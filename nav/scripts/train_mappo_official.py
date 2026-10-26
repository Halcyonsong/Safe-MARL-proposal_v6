"""Train pure MAPPO on the official GCBF+ DoubleIntegrator.

The actor uses local observations only. The critic is centralized over all
agents/goals/obstacles. There is no h(s), no correction head, no cost critic, and
no QP in this baseline.
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

import numpy as np
import torch
import yaml

from algos.baselines import (
    add_gae,
    build_mappo_model,
    collect_rollout,
    evaluate,
    global_state_dim,
    ppo_update,
    save_mappo_checkpoint,
)
from envs.gcbfplus import OfficialDoubleIntegrator


def _cwd_path(path):
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def _device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        default='configs/baselines/mappo_double_integrator_pure.yaml')
    parser.add_argument(
        '--out',
        default=str(ROOT / 'outputs/v5/baselines/mappo_pure'))
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

    env_config = config['env']
    env = OfficialDoubleIntegrator(
        num_agents=env_config['num_agents'],
        area_size=env_config['area_size'],
        max_step=env_config['max_step'],
        num_obstacles=env_config['num_obstacles'],
        n_rays=env_config['n_rays'],
        max_neighbors=env_config['max_neighbors'],
        seed=seed,
    )
    env.reset(seed=seed)
    obs_dim = env.observation_dim
    state_dim = global_state_dim(env)
    model = build_mappo_model(obs_dim, state_dim, config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config['training']['lr']))

    total_steps = int(config['training']['total_steps'])
    rollout_steps = int(config['training']['rollout_steps'])
    eval_interval = int(config['training']['eval_interval'])
    eval_episodes = int(config['training']['eval_episodes'])
    eval_seed = int(config['training'].get('eval_seed', 20000))
    save_interval = int(config['training']['save_interval'])
    reward_scale = float(config['training']['reward_scale'])
    reach_bonus = float(config['training']['reach_bonus'])

    steps = 0
    best_score = -float('inf')
    best_step = None
    if args.resume:
        saved = torch.load(_cwd_path(args.resume), map_location='cpu',
                           weights_only=False)
        model.load_state_dict(saved['model'])
        steps = int(saved.get('steps', 0))
        print(f'resumed from {args.resume} at steps={steps}', flush=True)

    history = []
    eval_history = []
    print(
        f'Pure MAPPO: device={device} obs_dim={obs_dim} state_dim={state_dim} '
        f'total_steps={total_steps} rollout_steps={rollout_steps}', flush=True)

    while steps < total_steps:
        rollout = collect_rollout(
            env, model, rollout_steps, seed + steps, device, config)
        add_gae(
            rollout, config['training']['gamma'],
            config['training']['gae_lambda'])
        update = ppo_update(model, optimizer, rollout, config, device)
        steps += rollout_steps

        recent = rollout.completed[-eval_episodes:] if rollout.completed else []
        completed = ({key: float(np.mean([row[key] for row in recent]))
                      for key in recent[0]} if recent else {})
        row = dict(
            steps=steps,
            update=update,
            completed=completed,
            reward_mean=float(rollout.rewards.mean()),
            action_abs=float(rollout.actions.abs().mean()),
        )
        history.append(row)
        with open(out / 'train.jsonl', 'a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(f'steps={steps} {row}', flush=True)

        if eval_interval and steps % eval_interval == 0:
            result = evaluate(
                env, model,
                seeds=[eval_seed + i for i in range(eval_episodes)],
                device=device, reward_scale=reward_scale,
                reach_bonus=reach_bonus)
            eval_history.append(dict(steps=steps, **result))
            with open(out / 'eval.jsonl', 'a') as stream:
                stream.write(json.dumps(dict(steps=steps, **result)) + '\n')
            # Independent validation selection: safe-and-reached is primary,
            # with safety and return as tie-breakers.
            score = (result['official_success']
                     + 0.1 * result['official_safety']
                     + 0.001 * result['episode_return'])
            if score > best_score:
                best_score = score
                best_step = steps
                save_mappo_checkpoint(
                    out / 'policy_best.pth', model, config, steps,
                    obs_dim, state_dim)
            print(f'validation steps={steps} {result}', flush=True)

        if save_interval and steps % save_interval == 0:
            save_mappo_checkpoint(
                out / f'policy_step_{steps}.pth', model, config, steps,
                obs_dim, state_dim)

    save_mappo_checkpoint(
        out / 'policy_final.pth', model, config, steps, obs_dim, state_dim)
    if args.skip_final_eval:
        result = {}
    else:
        result = evaluate(
            env, model, seeds=[eval_seed + i for i in range(eval_episodes)],
            device=device, reward_scale=reward_scale, reach_bonus=reach_bonus)
        eval_history.append(dict(steps=steps, **result))
        score = (result['official_success']
                 + 0.1 * result['official_safety']
                 + 0.001 * result['episode_return'])
        if score > best_score:
            best_score = score
            best_step = steps
            save_mappo_checkpoint(
                out / 'policy_best.pth', model, config, steps, obs_dim, state_dim)
        print(f'final validation: {result}', flush=True)

    (out / 'summary.json').write_text(json.dumps(dict(
        steps=steps, seed=seed, obs_dim=obs_dim, state_dim=state_dim,
        device=str(device), eval=result, best_step=best_step,
        best_score=best_score if best_step else None,
        history=history, eval_history=eval_history), indent=2) + '\n')
    print(f'saved: {out}', flush=True)


if __name__ == '__main__':
    main()
