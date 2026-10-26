"""Train standalone V6 navigation with a V6-learned h(s).

Main protocol: frozen V6 h + from-scratch MAPPO/PPO actor, no task channel.
Official h, QP/u_ref, expert labels, and V5 weights/data are forbidden.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from envs.gcbfplus import OfficialDoubleIntegrator  # noqa: E402
from v6.policy import QpFreePolicy  # noqa: E402
from v6.official_h import provider_from_checkpoint as official_h_provider  # noqa: E402
from v6.provenance import assert_independent_path  # noqa: E402
from v6.safety import LocalSafetyProvider  # noqa: E402
from v6.task_value import (  # noqa: E402
    TaskAdvantageEMA, build_task_value_net)
from v6.co_train import OfficialHCoTrainer  # noqa: E402
from v6.training import (  # noqa: E402
    add_gae_v6, collect_rollout_v6, evaluate_v6,
    load_v6_checkpoint, ppo_update_v6, save_v6_checkpoint,
    task_value_update)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/v6/v6_nav_independent.yaml')
    ap.add_argument('--out', default='outputs/v6/independent/policy')
    ap.add_argument('--total-steps', type=int)
    ap.add_argument('--rollout-steps', type=int)
    ap.add_argument('--eval-interval', type=int)
    ap.add_argument('--eval-episodes', type=int)
    ap.add_argument('--eval-seed', type=int,
                    help='fixed evaluation seed; defaults to seed+10000')
    ap.add_argument('--eval-stochastic', action=argparse.BooleanOptionalAction,
                    default=None,
                    help='sample actions during training-time safety eval')
    ap.add_argument('--save-interval', type=int)
    ap.add_argument('--seed', type=int)
    ap.add_argument('--device', default=None)
    ap.add_argument('--deterministic', action=argparse.BooleanOptionalAction,
                    default=None,
                    help='deterministic torch/CUDA settings for reproducibility')
    ap.add_argument('--unsafe-penalty', type=float, default=None)
    ap.add_argument('--cost-actor-coef', type=float, default=None)
    ap.add_argument('--h-barrier-coef', type=float, default=None,
                    help='dense safety cost coefficient from V6 h; 0 disables')
    ap.add_argument('--h-barrier-margin', type=float, default=None)
    ap.add_argument('--task-gate-low', type=float, default=None,
                    help='V6 h gate low threshold for positive task reward')
    ap.add_argument('--task-gate-high', type=float, default=None,
                    help='V6 h gate high threshold; <= low disables the gate')
    ap.add_argument('--task-gate-positive-only',
                    action=argparse.BooleanOptionalAction,
                    default=None,
                    help='gate only positive task progress/reach reward')
    ap.add_argument('--task-gate-scope',
                    choices=('all_positive', 'progress_only'),
                    default=None,
                    help='which task components the h gate scales: all '
                         'positive task reward (first gate experiment) or '
                         'dense progress only (Ant-style Stage A)')
    ap.add_argument('--h-feature-scale', type=float, default=None,
                    help='scale h/grad_h actor features (h semantics unchanged)')
    ap.add_argument('--h-checkpoint',
                    help='V6 self-trained h checkpoint; official h and V5 '
                         'artifacts are rejected')
    ap.add_argument('--use-h-features', action=argparse.BooleanOptionalAction,
                    default=None,
                    help='use the V6 h/grad_h input channels (main arm). '
                         'Disable for the matched no-h control.')
    ap.add_argument('--use-task-scalar', action=argparse.BooleanOptionalAction,
                    default=None)
    ap.add_argument('--use-task-grad', action=argparse.BooleanOptionalAction,
                    default=None)
    ap.add_argument('--task-value-checkpoint',
                    help='V6 task-value checkpoint to load after --resume; '
                         'used by the frozen-V_T warm-start')
    ap.add_argument('--freeze-task-value',
                    action=argparse.BooleanOptionalAction, default=None,
                    help='do not update V_T or its EMA during actor training')
    ap.add_argument('--zero-init-task-block',
                    action=argparse.BooleanOptionalAction, default=None,
                    help='zero the actor first-layer columns for the 5-D '
                         'task feature after loading')
    ap.add_argument('--h-co-train', action=argparse.BooleanOptionalAction,
                    default=None,
                    help='alternate h updates with PPO/MAPPO rollouts')
    ap.add_argument('--cbf-coef', type=float, default=None,
                    help='action-space CBF loss coefficient (0 disables)')
    ap.add_argument('--cbf-alpha', type=float, default=None)
    ap.add_argument('--cbf-eps', type=float, default=None)
    ap.add_argument('--resume',
                    help='V6-owned checkpoint for continuation; not a V5 '
                         'warm-start')
    ap.add_argument('--skip-final-eval', action='store_true')
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = NAV / cfg_path
    config = yaml.safe_load(cfg_path.read_text())
    if args.total_steps is not None:
        config['training']['total_steps'] = args.total_steps
    if args.rollout_steps is not None:
        config['training']['rollout_steps'] = args.rollout_steps
    if args.eval_interval is not None:
        config['training']['eval_interval'] = args.eval_interval
    if args.eval_episodes is not None:
        config['training']['eval_episodes'] = args.eval_episodes
    if args.eval_seed is not None:
        config['training']['eval_seed'] = args.eval_seed
    if args.eval_stochastic is not None:
        config['training']['eval_stochastic'] = bool(args.eval_stochastic)
    if args.save_interval is not None:
        config['training']['save_interval'] = args.save_interval
    if args.seed is not None:
        config['seed'] = args.seed
    if args.device is not None:
        config['training']['device'] = args.device
    if args.deterministic is not None:
        config['training']['deterministic'] = bool(args.deterministic)
    if args.unsafe_penalty is not None:
        config['training']['unsafe_penalty'] = float(args.unsafe_penalty)
    if args.cost_actor_coef is not None:
        config['training']['cost_actor_coef'] = float(args.cost_actor_coef)
    if args.h_barrier_coef is not None:
        config['training']['h_barrier_coef'] = float(args.h_barrier_coef)
    if args.h_barrier_margin is not None:
        config['training']['h_barrier_margin'] = float(
            args.h_barrier_margin)
    if args.task_gate_low is not None:
        config['training']['task_gate_low'] = float(args.task_gate_low)
    if args.task_gate_high is not None:
        config['training']['task_gate_high'] = float(args.task_gate_high)
    if args.task_gate_positive_only is not None:
        config['training']['task_gate_positive_only'] = bool(
            args.task_gate_positive_only)
    if args.task_gate_scope is not None:
        config['training']['task_gate_scope'] = str(args.task_gate_scope)
    if args.h_feature_scale is not None:
        config['h']['feature_scale'] = float(args.h_feature_scale)
    if args.use_task_scalar is not None:
        config['training']['use_task_scalar'] = bool(args.use_task_scalar)
    if args.use_task_grad is not None:
        config['training']['use_task_grad'] = bool(args.use_task_grad)
    if args.task_value_checkpoint is not None:
        config['task_value']['checkpoint'] = str(args.task_value_checkpoint)
    if args.freeze_task_value is not None:
        config['task_value']['freeze'] = bool(args.freeze_task_value)
    if args.zero_init_task_block is not None:
        config['task_value']['zero_init_block'] = bool(
            args.zero_init_task_block)
    if args.h_co_train is not None:
        config['h']['co_train'] = bool(args.h_co_train)
    if args.use_h_features is not None:
        config['h']['use_features'] = bool(args.use_h_features)
    if args.cbf_coef is not None:
        config['training']['cbf_coef'] = float(args.cbf_coef)
    if args.cbf_alpha is not None:
        config['training']['cbf_alpha'] = float(args.cbf_alpha)
    if args.cbf_eps is not None:
        config['training']['cbf_eps'] = float(args.cbf_eps)

    seed = int(config['seed'])
    config['training']['eval_seed'] = int(
        config['training'].get('eval_seed', seed + 10000))
    config['training']['deterministic'] = bool(
        config['training'].get('deterministic', False))
    deterministic = bool(config['training']['deterministic'])
    if deterministic:
        # Must be set before the first CUDA/cuBLAS context is created.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(config['training']['device'])
    out = assert_independent_path(
        _cwd_path(args.out), kind='V6 actor output')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))

    env_cfg = config['env']
    env = OfficialDoubleIntegrator(
        num_agents=env_cfg['num_agents'],
        area_size=env_cfg['area_size'],
        max_step=env_cfg['max_step'],
        num_obstacles=env_cfg['num_obstacles'],
        n_rays=env_cfg['n_rays'],
        max_neighbors=env_cfg['max_neighbors'],
        seed=seed,
    )
    obs_dim = env.observation_dim
    env.reset(seed=seed)
    h_path = assert_independent_path(
        _resolve(args.h_checkpoint or config['h']['checkpoint']),
        kind='V6 h checkpoint')
    if not h_path.exists():
        raise FileNotFoundError(
            f'learned h checkpoint missing: {h_path}; run '
            'train_v6_official_h.py or train_v6_h.py first')
    h_offset = config['h'].get('h_offset')
    if h_path.suffix.lower() in ('.pkl', '.pickle'):
        hnet = official_h_provider(
            h_path, env, h_offset=None if h_offset is None else float(h_offset),
            h_scale=config['h'].get('h_scale'))
    else:
        hnet = LocalSafetyProvider.from_checkpoint(
            h_path, env, device=device,
            max_neighbors=env_cfg['max_neighbors'],
            n_rays=env_cfg['n_rays'],
            comm_radius=env.comm_radius)
        if h_offset is not None:
            hnet.h_offset = float(h_offset)

    h_trainer = None
    if bool(config.get('h', {}).get('co_train', False)):
        if not hasattr(hnet, 'params'):
            raise ValueError(
                'h co-training requires the V6 full-graph h provider')
        h_trainer = OfficialHCoTrainer(
            hnet, config['h'], seed=seed)
        print('enabled alternating h co-training from actor rollouts',
              flush=True)

    model = QpFreePolicy(
        obs_dim,
        hidden=config['policy']['hidden'],
        correction_radius=config['policy']['correction_radius'],
        initial_log_std=config['policy']['initial_log_std'],
        extra_dim=5,
    ).to(device)
    # Main V6 actor is always from scratch. A checkpoint-based continuation,
    # if ever used, must be an explicitly V6-owned stage-1 checkpoint, not a
    # V5/V4/official artifact.
    config['policy']['init_checkpoint'] = None
    # Record the actually used h path.  CLI overrides otherwise leave the
    # saved config pointing at the config default, which makes fixed-h policy
    # checkpoints easy to evaluate with the wrong h.
    config['h']['checkpoint'] = str(h_path)
    (out / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    task_net = build_task_value_net(
        config['task_value'], obs_dim,
        max_neighbors=env_cfg['max_neighbors'],
        n_rays=env_cfg['n_rays'],
    ).to(device)
    ema = TaskAdvantageEMA.zeros(
        env.num_agents,
        warmup_count=float(config['task_value'].get('warmup_count', 1.0)),
        clip=float(config['task_value'].get('clip', 5.0)),
        decay=float(config['task_value'].get('ema_decay', 0.99)),
        var_floor=float(config['task_value'].get('var_floor', 1.0)),
    )
    policy_opt = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    task_opt = torch.optim.Adam(
        task_net.parameters(), lr=config['task_value']['lr'])

    steps = 0
    if args.resume:
        resume_path = assert_independent_path(
            _cwd_path(args.resume), kind='V6 resume checkpoint')
        model, task_net, ema, resume = load_v6_checkpoint(
            resume_path, device=device)
        policy_opt = torch.optim.Adam(
            model.parameters(), lr=config['training']['lr'])
        task_opt = torch.optim.Adam(
            task_net.parameters(), lr=config['task_value']['lr'])
        if 'policy_optimizer' in resume:
            policy_opt.load_state_dict(resume['policy_optimizer'])
        if 'task_optimizer' in resume:
            task_opt.load_state_dict(resume['task_optimizer'])
        steps = int(resume.get('steps', 0))
        print(f'resumed from {args.resume} at steps={steps}', flush=True)

    freeze_task_value = bool(config['task_value'].get('freeze', False))
    zero_init_task_block = bool(
        config['task_value'].get('zero_init_block', False))
    task_value_checkpoint = config['task_value'].get('checkpoint')
    if task_value_checkpoint:
        tv_path = assert_independent_path(
            _resolve(task_value_checkpoint), kind='V6 task-value checkpoint')
        if not tv_path.exists():
            raise FileNotFoundError(f'missing V_T checkpoint: {tv_path}')
        tv = torch.load(tv_path, map_location=device, weights_only=False)
        task_net.load_state_dict(tv['task_net'])
        ema = TaskAdvantageEMA.from_state_dict(tv['task_ema'])
        config['task_value']['loaded_checkpoint'] = str(tv_path)
        config['task_value']['loaded_target'] = tv.get('target')
        print(f'loaded V_T: {tv_path} target={tv.get("target")}', flush=True)
    if zero_init_task_block:
        with torch.no_grad():
            model.encoder[0].weight[:, -model.extra_dim:] = 0.0
        print('zero-initialized actor task-feature input columns', flush=True)
    if freeze_task_value:
        task_opt = None
        print('V_T and its EMA are frozen', flush=True)

    total_steps = int(config['training']['total_steps'])
    rollout_steps = int(config['training']['rollout_steps'])
    eval_interval = int(config['training']['eval_interval'])
    save_interval = int(config['training']['save_interval'])
    eval_episodes = int(config['training']['eval_episodes'])
    eval_seed = int(config['training'].get(
        'eval_seed', seed + 10000))
    eval_stochastic = bool(config['training'].get(
        'eval_stochastic', False))
    reward_scale = float(config['training']['reward_scale'])
    reach_bonus = float(config['training'].get('reach_bonus', 0.0))
    task_grad_scale = float(config['training'].get('task_grad_scale', 1.0))
    h_feature_scale = float(config['h'].get('feature_scale', 1.0))
    task_grad_start = int(config['training'].get('task_grad_start_steps', 0))
    history = []
    eval_history = []
    best_score = -float('inf')
    best_step = None
    print(
        f'V6 training: device={device} obs_dim={obs_dim} '
        f'h={h_path} total_steps={total_steps} rollout_steps={rollout_steps} '
        f'seed={seed} eval_seed={eval_seed} deterministic={deterministic} '
        f'task_gate_scope={config["training"].get("task_gate_scope", "all_positive")}',
        flush=True)

    while steps < total_steps:
        rollout = collect_rollout_v6(
            env, hnet, task_net, ema, model,
            rollout_steps, seed + steps, device, config,
            global_step=steps)
        add_gae_v6(
            rollout, config['training']['gamma'],
            config['training']['gae_lambda'])
        update = ppo_update_v6(model, policy_opt, rollout, config, device)
        if freeze_task_value:
            task_update = {}
        else:
            task_update = task_value_update(
                task_net, task_opt, rollout, device=device,
                epochs=int(config['task_value'].get('epochs', 4)),
                batch_size=int(config['task_value'].get('batch_size', 256)),
                max_grad_norm=float(config['training']['max_grad_norm']))
            ema.update(rollout.task_value)
        h_update = {}
        if (h_trainer is not None
                and (steps // rollout_steps) % int(config['h'].get(
                    'update_every', 1)) == 0):
            h_update = h_trainer.update(rollout)
        steps += rollout_steps

        recent = rollout.completed[-eval_episodes:] if rollout.completed else []
        completed = ({key: float(np.mean([row[key] for row in recent]))
                      for key in recent[0]} if recent else {})
        row = dict(
            steps=steps, update=update, task_update=task_update,
            h_update=h_update,
            completed=completed,
            reward_mean=float(rollout.reward.mean()),
            task_reward_mean=float(rollout.task_reward.mean()),
            task_gate_mean=float(rollout.task_gate.mean()),
            task_gate_closed=float(
                (rollout.task_gate < 0.5).float().mean()),
            task_gate_scope=str(config['training'].get(
                'task_gate_scope', 'all_positive')),
            unsafe_mean=float(rollout.cost_target.mean()),
            h_mean=float(rollout.h.mean()),
            h_neg_rate=float((rollout.h < 0).float().mean()),
            task_ema_count=float(ema.count),
            task_a_mean=float(rollout.task_feat[..., 0].mean()),
            task_a_absmax=float(rollout.task_feat[..., 0].abs().max()),
            task_grad_active=bool(steps >= task_grad_start),
        )
        history.append(row)
        with open(out / 'train.jsonl', 'a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(f'steps={steps} {row}', flush=True)

        if eval_interval and steps % eval_interval == 0:
            result = evaluate_v6(
                env, hnet, task_net, ema, model,
                seeds=[eval_seed + i for i in range(eval_episodes)],
                device=device, reward_scale=reward_scale,
                reach_bonus=reach_bonus,
                use_h_features=bool(config['h'].get('use_features', True)),
                h_feature_scale=h_feature_scale,
                max_neighbors=env_cfg['max_neighbors'],
                n_rays=env_cfg['n_rays'],
                task_grad_scale=task_grad_scale,
                use_task_scalar=bool(
                    config['training'].get('use_task_scalar', True)),
                use_task_grad=bool(
                    config['training'].get('use_task_grad', True))
                and steps >= task_grad_start,
                cbf_alpha=float(config['training'].get('cbf_alpha', 1.0)),
                cbf_eps=float(config['training'].get('cbf_eps', 0.02)),
                mass=float(config['training'].get('mass', 0.1)),
                stochastic=eval_stochastic)
            eval_history.append(dict(steps=steps, **result))
            with open(out / 'eval.jsonl', 'a') as stream:
                stream.write(json.dumps(dict(steps=steps, **result)) + '\n')
            score = (result['safety']
                     + 0.1 * result.get('reach_final', 0.0)
                     + 0.001 * result['episode_return'])
            if score > best_score:
                best_score = score
                best_step = steps
                save_v6_checkpoint(
                    out / 'policy_best.pth', model, task_net, ema,
                    config, steps, obs_dim)
                if h_trainer is not None:
                    h_trainer.save(out / 'h_best.pkl')
            print(f'eval steps={steps} {result}', flush=True)

        if save_interval and steps % save_interval == 0:
            save_v6_checkpoint(
                out / f'policy_step_{steps}.pth', model, task_net, ema,
                config, steps, obs_dim)

    save_v6_checkpoint(
        out / 'policy_final.pth', model, task_net, ema, config, steps, obs_dim)
    if h_trainer is not None:
        h_trainer.save(out / 'h_final.pkl')
        print(f'saved co-trained h: {out / "h_final.pkl"}', flush=True)
    if args.skip_final_eval:
        result = {}
    else:
        result = evaluate_v6(
            env, hnet, task_net, ema, model,
            seeds=[eval_seed + i for i in range(eval_episodes)],
            device=device, reward_scale=reward_scale,
            reach_bonus=reach_bonus,
            use_h_features=bool(config['h'].get('use_features', True)),
            h_feature_scale=h_feature_scale,
            max_neighbors=env_cfg['max_neighbors'],
            n_rays=env_cfg['n_rays'],
            task_grad_scale=task_grad_scale,
            use_task_scalar=bool(
                config['training'].get('use_task_scalar', True)),
            use_task_grad=bool(
                config['training'].get('use_task_grad', True))
            and steps >= task_grad_start,
            cbf_alpha=float(config['training'].get('cbf_alpha', 1.0)),
            cbf_eps=float(config['training'].get('cbf_eps', 0.02)),
            mass=float(config['training'].get('mass', 0.1)),
            stochastic=eval_stochastic)
        eval_history.append(dict(steps=steps, **result))
        score = (result['safety']
                 + 0.1 * result.get('reach_final', 0.0)
                 + 0.001 * result['episode_return'])
        if score > best_score:
            best_score = score
            best_step = steps
            save_v6_checkpoint(
                out / 'policy_best.pth', model, task_net, ema,
                config, steps, obs_dim)
            if h_trainer is not None:
                h_trainer.save(out / 'h_best.pkl')
        print(f'final eval: {result}', flush=True)
    (out / 'summary.json').write_text(json.dumps(dict(
        steps=steps, seed=seed, obs_dim=obs_dim, eval=result,
        h_co_train=h_trainer is not None,
        h_checkpoint=(str(out / 'h_final.pkl')
                      if h_trainer is not None else None),
        best_step=best_step,
        best_score=best_score if best_step else None,
        history=history, eval_history=eval_history), indent=2) + '\n')
    print(f'saved: {out}', flush=True)


if __name__ == '__main__':
    main()
