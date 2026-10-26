"""Train V6 from-scratch MAPPO on Ant-2x4 with optional frozen h features.

Arms:
    h_only    : h + grad_s h features, fixed cost penalty
    cost_only : no h features, fixed cost penalty
    task_only : no h features, no cost penalty

No MAPPO-Lagrangian multiplier, no QP/CBF correction, no V5 warm-start and no
official h checkpoint are used.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from v6.ant_env import Ant2x4Env  # noqa: E402
from v6.ant_h import AntSafetyProvider  # noqa: E402
from v6.ant_mappo import (  # noqa: E402
    MappoAntModel, collect_ant_rollout, compute_ant_gae,
    concat_ant_rollouts, evaluate_ant_mappo, load_ant_mappo_model,
    ppo_update_ant, save_ant_mappo_checkpoint,
)
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


def _set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _build_provider(config: dict, h_checkpoint: str | None,
                    device: torch.device):
    use_h = bool(config.get('h', {}).get('use_features', True))
    if use_h:
        if not h_checkpoint:
            raise ValueError('h.use_features=true requires --h-checkpoint or '
                             'h.checkpoint in the config')
        path = _resolve(h_checkpoint)
        assert_independent_path(path, kind='V6 Ant h checkpoint')
        provider = AntSafetyProvider.from_checkpoint(path, device=device)
        return provider, path, True
    # No-h baselines do not need the analytic finite-difference h at all.
    # Keeping a zero provider makes task_only/cost_only a clean MAPPO
    # baseline and removes nine geometry-gradient evaluations per step.
    return NullAntSafetyProvider(), None, False


def _cost_penalty_at(total_steps: int, train_cfg: dict,
                     fallback: float) -> float:
    """Fixed base penalty, optionally ramped once to a final value.

    This is a pre-scheduled coefficient, not a Lagrangian multiplier: the
    coefficient does not depend on observed costs and no dual update is used.
    """
    base = float(train_cfg.get('cost_penalty', fallback))
    end = train_cfg.get('cost_penalty_end')
    if end is None:
        return base
    start = float(train_cfg.get('cost_penalty_ramp_start', 0))
    ramp = max(1.0, float(train_cfg.get('cost_penalty_ramp_steps', 1)))
    frac = min(1.0, max(0.0, (float(total_steps) - start) / ramp))
    return base + frac * (float(end) - base)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/v6/v6_ant_h_only.yaml')
    ap.add_argument('--out')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--total-steps', type=int)
    ap.add_argument('--rollout-steps', type=int)
    ap.add_argument('--num-envs', type=int)
    ap.add_argument('--eval-interval', type=int)
    ap.add_argument('--eval-episodes', type=int)
    ap.add_argument('--eval-seed', type=int)
    ap.add_argument('--save-interval', type=int)
    ap.add_argument('--episode-limit', type=int)
    ap.add_argument('--cost-penalty', type=float)
    ap.add_argument('--device', default=None)
    ap.add_argument('--deterministic', action=argparse.BooleanOptionalAction,
                    default=None)
    ap.add_argument('--h-checkpoint')
    ap.add_argument('--resume',
                    help='Warm-start/continue from an Ant MAPPO checkpoint')
    ap.add_argument('--task-value-checkpoint',
                    help='Frozen Ant V_T checkpoint for the A_T actor input')
    ap.add_argument('--freeze-task-value',
                    action=argparse.BooleanOptionalAction, default=True,
                    help='Freeze V_T and its EMA (required for this arm)')
    ap.add_argument('--skip-final-eval', action='store_true')
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = NAV / cfg_path
    config = yaml.safe_load(cfg_path.read_text())
    cfg = dict(config)
    seed = int(args.seed if args.seed is not None else cfg.get('seed', 0))
    train_cfg = dict(cfg.get('training', {}))
    env_cfg = dict(cfg.get('env', {}))
    policy_cfg = dict(cfg.get('policy', {}))
    h_cfg = dict(cfg.get('h', {}))

    out = _resolve(args.out or cfg.get(
        'out', 'outputs/v6/independent/ant/mappo_h_only'))
    out.mkdir(parents=True, exist_ok=True)
    assert_independent_path(out, kind='V6 Ant MAPPO output')
    device = _device(args.device or train_cfg.get('device', 'auto'))
    deterministic = bool(train_cfg.get('deterministic', True)
                         if args.deterministic is None
                         else args.deterministic)

    total_steps_target = int(
        args.total_steps if args.total_steps is not None
        else train_cfg.get('total_steps', 1_000_000))
    rollout_steps = int(
        args.rollout_steps if args.rollout_steps is not None
        else train_cfg.get('rollout_steps', 2048))
    num_envs = max(1, int(
        args.num_envs if args.num_envs is not None
        else train_cfg.get('num_envs', 1)))
    eval_interval = int(
        args.eval_interval if args.eval_interval is not None
        else train_cfg.get('eval_interval', 100_000))
    eval_episodes = int(
        args.eval_episodes if args.eval_episodes is not None
        else train_cfg.get('eval_episodes', 20))
    eval_seed = int(
        args.eval_seed if args.eval_seed is not None
        else train_cfg.get('eval_seed', seed + 20000))
    save_interval = int(
        args.save_interval if args.save_interval is not None
        else train_cfg.get('save_interval', 100_000))
    cost_penalty = float(
        args.cost_penalty if args.cost_penalty is not None
        else train_cfg.get('cost_penalty', 10.0))
    h_barrier_coef = float(train_cfg.get('h_barrier_coef', 0.0))
    h_barrier_margin = float(train_cfg.get('h_barrier_margin', 0.0))
    forward_gate_low = float(train_cfg.get('forward_gate_low', 0.0))
    forward_gate_high = float(train_cfg.get('forward_gate_high', 0.0))
    h_checkpoint = args.h_checkpoint or h_cfg.get('checkpoint')
    resume_path = _resolve(args.resume) if args.resume else None
    base_ckpt = None
    if resume_path is not None:
        assert_independent_path(
            resume_path, kind='V6 Ant warm-start base checkpoint')
        base_ckpt = torch.load(
            resume_path, map_location='cpu', weights_only=False)
        if h_checkpoint is None:
            h_checkpoint = base_ckpt.get('h_checkpoint')
    task_value_path = (
        _resolve(args.task_value_checkpoint)
        if args.task_value_checkpoint else None)
    if task_value_path is not None:
        assert_independent_path(
            task_value_path, kind='V6 Ant task-value checkpoint')
        if not args.freeze_task_value:
            raise NotImplementedError(
                'this diagnostic requires a frozen V_T and EMA')
    provider, h_path, use_h = _build_provider(
        cfg, h_checkpoint, device)
    _set_seed(seed, deterministic)

    env_kwargs = dict(
        episode_limit=int(args.episode_limit if args.episode_limit is not None
                           else env_cfg.get('episode_limit', 1000)),
        agent_obsk=int(env_cfg.get('agent_obsk', 1)),
        h_kappa=float(env_cfg.get('h_kappa', 0.0)),
        h_kappa_z=float(env_cfg.get('h_kappa_z', 0.0)),
        upright_scale=float(env_cfg.get('upright_scale', 0.2)),
        safety_provider=provider,
        h_scale=float(env_cfg.get('h_scale', 0.5)),
    )
    envs = [
        Ant2x4Env(seed=seed + i, **env_kwargs)
        for i in range(num_envs)
    ]
    env = envs[0]
    task_value = None
    task_feature_fn = None
    if task_value_path is not None:
        task_value = load_ant_task_value(task_value_path, device=device)
        task_feature_fn = make_task_feature_fn(task_value, device=device)
        print(f'loaded frozen Ant V_T: {task_value_path}', flush=True)
    if resume_path is not None:
        model, base_ckpt = load_ant_mappo_model(
            resume_path, device=device,
            task_value_expand=bool(task_value_path),
            task_dim=1 if task_value_path else 0)
        base_use_h = bool(
            base_ckpt.get('model_config', {}).get('use_h_features', True))
        if base_use_h != use_h:
            raise ValueError(
                f'resume checkpoint use_h_features={base_use_h} but the '
                f'config has use_features={use_h}')
        start_step = int(base_ckpt.get('total_steps', 0))
        print(
            f'resumed Ant policy: {resume_path} start_step={start_step} '
            f'task_value={bool(task_value_path)}', flush=True)
    else:
        model = MappoAntModel(
            obs_dim=env.observation_dim,
            state_dim=env.global_state_dim,
            action_dim=env.action_dim,
            num_agents=env.num_agents,
            use_h_features=use_h,
            use_task_value=bool(task_value_path),
            task_dim=1 if task_value_path else 0,
            actor_hidden=tuple(policy_cfg.get('actor_hidden', (64, 64))),
            critic_hidden=tuple(policy_cfg.get('critic_hidden', (128, 128))),
            actor_activation=str(policy_cfg.get(
                'actor_activation', 'tanh')),
            critic_activation=str(policy_cfg.get(
                'critic_activation', 'tanh')),
            initial_log_std=float(policy_cfg.get('initial_log_std', -0.5)),
            h_feature_scale=float(policy_cfg.get('h_feature_scale', 1.0)),
            grad_feature_scale=float(
                policy_cfg.get('grad_feature_scale', 1.0)),
            actor_feature_norm=bool(
                policy_cfg.get('actor_feature_norm', False)),
            actor_layer_norm=bool(
                policy_cfg.get('actor_layer_norm', False)),
            critic_feature_norm=bool(
                policy_cfg.get('critic_feature_norm', False)),
            critic_layer_norm=bool(
                policy_cfg.get('critic_layer_norm', False)),
            use_value_norm=bool(
                policy_cfg.get('use_value_norm', False)),
            action_dist=str(policy_cfg.get('action_dist', 'bounded')),
        ).to(device)
        start_step = 0
    if start_step >= total_steps_target:
        raise ValueError(
            f'start_step {start_step} >= total_steps_target '
            f'{total_steps_target}; pass a larger --total-steps')
    # Match upstream MAPPO's separate actor/critic learning rates.  The critic
    # often needs a much larger step size than the policy on Ant.
    actor_optimizer = torch.optim.Adam(
        model.actors.parameters(),
        lr=float(train_cfg.get('lr', 3e-4)),
        eps=float(train_cfg.get('opti_eps', 1e-5)),
    )
    critic_optimizer = torch.optim.Adam(
        model.critic.parameters(),
        lr=float(train_cfg.get('critic_lr', train_cfg.get('lr', 3e-4))),
        eps=float(train_cfg.get('opti_eps', 1e-5)),
    )
    # Backward-compatible alias used by ppo_update_ant; the actual updates use
    # the separate actor/critic optimizers.
    optimizer = actor_optimizer

    resolved = dict(
        schema='v6.ant_mappo.train.config.v1',
        config_path=str(cfg_path),
        seed=seed,
        device=str(device),
        deterministic=deterministic,
        total_steps=total_steps_target,
        start_step=int(start_step),
        resume=str(resume_path) if resume_path else None,
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        eval_seed=eval_seed,
        save_interval=save_interval,
        cost_penalty=cost_penalty,
        cost_penalty_end=train_cfg.get('cost_penalty_end'),
        cost_penalty_ramp_start=train_cfg.get(
            'cost_penalty_ramp_start', 0),
        cost_penalty_ramp_steps=train_cfg.get(
            'cost_penalty_ramp_steps', 1),
        h_barrier_coef=h_barrier_coef,
        h_barrier_margin=h_barrier_margin,
        forward_gate_low=forward_gate_low,
        forward_gate_high=forward_gate_high,
        use_h_features=use_h,
        h_checkpoint=str(h_path) if h_path else None,
        task_value=(
            dict(
                checkpoint=str(task_value_path),
                freeze=True,
                feature='A_T',
                feature_dim=1,
            ) if task_value_path else None),
        env=env.get_env_info(),
        env_cfg=env_cfg,
        policy=policy_cfg,
        policy_cfg=policy_cfg,
        training=train_cfg,
        training_cfg=train_cfg,
        h=h_cfg,
        model_config=model.model_config(),
    )
    (out / 'resolved_config.json').write_text(
        json.dumps(resolved, indent=2) + '\n')
    print(json.dumps(resolved, indent=2), flush=True)

    total_steps = int(start_step)
    update = 0
    next_eval = (start_step + eval_interval
                 if eval_interval > 0 else total_steps_target + 1)
    next_save = (start_step + save_interval
                 if save_interval > 0 else total_steps_target + 1)
    history: list[dict] = []
    final_eval: dict = {}
    started = time.time()

    while total_steps < total_steps_target:
        remaining = total_steps_target - total_steps
        current_cost_penalty = _cost_penalty_at(
            total_steps, train_cfg, cost_penalty)
        # ``rollout_steps`` is per environment.  With the configured env count
        # this gives an ~16k-transition update, matching the upstream MAPPO
        # batch scale instead of a single correlated 2k trajectory.
        per_env_steps = max(1, min(rollout_steps, remaining // num_envs))
        rollouts = []
        collected = 0
        for i, env_i in enumerate(envs):
            steps_i = min(per_env_steps, remaining - collected)
            if steps_i <= 0:
                break
            rollout_i = collect_ant_rollout(
                env_i, model,
                rollout_steps=steps_i,
                seed=seed + update * 1_000_003 + i * 100_003,
                device=device,
                cost_penalty=current_cost_penalty,
                h_barrier_coef=h_barrier_coef,
                h_barrier_margin=h_barrier_margin,
                forward_gate_low=forward_gate_low,
                forward_gate_high=forward_gate_high,
                deterministic=False,
                task_feature_fn=task_feature_fn,
            )
            rollouts.append(rollout_i)
            collected += len(rollout_i)
        rollout = concat_ant_rollouts(rollouts)
        advantages_list = []
        returns_list = []
        gae_infos = []
        for r in rollouts:
            adv_i, ret_i, info_i = compute_ant_gae(
                r,
                gamma=float(train_cfg.get('gamma', 0.99)),
                gae_lambda=float(train_cfg.get('gae_lambda', 0.95)),
                normalize_advantage=bool(
                    train_cfg.get('normalize_advantage', True)),
            )
            advantages_list.append(adv_i)
            returns_list.append(ret_i)
            gae_infos.append(info_i)
        advantages = np.concatenate(advantages_list, axis=0)
        returns = np.concatenate(returns_list, axis=0)
        gae_info = dict(
            adv_mean=float(np.mean([g['adv_mean'] for g in gae_infos])),
            adv_std=float(np.mean([g['adv_std'] for g in gae_infos])),
        )
        ppo_logs = ppo_update_ant(
            model, optimizer, rollout, advantages, returns,
            device=device,
            epochs=int(train_cfg.get('epochs', 4)),
            minibatch_size=int(train_cfg.get('minibatch_size', 256)),
            clip_range=float(train_cfg.get('clip_range', 0.2)),
            value_coef=float(train_cfg.get('value_coef', 0.5)),
            entropy_coef=float(train_cfg.get('entropy_coef', 0.001)),
            max_grad_norm=float(train_cfg.get('max_grad_norm', 0.5)),
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            value_huber_delta=float(train_cfg.get('huber_delta', 0.0)),
            update_value_norm=bool(
                train_cfg.get('update_value_norm', True)),
        )
        total_steps += len(rollout)
        update += 1
        row = dict(
            update=update,
            total_steps=int(total_steps),
            wall_time=float(time.time() - started),
            rollout=rollout.summary(),
            gae=gae_info,
            ppo=ppo_logs,
            cost_penalty=current_cost_penalty,
            h_barrier_coef=h_barrier_coef,
            h_barrier_margin=h_barrier_margin,
            forward_gate_low=forward_gate_low,
            forward_gate_high=forward_gate_high,
            num_envs=num_envs,
        )
        history.append(row)
        print(json.dumps(row), flush=True)

        if total_steps >= next_eval:
            eval_metrics = evaluate_ant_mappo(
                env, model, episodes=eval_episodes, seed=eval_seed,
                device=device, task_feature_fn=task_feature_fn)
            (out / f'eval_{total_steps:09d}.json').write_text(
                json.dumps(eval_metrics, indent=2) + '\n')
            row['eval'] = eval_metrics
            print(json.dumps(dict(eval_at=total_steps, **eval_metrics)),
                  flush=True)
            while next_eval <= total_steps:
                next_eval += max(1, eval_interval)

        if total_steps >= next_save:
            save_ant_mappo_checkpoint(
                out / f'policy_{total_steps:09d}.pth', model,
                config=resolved, total_steps=total_steps,
                h_checkpoint=str(h_path) if h_path else None,
            )
            while next_save <= total_steps:
                next_save += max(1, save_interval)

    final_path = save_ant_mappo_checkpoint(
        out / 'policy_final.pth', model, config=resolved,
        total_steps=total_steps,
        h_checkpoint=str(h_path) if h_path else None,
        eval_metrics=final_eval,
    )
    if not args.skip_final_eval:
        final_eval = evaluate_ant_mappo(
            env, model, episodes=eval_episodes, seed=eval_seed,
            device=device, task_feature_fn=task_feature_fn)
        (out / 'eval_final.json').write_text(
            json.dumps(final_eval, indent=2) + '\n')
        # Rewrite with final metrics in metadata.
        final_path = save_ant_mappo_checkpoint(
            out / 'policy_final.pth', model, config=resolved,
            total_steps=total_steps,
            h_checkpoint=str(h_path) if h_path else None,
            eval_metrics=final_eval,
        )
    summary = dict(
        schema='v6.ant_mappo.train.v1',
        checkpoint=str(final_path),
        total_steps=int(total_steps),
        updates=int(update),
        wall_time=float(time.time() - started),
        final_eval=final_eval,
        history=history,
        config=resolved,
    )
    (out / 'train_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    for env_i in envs:
        env_i.close()
    print(json.dumps(dict(
        checkpoint=str(final_path), total_steps=total_steps,
        final_eval=final_eval), indent=2), flush=True)


if __name__ == '__main__':
    main()
