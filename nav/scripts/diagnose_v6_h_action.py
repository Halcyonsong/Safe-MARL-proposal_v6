"""Action-channel diagnostics for a learned V6 h (legacy diagnostic).

This script is not part of the independent V6 main protocol.  It may be used
for V6-only diagnostics; any official-h comparison or V5 checkpoint path is
legacy and must not be used for training or calibration.

Classification accuracy is not enough: the h used by the policy must have a
useful derivative through the action channel.  For the DoubleIntegrator

    h_dot = grad_p(h) . v + grad_v(h) . u / m,

so ``grad_v(h)`` is the only part of the state gradient that can change the
action.  This script reports:

- gradient norms and the velocity-block norm;
- CBF feasibility: can *some* box-bounded action satisfy h_dot + alpha*h >= 0;
- CBF satisfaction under the behavior policy;
- optionally, Pearson/cosine agreement with a reference h on the same states
  (legacy diagnostic only; official h is forbidden in the independent path).

Run without a policy checkpoint for random-state coverage.  A V5 checkpoint is
legacy and must not be used for V6 main-path tuning.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from algos.safe_policy.training import load_checkpoint  # noqa: E402
from envs.gcbfplus import OfficialDoubleIntegrator  # noqa: E402
from v6.official_h import (  # noqa: E402
    OfficialLearnedH, provider_from_checkpoint)
from v6.safety_labels import cbf_terms_numpy  # noqa: E402


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _provider(path: Path, env):
    with path.open('rb') as stream:
        payload = pickle.load(stream)
    if payload.get('schema') == 'v6-official-cbf-h':
        return provider_from_checkpoint(path, env)
    if 'params' in payload:
        return OfficialLearnedH(
            payload['params'], env, n_agents=env.num_agents, gnn_layers=1)
    raise ValueError(f'unsupported h checkpoint: {path}')


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    valid = denom > 1e-12
    if not np.any(valid):
        return 0.0
    return float(np.mean(np.sum(a * b, axis=-1)[valid] / denom[valid]))


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if len(a) < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/v6/v6_nav_independent.yaml')
    ap.add_argument('--h-checkpoint', required=True)
    ap.add_argument('--reference-h',
                    help='legacy official cbf.pkl comparison; forbidden for '
                         'V6 training/calibration')
    ap.add_argument('--policy-checkpoint',
                    help='legacy V5/V6 policy; V5 is forbidden for V6 main')
    ap.add_argument('--behavior-h-checkpoint',
                    help='optional fixed h for the policy features; keeps the '
                         'state distribution identical across compared h values')
    ap.add_argument('--behavior-zero', action='store_true',
                    help='use zero h/grad features for the behavior policy')
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--steps', type=int, default=256)
    ap.add_argument('--seed', type=int, default=10000)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--cbf-alpha', type=float, default=1.0)
    ap.add_argument('--cbf-eps', type=float, default=0.02)
    ap.add_argument('--out')
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = NAV / cfg_path
    config = yaml.safe_load(cfg_path.read_text())
    env_cfg = config['env']
    env = OfficialDoubleIntegrator(
        num_agents=env_cfg['num_agents'],
        area_size=env_cfg['area_size'],
        max_step=env_cfg['max_step'],
        num_obstacles=env_cfg['num_obstacles'],
        n_rays=env_cfg['n_rays'],
        max_neighbors=env_cfg['max_neighbors'],
        seed=args.seed)
    env.reset(seed=args.seed)
    mass = float(env._env.params.get('m', 0.1))
    hnet = _provider(_resolve(args.h_checkpoint), env)
    ref = _provider(_resolve(args.reference_h), env) if args.reference_h else None
    behavior_h = (_provider(_resolve(args.behavior_h_checkpoint), env)
                  if args.behavior_h_checkpoint else None)
    policy = None
    if args.policy_checkpoint:
        policy, _ = load_checkpoint(
            _resolve(args.policy_checkpoint), device=args.device)
        policy.eval()

    rng = np.random.default_rng(args.seed)
    h_all, href_all = [], []
    g_all, gref_all = [], []
    grad_norm, grad_p_norm, grad_v_norm, grad_v_scaled_norm = [], [], [], []
    hdot_pred, hdot_true, feasible, violation = [], [], [], []
    sat_pred, sat_true = [], []

    for episode in range(int(args.episodes)):
        obs = env.reset(seed=args.seed + episode)
        h, grad = hnet.value_and_local_grad()
        for _ in range(int(args.steps)):
            states = env.get_agent_states()
            if policy is None:
                action = rng.uniform(-1.0, 1.0, (env.num_agents, 2)).astype(
                    np.float32)
            else:
                if args.behavior_zero:
                    hb = np.zeros(env.num_agents, dtype=np.float32)
                    gb = np.zeros((env.num_agents, 4), dtype=np.float32)
                elif behavior_h is not None:
                    hb, gb = behavior_h.value_and_local_grad()
                else:
                    hb, gb = h, grad
                with torch.no_grad():
                    out = policy(
                        torch.tensor(obs, dtype=torch.float32,
                                     device=args.device),
                        torch.tensor(hb, dtype=torch.float32,
                                     device=args.device).unsqueeze(-1),
                        torch.tensor(gb, dtype=torch.float32,
                                     device=args.device))
                action = out['action'].detach().cpu().numpy().astype(np.float32)
            hd, viol, feas = cbf_terms_numpy(
                h, grad, states, action, mass=mass,
                alpha=args.cbf_alpha, eps=args.cbf_eps)
            hdot_pred.append(hd)
            violation.append(viol)
            feasible.append(feas)
            sat_pred.append(hd + args.cbf_alpha * np.asarray(h) >= 0.0)
            grad_norm.append(np.linalg.norm(grad, axis=-1))
            grad_p_norm.append(np.linalg.norm(grad[:, :2], axis=-1))
            grad_v_norm.append(np.linalg.norm(grad[:, 2:], axis=-1))
            grad_v_scaled_norm.append(
                np.linalg.norm(grad[:, 2:] / mass, axis=-1))
            h_all.append(np.asarray(h))
            g_all.append(np.asarray(grad))
            if ref is not None:
                href, gref = ref.value_and_local_grad()
                href_all.append(np.asarray(href))
                gref_all.append(np.asarray(gref))

            obs, _, terminated, truncated, _ = env.step(action)
            h_next, grad_next = hnet.value_and_local_grad()
            hd_true = (np.asarray(h_next) - np.asarray(h)) / env.dt
            hdot_true.append(hd_true)
            sat_true.append(hd_true + args.cbf_alpha * np.asarray(h) >= 0.0)
            h, grad = h_next, grad_next
            if terminated or truncated:
                break

    def _mean(rows):
        if not rows:
            return 0.0
        return float(np.mean(np.concatenate(
            [np.asarray(r).reshape(-1) for r in rows])))

    result = dict(
        episodes=int(args.episodes),
        h_mean=float(np.mean(np.concatenate(h_all))),
        h_std=float(np.std(np.concatenate(h_all))),
        h_min=float(np.min(np.concatenate(h_all))),
        grad_norm=_mean(grad_norm),
        grad_pos_norm=_mean(grad_p_norm),
        grad_vel_norm=_mean(grad_v_norm),
        grad_vel_scaled_norm=_mean(grad_v_scaled_norm),
        cbf_pred_sat=_mean(sat_pred),
        cbf_true_sat=_mean(sat_true),
        cbf_feasible=_mean(feasible),
        cbf_violation=_mean(violation),
        hdot_pred_mean=_mean(hdot_pred),
        hdot_true_mean=_mean(hdot_true),
        states=int(sum(len(x) for x in h_all)),
    )
    if ref is not None and href_all:
        ref_g = np.concatenate(gref_all)
        ref_h = np.concatenate(href_all)
        result.update(
            ref_h_mean=float(np.mean(ref_h)),
            ref_h_pearson=_pearson(np.concatenate(h_all), ref_h),
            ref_grad_norm=float(np.mean(np.linalg.norm(ref_g, axis=-1))),
            ref_grad_vel_norm=float(np.mean(
                np.linalg.norm(ref_g[..., 2:], axis=-1))),
            ref_grad_vel_scaled_norm=float(np.mean(
                np.linalg.norm(ref_g[..., 2:] / mass, axis=-1))),
            ref_grad_cosine=_cosine(np.concatenate(g_all), ref_g),
            ref_grad_vel_cosine=_cosine(
                np.concatenate(g_all)[..., 2:], ref_g[..., 2:]),
        )
    result.update(
        h_checkpoint=str(_resolve(args.h_checkpoint)),
        reference_h=(str(_resolve(args.reference_h))
                     if args.reference_h else None),
        policy_checkpoint=(str(_resolve(args.policy_checkpoint))
                           if args.policy_checkpoint else None),
        behavior_h_checkpoint=(str(_resolve(args.behavior_h_checkpoint))
                               if args.behavior_h_checkpoint else None),
        behavior_zero=bool(args.behavior_zero),
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2))
    if args.out:
        out = _resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
