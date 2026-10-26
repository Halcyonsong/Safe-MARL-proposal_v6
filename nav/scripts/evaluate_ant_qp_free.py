"""Evaluate a V5 Ant smoke checkpoint on fixed seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

NAV = Path(__file__).resolve().parents[1]
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from algos.safe_policy.qp_free_ant import QpFreeAntEnsemble  # noqa: E402
from envs.mamujoco import Ant2x4Env  # noqa: E402
from scripts.train_ant_qp_free import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--num-episodes', type=int, default=10)
    ap.add_argument('--seed', type=int, default=10000)
    ap.add_argument('--safety-h-checkpoint', default=None,
                    help='learned h checkpoint; defaults to train config if present')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out')
    args = ap.parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    train_cfg = (ckpt.get('config') or {}).get('config', {})
    safety_h = args.safety_h_checkpoint or train_cfg.get('safety_h_checkpoint')
    provider = None
    if safety_h:
        from v5.safety.checkpoint import LearnedSafetyProvider
        h_path = Path(safety_h).expanduser()
        if not h_path.is_absolute():
            h_path = (Path.cwd() / h_path).resolve()
        provider = LearnedSafetyProvider.from_checkpoint(h_path, device=args.device)
    env = Ant2x4Env(seed=args.seed, episode_limit=1000,
                    h_kappa=train_cfg.get('h_kappa', 0.0),
                    h_kappa_z=train_cfg.get('h_kappa_z', 0.0),
                    safety_provider=provider)
    model = QpFreeAntEnsemble(
        env.num_agents, obs_dim=env.observation_dim, action_dim=env.action_dim,
        h_dim=4, grad_dim=9, hidden=128,
        correction_radius=train_cfg.get('correction_radius', 0.2),
        gate_margin=train_cfg.get('gate_margin', 0.1),
        gate_tau=train_cfg.get('gate_tau', 0.05),
        gate_on_wall=train_cfg.get('gate_on_wall', False),
        initial_log_std=train_cfg.get('initial_log_std', -3.0))
    model.load_state_dict(ckpt['model'], strict=False)
    device = torch.device(args.device)
    model.to(device)
    metrics = evaluate(env, model, list(range(args.seed, args.seed + args.num_episodes)),
                       device, cfg={})
    payload = dict(checkpoint=args.checkpoint, episodes=args.num_episodes,
                   eval_seed=args.seed, metrics=metrics, checkpoint_info=ckpt.get('config'))
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + '\n')
    env.close()


if __name__ == '__main__':
    main()
