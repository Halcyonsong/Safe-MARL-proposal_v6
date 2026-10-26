"""Train the official V5 learned safety heads (cost + survival).

This is stage 1 of the official chain.  It consumes an offline safety dataset:
states, official cost labels, terminal/survival labels, and optionally next
states.  It does not use return, QP labels, or an expert u_ref.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from v5.safety.dataset import SafetyDataset  # noqa: E402
from v5.safety.learned import LearnedSafetyModel  # noqa: E402
from v5.safety.train import train_learned_h  # noqa: E402


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/v5/learned_h_ant.yaml')
    ap.add_argument('--data')
    ap.add_argument('--out')
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = NAV / cfg_path
    config = yaml.safe_load(cfg_path.read_text())
    train_cfg = dict(config.get('train', {}))
    if args.data:
        train_cfg['data'] = args.data
    if args.out:
        train_cfg['out'] = args.out
    if args.epochs is not None:
        train_cfg['epochs'] = args.epochs
    data_path = _resolve(train_cfg['data'])
    out = _resolve(train_cfg.get('out', 'outputs/v5/official/ant/learned_h'))
    dataset = SafetyDataset.from_npz(data_path)
    cost_names = tuple(config.get('cost_names', dataset.cost_labels.shape[1] * ['cost']))
    if dataset.cost_dim != len(cost_names):
        raise ValueError(
            f'dataset has {dataset.cost_dim} cost columns, config has '
            f'{len(cost_names)} cost_names')
    model = LearnedSafetyModel(
        state_dim=int(config.get('state_dim', dataset.state_dim)),
        cost_names=cost_names,
        hidden=config.get('model', {}).get('hidden', (128, 128)),
        survival_hidden=config.get('model', {}).get('survival_hidden', (128,)),
    )
    summary = train_learned_h(
        model, dataset, dict(config, train=train_cfg), out, device=args.device)
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == '__main__':
    main()
