"""Train the learned safety value h used by V6 navigation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

NAV = Path(__file__).resolve().parents[1]
ROOT = NAV.parent
if str(NAV) not in sys.path:
    sys.path.insert(0, str(NAV))

from v6.graph_nets import LocalGraphNet  # noqa: E402
from v6.safety import (  # noqa: E402
    LocalSafetyNet, SafetyDataset, evaluate_safety, train_safety_net)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/v6/v6_nav_independent.yaml')
    ap.add_argument('--data-dir')
    ap.add_argument('--train-data')
    ap.add_argument('--out')
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--unsafe-weight', type=float)
    ap.add_argument('--margin', type=float)
    ap.add_argument('--margin-coef', type=float)
    ap.add_argument('--hdot-coef', type=float)
    ap.add_argument('--hdot-alpha', type=float)
    ap.add_argument('--hdot-eps', type=float)
    ap.add_argument('--invariance-coef', type=float)
    ap.add_argument('--arch', choices=('graph', 'mlp'))
    ap.add_argument('--graph-hidden', type=int)
    ap.add_argument('--graph-layers', type=int)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = NAV / cfg_path
    config = yaml.safe_load(cfg_path.read_text())
    hcfg = dict(config['h']['train'])
    if args.data_dir:
        hcfg['data'] = args.data_dir
    if args.out:
        hcfg['out'] = args.out
    if args.epochs is not None:
        hcfg['epochs'] = args.epochs
    data_dir = _resolve(hcfg['data'])
    out = _resolve(hcfg['out'])
    train_override = args.train_data or hcfg.get('train_data')
    train_path = (_resolve(train_override) if train_override
                  else data_dir / 'train_data.npz')

    train = SafetyDataset.from_npz(train_path)
    val = SafetyDataset.from_npz(data_dir / 'val_data.npz')
    test = SafetyDataset.from_npz(data_dir / 'test_data.npz')
    arch = args.arch or hcfg.get('arch', 'graph')
    if arch == 'graph':
        env_cfg = config['env']
        model = LocalGraphNet(
            int(train.obs.shape[-1]),
            hidden=int(args.graph_hidden or hcfg.get('graph_hidden', 128)),
            layers=int(args.graph_layers or hcfg.get('graph_layers', 2)),
            max_neighbors=int(env_cfg['max_neighbors']),
            n_rays=int(env_cfg['n_rays']),
            output_activation='tanh')
    else:
        model = LocalSafetyNet(
            int(train.obs.shape[-1]),
            tuple(hcfg.get('hidden', (256, 256))))
    summary = train_safety_net(
        model, train, val,
        out=out,
        device=args.device,
        epochs=int(hcfg.get('epochs', 30)),
        batch_size=int(hcfg.get('batch_size', 512)),
        lr=float(hcfg.get('lr', 1e-3)),
        weight_decay=float(hcfg.get('weight_decay', 1e-4)),
        margin=float(
            args.margin if args.margin is not None
            else hcfg.get('margin', 0.05)),
        unsafe_weight=float(
            args.unsafe_weight if args.unsafe_weight is not None
            else hcfg.get('unsafe_weight', 1.0)),
        margin_coef=float(
            args.margin_coef if args.margin_coef is not None
            else hcfg.get('margin_coef', 0.5)),
        hdot_coef=float(
            args.hdot_coef if args.hdot_coef is not None
            else hcfg.get('hdot_coef', 0.0)),
        hdot_alpha=float(
            args.hdot_alpha if args.hdot_alpha is not None
            else hcfg.get('hdot_alpha', 1.0)),
        hdot_eps=float(
            args.hdot_eps if args.hdot_eps is not None
            else hcfg.get('hdot_eps', 0.02)),
        dt=float(hcfg.get('dt', 0.03)),
        invariance_coef=float(
            args.invariance_coef if args.invariance_coef is not None
            else hcfg.get('invariance_coef', 0.1)),
        alpha=float(hcfg.get('alpha', 0.1)),
        seed=int(config.get('seed', 0)),
    )
    device = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
              if args.device == 'auto' else torch.device(args.device))
    test_metrics = evaluate_safety(
        model, test, device, threshold=float(summary.get('threshold', 0.0)))
    summary['test'] = test_metrics
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(yaml.safe_dump({'summary': summary}, sort_keys=False))


if __name__ == '__main__':
    main()
