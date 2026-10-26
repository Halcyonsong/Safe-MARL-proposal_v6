"""Train the V6 Ant 9-D safety h(s) from V6-collected data."""

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

from v6.ant_h import AntSafetyDataset, AntSafetyNet, train_ant_safety_h  # noqa: E402


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _hidden(value) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(v) for v in value.split(',') if v.strip())
    return tuple(int(v) for v in value)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/v6/v6_ant_h.yaml')
    ap.add_argument('--data-dir')
    ap.add_argument('--out')
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--batch-size', type=int)
    ap.add_argument('--lr', type=float)
    ap.add_argument('--margin-coef', type=float)
    ap.add_argument('--grad-coef', type=float)
    ap.add_argument('--unsafe-weight', type=float)
    ap.add_argument('--h-scale', type=float)
    ap.add_argument('--kappa', type=float)
    ap.add_argument('--kappa-z', type=float)
    ap.add_argument('--horizon', type=int)
    ap.add_argument('--hidden')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = NAV / cfg_path
    config = yaml.safe_load(cfg_path.read_text())
    data_dir = _resolve(args.data_dir or config['data'])
    out = _resolve(args.out or config['out'])
    train_cfg = dict(config.get('train', {}))
    h_scale = float(args.h_scale if args.h_scale is not None
                    else config.get('h_scale', 0.5))
    horizon = int(args.horizon if args.horizon is not None
                  else config.get('horizon', 32))
    seed = int(args.seed if args.seed is not None else config.get('seed', 0))
    hidden = _hidden(args.hidden or config.get('model', {}).get(
        'hidden', [128, 128]))

    h_kappa = float(args.kappa if args.kappa is not None
                    else config.get('kappa', 0.0))
    h_kappa_z = float(args.kappa_z if args.kappa_z is not None
                      else config.get('kappa_z', 0.0))
    train_data = AntSafetyDataset.from_npz(
        data_dir / 'train_data.npz', h_kappa=h_kappa,
        h_kappa_z=h_kappa_z)
    val_data = AntSafetyDataset.from_npz(
        data_dir / 'val_data.npz', h_kappa=h_kappa,
        h_kappa_z=h_kappa_z)
    test_data = AntSafetyDataset.from_npz(
        data_dir / 'test_data.npz', h_kappa=h_kappa,
        h_kappa_z=h_kappa_z)
    print(f'train={len(train_data)} val={len(val_data)} test={len(test_data)}')
    print(f'h_scale={h_scale} kappa={h_kappa} kappa_z={h_kappa_z} '
          f'horizon={horizon} hidden={hidden}')

    model = AntSafetyNet(geom_dim=9, hidden=hidden)
    summary = train_ant_safety_h(
        model,
        train_data,
        val_data,
        test_data=test_data,
        out=out,
        device=args.device,
        epochs=int(args.epochs if args.epochs is not None
                   else train_cfg.get('epochs', 40)),
        batch_size=int(args.batch_size if args.batch_size is not None
                       else train_cfg.get('batch_size', 512)),
        lr=float(args.lr if args.lr is not None
                 else train_cfg.get('lr', 1e-3)),
        weight_decay=float(train_cfg.get('weight_decay', 1e-4)),
        margin=float(train_cfg.get('margin', 0.05)),
        margin_coef=float(args.margin_coef if args.margin_coef is not None
                          else train_cfg.get('margin_coef', 0.5)),
        grad_coef=float(args.grad_coef if args.grad_coef is not None
                        else train_cfg.get('grad_coef', 0.1)),
        unsafe_weight=float(args.unsafe_weight
                            if args.unsafe_weight is not None
                            else train_cfg.get('unsafe_weight', 1.0)),
        target_on_all=bool(train_cfg.get('target_on_all', False)),
        grad_on_all=bool(train_cfg.get('grad_on_all', False)),
        max_grad_norm=float(train_cfg.get('max_grad_norm', 1.0)),
        h_scale=h_scale,
        h_kappa=h_kappa,
        h_kappa_z=h_kappa_z,
        horizon=horizon,
        seed=seed,
    )
    resolved = dict(
        schema='v6.ant_h.train.config.v1',
        config_path=str(cfg_path),
        data_dir=str(data_dir),
        out=str(out),
        hidden=list(hidden),
        h_scale=h_scale,
        h_kappa=h_kappa,
        h_kappa_z=h_kappa_z,
        horizon=horizon,
        seed=seed,
        target_on_all=bool(train_cfg.get('target_on_all', False)),
        grad_on_all=bool(train_cfg.get('grad_on_all', False)),
        train_cfg=train_cfg,
        summary=summary,
    )
    (out / 'resolved_config.json').write_text(
        json.dumps(resolved, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
