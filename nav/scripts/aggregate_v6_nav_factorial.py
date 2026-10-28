"""Aggregate the Nav P0 h-quality row and the P1 repaired-h 2x2 factorial.

Rows (100 deterministic episodes, seeds 0/1/2):
  P0  original h + gate ON                    nav_h_gate_16k
  A   repaired h + gate ON (existing row)     nav_stageb_repair_16k
  B   repaired h + gate OFF                   nav_repair_factorial/B_h_on_gate_off
  C   no actor h input + gate ON              nav_repair_factorial/C_h_off_gate_on
  D   no actor h input + gate OFF             nav_repair_factorial/D_h_off_gate_off

The h network (where the gate is active) is always the same repaired V6 h in
A/B/C/D.  C/D zero the actor's h/grad_h channels; the gate still reads the
frozen h's h_next, so C is "gate-only, no h actor input".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (0, 1, 2)
KEYS = [
    'safe_episode', 'collision_episode', 'official_safety',
    'official_success', 'official_reach', 'episode_return',
]
LABELS = {
    'safe_episode': 'safe_ep',
    'collision_episode': 'collision_ep',
    'official_safety': 'official_safety',
    'official_success': 'success',
    'official_reach': 'reach',
    'episode_return': 'return',
}

ROWS = [
    dict(id='H0', h='original v6_h.pkl', h_input=True, gate=False,
         dir='outputs/v6/independent/formal_3seed/h_only',
         note='existing formal h-only row, not retrained; reference'),
    dict(id='P0', h='original v6_h.pkl', h_input=True, gate=True,
         dir='outputs/v6/independent/nav_h_gate_16k',
         note='original-h + broad gate, 3-seed completion'),
    dict(id='A', h='repaired', h_input=True, gate=True,
         dir='outputs/v6/independent/nav_stageb_repair_16k',
         note='existing promoted Stage-B row, not retrained'),
    dict(id='B', h='repaired', h_input=True, gate=False,
         dir='outputs/v6/independent/nav_repair_factorial/B_h_on_gate_off',
         note='h-input effect with gate off'),
    dict(id='C', h='repaired', h_input=False, gate=True,
         dir='outputs/v6/independent/nav_repair_factorial/C_h_off_gate_on',
         note='gate effect without actor h input'),
    dict(id='D', h='repaired', h_input=False, gate=False,
         dir='outputs/v6/independent/nav_repair_factorial/D_h_off_gate_off',
         note='V6-code no-h/no-gate cell'),
]


def _read_seed_dir(path: Path, seed: int):
    eval_path = path / f'seed{seed}' / 'eval_100ep_final.json'
    if not eval_path.exists():
        return None
    payload = json.loads(eval_path.read_text())
    metrics = payload.get('metrics', {})
    return {
        'source': str(eval_path.relative_to(ROOT)),
        'episodes': int(payload.get('episodes', 0)),
        'eval_seed': int(payload.get('eval_seed', -1)),
        **{key: float(metrics[key]) for key in KEYS},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=(
        'outputs/v6/independent/nav_repair_factorial/aggregate_100ep.json'))
    ap.add_argument('--markdown', default=(
        'outputs/v6/independent/nav_repair_factorial/aggregate_100ep.md'))
    args = ap.parse_args()

    out = ROOT / args.out
    md = ROOT / args.markdown
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in ROWS:
        seed_rows = [_read_seed_dir(ROOT / spec['dir'], s) for s in SEEDS]
        complete = all(row is not None for row in seed_rows)
        aggregate = {}
        if complete:
            for key in KEYS:
                values = np.array([row[key] for row in seed_rows])
                aggregate[key] = dict(
                    mean=float(values.mean()),
                    std=float(values.std(ddof=1)),
                    values=[float(v) for v in values],
                )
        rows.append(dict(
            **{k: v for k, v in spec.items() if k != 'dir'},
            directory=spec['dir'],
            complete=complete,
            seeds=dict(zip((str(s) for s in SEEDS), seed_rows)),
            aggregate=aggregate,
        ))

    payload = dict(
        protocol=dict(
            environment='official GCBF+ DoubleIntegrator',
            num_agents=8, area_size=4.0, max_step=256,
            steps=16384, training_seeds=[0, 1, 2],
            episodes=100, eval_seed=10000,
            deterministic=True,
            fixed_unsafe_penalty=10.0,
            repaired_h=(
                'outputs/v6/independent/h_stageb_repair/'
                'v6_h_stageb_repair.pkl'),
            original_h='outputs/v6/independent/h/v6_h.pkl',
            gate_high_repaired=0.26726993918418884,
            gate_high_original=0.13,
        ),
        rows=rows,
        warnings=[
            'P0 uses the original h; A/B/C/D use the repaired h. P0 is the '
            'h-quality control, not a cell of the repaired-h 2x2.',
            'C/D call --no-use-h-features, which zeros the policy h/grad_h '
            'channels (shared actor/value trunk). The gate in C still uses '
            'the frozen repaired h h_next. Both cells collapse, and D is '
            'consistent with the matched no-h fixed-penalty baseline.',
            'The gate is disabled at evaluation in every row; gate ON/OFF is '
            'a training-time reward-shaping difference.',
        ],
    )
    out.write_text(json.dumps(payload, indent=2) + '\n')

    lines = [
        '# Nav P0 + P1 factorial aggregate',
        '',
        '100 deterministic episodes per seed, eval scenes 10000..10099; '
        'mean ± sample std (n-1) over seeds 0/1/2.',
        '',
        '| row | h checkpoint | actor h input | gate | complete | '
        + ' | '.join(LABELS[k] for k in KEYS) + ' |',
        '|---|---|:---:|:---:|:---:|' + '---:|' * len(KEYS),
    ]
    for row in rows:
        label = row['id']
        if row['complete']:
            vals = ' | '.join(
                f"{row['aggregate'][k]['mean']:.4f} ± "
                f"{row['aggregate'][k]['std']:.4f}" for k in KEYS)
        else:
            vals = ' | '.join('TBD' for _ in KEYS)
        lines.append(
            f"| {label} | {row['h']} | "
            f"{'on' if row['h_input'] else 'off'} | "
            f"{'on' if row['gate'] else 'off'} | "
            f"{'yes' if row['complete'] else 'no'} | {vals} |")
    lines += [
        '',
        'Row definitions:',
        '',
        '- **H0**: original `v6_h.pkl`, actor h input on, gate off '
        '(existing formal h-only reference).',
        '- **P0**: original `v6_h.pkl`, actor h input on, broad gate '
        '(high `0.13`).',
        '- **A**: repaired h, actor h input on, gate high `0.26727` '
        '(existing `nav_stageb_repair_16k`).',
        '- **B**: repaired h, actor h input on, gate off.',
        '- **C**: repaired h frozen for the gate, actor h/grad channels '
        'zeroed, gate on.',
        '- **D**: repaired h frozen for the gate, actor h/grad channels '
        'zeroed, gate off.',
        '',
        'Caveats:',
        '',
        '- **H0 vs B is the clean h-retrain effect with the gate off** (both '
        'use policy h input and no training-time gate): `+18.3 pp` safe but '
        '`-8.4 pp` success and `-6.1` return, so repaired h alone is a '
        'safety/task trade.',
        '- **C/D collapse**: removing policy h input gives safe `0.020/0.033` '
        'and success `0.037` in both cells across all six seeds; D is '
        'consistent with the matched no-h fixed-penalty baseline.  The gate '
        'cannot compensate (C - D safe `-1.3 pp`, success `0.0 pp`).',
        '- P0 differs from A in h checkpoint, gate threshold, and gate scope; '
        'it completes the original-h broad-gate row, not a single factor of '
        'the repaired-h 2x2.  P0 vs H0 isolates the broad-gate effect on the '
        'original h only.',
        '- The gate is off at evaluation, so gate ON/OFF is a training-time '
        'reward-shaping difference, not an inference-time filter.',
        '- B vs D isolates the actor h-input effect with the gate off; '
        'A vs C isolates it with the gate on; A vs B and C vs D isolate the '
        'gate effect at each h-input level.',
    ]
    md.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
