#!/usr/bin/env python3
"""Summarize the C1s actor1+h2 seed-replication diagnostic.

C1s is an evaluation-only, off-policy h-swap.  It puts the frozen C0 h2 seed-0
checkpoint into fixed Stage-B actor1 seeds 1 and 2 and compares each with that
actor's h1 baseline.  The primary test is the two new actor seeds; seed 0 is a
reference.

See ``outputs/v6/independent/nav/STAGE_C1S_H2_ACTOR_SEED_SWAP.md`` for the
pre-registered rule.  This is not a formal C0/C1 result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = (
    'safe_episode',
    'collision_episode',
    'warning_episode',
    'official_safety',
    'official_success',
    'official_reach',
    'episode_return',
    'h_mean',
    'h_min',
    'cbf_violation',
    'action_abs',
)

NEW_SEEDS = (1, 2)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())['metrics']


def _delta(cur: dict, base: dict) -> dict:
    return {k: float(cur[k] - base[k]) for k in KEYS if k in cur and k in base}


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-root', default=(
        'outputs/v6/independent/nav_c1s_h2_actor_seed_swap'))
    ap.add_argument('--baseline-root', default=(
        'outputs/v6/independent/nav_stageb_repair_16k'))
    ap.add_argument('--seed0-swap', default=(
        'outputs/v6/independent/nav_c0_h2_actor1_swap/seed0_100ep.json'))
    ap.add_argument('--out')
    args = ap.parse_args()

    out_root = Path(args.out_root)
    baseline_root = Path(args.baseline_root)

    per_seed = {}
    for seed in NEW_SEEDS:
        swap_path = out_root / f'seed{seed}' / 'eval_100ep_final.json'
        base_path = baseline_root / f'seed{seed}' / 'eval_100ep_final.json'
        if not swap_path.exists():
            raise FileNotFoundError(f'missing C1s eval: {swap_path}')
        if not base_path.exists():
            raise FileNotFoundError(f'missing h1 baseline: {base_path}')
        swap = _load(swap_path)
        base = _load(base_path)
        per_seed[seed] = dict(
            swap_path=str(swap_path.resolve()),
            baseline_path=str(base_path.resolve()),
            swap=swap,
            baseline=base,
            delta=_delta(swap, base),
        )

    mean_delta = {
        k: _mean([per_seed[s]['delta'][k] for s in NEW_SEEDS])
        for k in KEYS
        if all(k in per_seed[s]['delta'] for s in NEW_SEEDS)
    }

    checks = {
        'delta_safe_nonnegative_both': bool(all(
            per_seed[s]['delta']['safe_episode'] >= 0 for s in NEW_SEEDS)),
        'delta_official_safety_nonnegative_both': bool(all(
            per_seed[s]['delta']['official_safety'] >= 0
            for s in NEW_SEEDS)),
        'mean_delta_safe_ge_0p03': bool(
            mean_delta['safe_episode'] >= 0.03),
        'mean_delta_official_safety_ge_0p005': bool(
            mean_delta['official_safety'] >= 0.005),
        'mean_delta_success_ge_minus_0p0275': bool(
            mean_delta['official_success'] >= -0.0275),
        'mean_delta_reach_ge_minus_0p035': bool(
            mean_delta['official_reach'] >= -0.035),
        'mean_delta_return_ge_minus_2p91': bool(
            mean_delta['episode_return'] >= -2.91),
        'no_catastrophic_task_regression': bool(all(
            per_seed[s]['delta']['official_success'] >= -0.05
            and per_seed[s]['delta']['episode_return'] >= -5.0
            for s in NEW_SEEDS)),
    }
    consistent_positive = bool(all(checks.values()))

    positive_safety_seeds = [
        s for s in NEW_SEEDS
        if per_seed[s]['delta']['safe_episode'] > 0
        and per_seed[s]['delta']['official_safety'] > 0
    ]
    catastrophic = not checks['no_catastrophic_task_regression']
    if consistent_positive:
        classification = 'consistent_positive'
    elif positive_safety_seeds and not catastrophic:
        classification = 'mixed'
    else:
        classification = 'negative'

    seed0_reference = None
    seed0_path = Path(args.seed0_swap)
    base0_path = baseline_root / 'seed0' / 'eval_100ep_final.json'
    if seed0_path.exists() and base0_path.exists():
        swap0 = _load(seed0_path)
        base0 = _load(base0_path)
        seed0_reference = dict(
            swap_path=str(seed0_path.resolve()),
            baseline_path=str(base0_path.resolve()),
            swap=swap0,
            baseline=base0,
            delta=_delta(swap0, base0),
        )

    if seed0_reference is not None:
        three_seed_mean = {}
        for k in KEYS:
            if all(k in per_seed[s]['delta'] for s in NEW_SEEDS) \
                    and k in seed0_reference['delta']:
                three_seed_mean[k] = _mean([
                    seed0_reference['delta'][k],
                    per_seed[1]['delta'][k],
                    per_seed[2]['delta'][k],
                ])
    else:
        three_seed_mean = None

    if classification == 'consistent_positive':
        interpretation = (
            'the h2 seed-0 swap has a consistent non-negative safety signal '
            'on actor1 seeds 1/2 and does not violate the task margins; this '
            'is a robustness signal for h2 seed 0 as a frozen input, not a '
            'formal pass and not evidence that h2 is better in general')
    elif classification == 'mixed':
        interpretation = (
            'the h2 seed-0 swap signal is mixed across actor1 seeds 1/2; the '
            'seed-0 result is not robust enough to justify a matched h2-seed '
            'study')
    else:
        interpretation = (
            'the h2 seed-0 swap signal does not replicate on actor1 seeds 1/2 '
            'or has a catastrophic task regression; stop the h2 pairing line')

    result = dict(
        schema='v6.nav.c1s_h2_actor_seed_swap.v1',
        protocol=dict(
            description='no-training h2 seed-0 swap on fixed actor1 seeds 1/2',
            h_checkpoint='outputs/v6/independent/h_c0/seed0/v6_h_c0.pkl',
            episodes=100,
            eval_seed=10000,
            deterministic=True,
            gate_active_at_eval=False,
            off_policy=True,
        ),
        per_seed={str(s): per_seed[s] for s in NEW_SEEDS},
        mean_new_seeds=mean_delta,
        checks=checks,
        classification=classification,
        consistent_positive=consistent_positive,
        positive_safety_seeds=positive_safety_seeds,
        seed0_reference=seed0_reference,
        three_seed_mean=three_seed_mean,
        interpretation=interpretation,
        caveats=[
            'off-policy: actor1 seeds 1/2 were trained with h1',
            'h2 seed 0 was trained on actor1 seed 0 data; seeds 1/2 are '
            'distribution-shifted for h2',
            'n=3 including seed 0; the seed-0 difference is small and not '
            'statistically significant',
            'not a C0/C1 pass and not a from-scratch V6 result',
            'do not use to revive C1b or the from-scratch actor2 branch',
        ],
    )

    out = Path(args.out) if args.out else out_root / 'decision.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    print(f'saved: {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
