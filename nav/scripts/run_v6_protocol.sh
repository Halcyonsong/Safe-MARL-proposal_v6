#!/usr/bin/env bash
# Standalone V6 protocol: V6 self-collected h data -> V6 h -> from-scratch actor.
#
# Main experiment is h-only, no task channel, no QP/projection, no official h,
# and no V5/V4 weights or data.  Default is a single-seed pilot; pass
# SEEDS="0 1 2" and EPISODES=100 for the formal comparison after the pilot.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/nav"

CONFIG="${CONFIG:-configs/v6/v6_nav_independent.yaml}"
H_DATA="${H_DATA:-$ROOT/outputs/v6/independent/data}"
H_OUT="${H_OUT:-$ROOT/outputs/v6/independent/h}"
OUT="${OUT:-$ROOT/outputs/v6/independent/runs}"
SEEDS="${SEEDS:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-8192}"
EPISODES="${EPISODES:-20}"
EVAL_SEED="${EVAL_SEED:-10000}"
DEVICE="${DEVICE:-auto}"
H_CHECKPOINT="$H_OUT/v6_h.pkl"

mkdir -p "$H_DATA" "$H_OUT" "$OUT"
python scripts/check_v6_independence.py

# 1) V6-owned h data: random behavior + near-boundary augmentation.
if [ ! -f "$H_DATA/train_data.npz" ]; then
  python scripts/collect_v6_h_data.py \
    --config "$CONFIG" --out "$H_DATA" \
    --episodes 100 --steps 256 --seed 2026
fi

if [ ! -f "$H_DATA/train_aug.npz" ]; then
  python scripts/augment_v6_h_data.py \
    --config "$CONFIG" --data-dir "$H_DATA" \
    --out "$H_DATA/train_aug.npz" --scenes 8000 --seed 777
fi

if [ ! -f "$H_DATA/train_merged.npz" ]; then
  python scripts/merge_v6_h_data.py \
    --data-dir "$H_DATA" --out "$H_DATA/train_merged.npz"
fi

# 2) V6 h from scratch; threshold calibrated on V6 validation only.
if [ ! -f "$H_CHECKPOINT" ]; then
  python scripts/train_v6_official_h.py \
    --config "$CONFIG" \
    --train-data "$H_DATA/train_merged.npz" \
    --val-data "$H_DATA/val_data.npz" \
    --test-data "$H_DATA/test_data.npz" \
    --out "$H_OUT" --tag v6_h
fi

# 3) From-scratch actor: h-on main arm and matched h-off control.
for seed in $SEEDS; do
  for arm in h no_h; do
    RUN_DIR="$OUT/seed${seed}_${arm}"
    EXTRA=""
    if [ "$arm" = "no_h" ]; then
      EXTRA="--no-use-h-features"
    fi
    echo "=== V6 train seed=$seed arm=$arm ==="
    python scripts/train_v6.py \
      --config "$CONFIG" \
      --h-checkpoint "$H_CHECKPOINT" \
      --seed "$seed" \
      --total-steps "$TRAIN_STEPS" \
      --rollout-steps 128 \
      --eval-interval 2048 \
      --eval-episodes "$EPISODES" \
      --save-interval 0 \
      --skip-final-eval \
      --device "$DEVICE" \
      --out "$RUN_DIR" \
      $EXTRA

    echo "=== V6 eval seed=$seed arm=$arm ==="
    python scripts/evaluate_v6.py \
      --checkpoint "$RUN_DIR/policy_final.pth" \
      --h-checkpoint "$H_CHECKPOINT" \
      --num-episodes "$EPISODES" \
      --seed "$EVAL_SEED" \
      --device "$DEVICE" \
      --out "$RUN_DIR/eval_${EPISODES}ep.json"
  done
done

OUT_ABS="$OUT" python - <<'PY'
import glob
import json
import os
from pathlib import Path

files = sorted(glob.glob(os.environ['OUT_ABS'] + '/seed*/eval_*ep.json'))
rows = [json.loads(Path(f).read_text()) for f in files]
if not rows:
    raise SystemExit('no eval files')
print(json.dumps(rows, indent=2))
PY
