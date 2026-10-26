#!/usr/bin/env bash
# Reproducible V5 QP-free protocol: 3 training seeds + fixed-scene evaluation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/nav"

SEEDS="${SEEDS:-0 1 2}"
CONFIG="${CONFIG:-configs/v5/qp_free_double_integrator_safety.yaml}"
TRAIN_STEPS="${TRAIN_STEPS:-16384}"
EPISODES="${EPISODES:-100}"
EVAL_SEED="${EVAL_SEED:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2048}"
SELECT_EPISODES="${SELECT_EPISODES:-5}"
OUT="${OUT:-outputs/v5/protocol}"

mkdir -p "$ROOT/$OUT"
for seed in $SEEDS; do
  echo "=== train seed $seed ==="
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
  python scripts/train_qp_free.py \
    --config "$CONFIG" \
    --seed "$seed" \
    --total-steps "$TRAIN_STEPS" \
    --rollout-steps 128 \
    --eval-interval "$EVAL_INTERVAL" \
    --eval-episodes "$SELECT_EPISODES" \
    --save-interval 0 \
    --skip-final-eval \
    --device cpu \
    --out "../$OUT/seed${seed}"
  echo "=== eval seed $seed ==="
  CKPT="../$OUT/seed${seed}/policy_best.pth"
  if [ ! -f "$CKPT" ]; then
    CKPT="../$OUT/seed${seed}/policy_final.pth"
  fi
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
  python scripts/evaluate_qp_free.py \
    --checkpoint "$CKPT" \
    --num-episodes "$EPISODES" \
    --seed "$EVAL_SEED" \
    --device cpu \
    --out "../$OUT/seed${seed}/eval_${EPISODES}ep.json"
done

python scripts/aggregate_qp_free.py \
  --pattern "../$OUT/seed*/eval_${EPISODES}ep.json" \
  --out "../$OUT/aggregate_${EPISODES}ep.json"
echo "saved: $ROOT/$OUT"
