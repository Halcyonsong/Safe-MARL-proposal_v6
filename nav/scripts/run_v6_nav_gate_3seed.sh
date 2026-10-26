#!/usr/bin/env bash
# Evaluate and aggregate the V6 navigation predictive-h task-gate experiment.
#
# Training is intentionally separate: this script only evaluates existing
# final checkpoints, using the matching V6 h and fixed eval seed.  It is
# resumable and rejects missing/incomplete runs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/nav"

PYTHON="${PYTHON:-python}"
OUT="${OUT:-$ROOT/outputs/v6/independent/nav_h_gate_16k}"
case "$OUT" in
  /*) ;;
  *) OUT="$ROOT/$OUT" ;;
esac
H_CHECKPOINT="${H_CHECKPOINT:-$ROOT/outputs/v6/independent/h/v6_h.pkl}"
SEEDS="${SEEDS:-0 1 2}"
EPISODES="${EPISODES:-100}"
EVAL_SEED="${EVAL_SEED:-10000}"
DEVICE="${DEVICE:-cuda}"

"$PYTHON" scripts/check_v6_independence.py

for seed in $SEEDS; do
  run="$OUT/seed$seed"
  ckpt="$run/policy_final.pth"
  eval_file="$run/eval_${EPISODES}ep_final.json"
  if [ ! -f "$ckpt" ]; then
    echo "missing final checkpoint: $ckpt (train seed $seed first)" >&2
    exit 2
  fi
  if [ ! -f "$eval_file" ]; then
    echo "=== V6 nav gate ${EPISODES}ep eval seed=$seed ==="
    "$PYTHON" scripts/evaluate_v6.py \
      --checkpoint "$ckpt" \
      --h-checkpoint "$H_CHECKPOINT" \
      --num-episodes "$EPISODES" \
      --seed "$EVAL_SEED" \
      --device "$DEVICE" \
      --out "$eval_file" 2>&1 | tee "$run/eval_${EPISODES}ep.log"
  fi
done

"$PYTHON" scripts/summarize_v6_formal.py \
  --dir "$OUT" \
  --out "$OUT/aggregate_${EPISODES}ep.json"
