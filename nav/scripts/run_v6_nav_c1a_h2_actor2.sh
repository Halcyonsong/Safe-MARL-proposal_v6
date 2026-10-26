#!/usr/bin/env bash
# V6 navigation C1a: actor2 from scratch with C0 h2 seed 0 (seed-0 pilot).
#
# This is a new, explicitly approved pilot outside the original C0/C1 gate.
# It trains actor2 from scratch with the frozen h2 checkpoint, then runs the
# formal 100-episode deterministic evaluation (eval seed 10000).
#
# It does NOT train h, does NOT warm-start from actor1, and does NOT change
# the gate threshold, cost schedule, task channel, or action space.
#
# Full pilot:
#   CONFIRM=1 bash nav/scripts/run_v6_nav_c1a_h2_actor2.sh
# Smoke (pipeline only, no claim):
#   SMOKE=1 bash nav/scripts/run_v6_nav_c1a_h2_actor2.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SEED="${SEED:-0}"
CONFIG="${CONFIG:-configs/v6/v6_nav_c1a_h2_actor2.yaml}"
H="${H:-outputs/v6/independent/h_c0/seed0/v6_h_c0.pkl}"
OUT="${OUT:-outputs/v6/independent/nav_c1a_h2_actor2/seed${SEED}}"
BASELINE="${BASELINE:-outputs/v6/independent/nav_stageb_repair_16k/seed0/eval_100ep_final.json}"
STEPS="${STEPS:-16384}"
EPISODES="${EPISODES:-100}"
EVAL_SEED="${EVAL_SEED:-10000}"
SMOKE="${SMOKE:-0}"
CONFIRM="${CONFIRM:-0}"

if [ "$SMOKE" = "1" ]; then
  # Smoke must never inherit the full-run defaults above.
  STEPS="${SMOKE_STEPS:-256}"
  EPISODES="${SMOKE_EPISODES:-2}"
  EVAL_INTERVAL="${SMOKE_EVAL_INTERVAL:-128}"
  EVAL_EPISODES="${SMOKE_EVAL_EPISODES:-1}"
  SAVE_INTERVAL=0
else
  if [ "$CONFIRM" != "1" ]; then
    echo "Refusing the C1a pilot without CONFIRM=1." >&2
    echo "Use SMOKE=1 for a pipeline check." >&2
    exit 2
  fi
  EVAL_INTERVAL="${EVAL_INTERVAL:-8192}"
  EVAL_EPISODES="${EVAL_EPISODES:-20}"
  SAVE_INTERVAL="${SAVE_INTERVAL:-4096}"
fi

if [ ! -f "$H" ]; then
  echo "missing h2 checkpoint: $H" >&2
  exit 1
fi

mkdir -p "$OUT"
if [ -f "$OUT/policy_final.pth" ] && [ "$SMOKE" != "1" ]; then
  echo "refusing to overwrite existing actor2 checkpoint: $OUT/policy_final.pth" >&2
  echo "move it aside or set OUT to a new directory" >&2
  exit 1
fi

echo "C1a actor2: seed=$SEED steps=$STEPS episodes=$EPISODES smoke=$SMOKE"
echo "h2: $H"
echo "out: $OUT"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] train actor2 seed=$SEED from scratch"
"$PYTHON" -u nav/scripts/train_v6.py \
  --config "$CONFIG" \
  --out "$OUT" \
  --seed "$SEED" \
  --device cuda \
  --total-steps "$STEPS" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-episodes "$EVAL_EPISODES" \
  --eval-seed "$EVAL_SEED" \
  --save-interval "$SAVE_INTERVAL" \
  > "$OUT/train.log" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] 100ep deterministic evaluation"
"$PYTHON" -u nav/scripts/evaluate_v6.py \
  --checkpoint "$OUT/policy_final.pth" \
  --h-checkpoint "$H" \
  --num-episodes "$EPISODES" \
  --seed "$EVAL_SEED" \
  --device cuda \
  --out "$OUT/eval_100ep_final.json" \
  > "$OUT/eval_100ep.log" 2>&1

if [ "$SMOKE" != "1" ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] apply C1a pilot rule"
  "$PYTHON" -u nav/scripts/summarize_v6_nav_c1a.py \
    --eval "$OUT/eval_100ep_final.json" \
    --baseline "$BASELINE" \
    --out "$OUT/pilot_decision.json" \
    > "$OUT/pilot_decision.log" 2>&1
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done: $OUT"
