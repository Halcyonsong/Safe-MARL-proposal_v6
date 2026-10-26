#!/usr/bin/env bash
# V6 navigation C1w: 4k warm-start of actor1 seed 0 with C0 h2 seed 0.
#
# Diagnostic adaptation arm only: no h1-continued control, so changes cannot
# be attributed to h2 rather than to the extra 4,096 training steps.  It is
# not a from-scratch V6 result and not a formal C0/C1 claim.
#
# Full run:
#   CONFIRM=1 bash nav/scripts/run_v6_nav_c1w_h2_actor1_4k.sh
# Smoke (pipeline only, no claim):
#   SMOKE=1 bash nav/scripts/run_v6_nav_c1w_h2_actor1_4k.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SEED="${SEED:-0}"
CONFIG="${CONFIG:-configs/v6/v6_nav_c1w_h2_actor1_4k.yaml}"
RESUME="${RESUME:-outputs/v6/independent/nav_stageb_repair_16k/seed0/policy_final.pth}"
H="${H:-outputs/v6/independent/h_c0/seed0/v6_h_c0.pkl}"
OUT="${OUT:-outputs/v6/independent/nav_c1w_h2_actor1_4k/seed${SEED}}"
BASELINE="${BASELINE:-outputs/v6/independent/nav_stageb_repair_16k/seed0/eval_100ep_final.json}"
SWAP="${SWAP:-outputs/v6/independent/nav_c0_h2_actor1_swap/seed0_100ep.json}"
TOTAL_STEPS="${TOTAL_STEPS:-20480}"
EPISODES="${EPISODES:-100}"
EVAL_SEED="${EVAL_SEED:-10000}"
SMOKE="${SMOKE:-0}"
CONFIRM="${CONFIRM:-0}"

if [ "$SMOKE" = "1" ]; then
  # Smoke must never inherit the full-run defaults above.
  TOTAL_STEPS="${SMOKE_TOTAL_STEPS:-16512}"
  EPISODES="${SMOKE_EPISODES:-2}"
  EVAL_INTERVAL="${SMOKE_EVAL_INTERVAL:-128}"
  EVAL_EPISODES="${SMOKE_EVAL_EPISODES:-1}"
  SAVE_INTERVAL=0
else
  if [ "$CONFIRM" != "1" ]; then
    echo "Refusing the C1w warm-start without CONFIRM=1." >&2
    echo "Use SMOKE=1 for a pipeline check." >&2
    exit 2
  fi
  EVAL_INTERVAL="${EVAL_INTERVAL:-2048}"
  EVAL_EPISODES="${EVAL_EPISODES:-20}"
  SAVE_INTERVAL="${SAVE_INTERVAL:-2048}"
fi

if [ ! -f "$H" ]; then
  echo "missing h2 checkpoint: $H" >&2
  exit 1
fi
if [ ! -f "$RESUME" ]; then
  echo "missing actor1 resume checkpoint: $RESUME" >&2
  exit 1
fi

mkdir -p "$OUT"
if [ -f "$OUT/policy_final.pth" ] && [ "$SMOKE" != "1" ]; then
  echo "refusing to overwrite existing checkpoint: $OUT/policy_final.pth" >&2
  echo "move it aside or set OUT to a new directory" >&2
  exit 1
fi

echo "C1w warm-start: seed=$SEED total_steps=$TOTAL_STEPS episodes=$EPISODES smoke=$SMOKE"
echo "resume: $RESUME"
echo "h2: $H"
echo "out: $OUT"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] continue actor1 seed=$SEED with h2"
"$PYTHON" -u nav/scripts/train_v6.py \
  --config "$CONFIG" \
  --out "$OUT" \
  --resume "$RESUME" \
  --seed "$SEED" \
  --device cuda \
  --total-steps "$TOTAL_STEPS" \
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
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] apply C1w diagnostic summary"
  "$PYTHON" -u nav/scripts/summarize_v6_nav_c1w.py \
    --eval "$OUT/eval_100ep_final.json" \
    --baseline "$BASELINE" \
    --swap "$SWAP" \
    --out "$OUT/diagnostic_decision.json" \
    > "$OUT/diagnostic_decision.log" 2>&1
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done: $OUT"
