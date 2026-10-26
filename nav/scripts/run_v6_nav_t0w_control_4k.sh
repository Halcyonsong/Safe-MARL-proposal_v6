#!/usr/bin/env bash
# V6 navigation T0w matched no-V_T 4k control.
#
# Diagnostic only.  Same actor1 seed-0 init, same 4,096 extra steps, same
# h1/gate/cost/reward/optimizer/seed and same 100ep eval protocol as the T0w
# V_T warm-start; the 5-D task feature is held at zero and no V_T checkpoint
# is loaded.  Run only after the warm-start showed a task benefit.
#
# Full run:
#   CONFIRM=1 bash nav/scripts/run_v6_nav_t0w_control_4k.sh
# Smoke (pipeline only, no claim):
#   SMOKE=1 bash nav/scripts/run_v6_nav_t0w_control_4k.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SEED="${SEED:-0}"
CONFIG="${CONFIG:-configs/v6/v6_nav_t0w_control_4k.yaml}"
RESUME="${RESUME:-outputs/v6/independent/nav_stageb_repair_16k/seed0/policy_final.pth}"
H="${H:-outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl}"
WARMSTART="${WARMSTART:-outputs/v6/independent/nav_t0w_vt_warmstart_4k/seed0/eval_100ep_final.json}"
OUT="${OUT:-outputs/v6/independent/nav_t0w_control_4k/seed${SEED}}"
TOTAL_STEPS="${TOTAL_STEPS:-20480}"
EPISODES="${EPISODES:-100}"
EVAL_SEED="${EVAL_SEED:-10000}"
SMOKE="${SMOKE:-0}"
CONFIRM="${CONFIRM:-0}"

if [ "$SMOKE" = "1" ]; then
  TOTAL_STEPS="${SMOKE_TOTAL_STEPS:-16512}"
  EPISODES="${SMOKE_EPISODES:-1}"
  EVAL_INTERVAL="${SMOKE_EVAL_INTERVAL:-128}"
  EVAL_EPISODES="${SMOKE_EVAL_EPISODES:-1}"
  SAVE_INTERVAL=0
  OUT="${SMOKE_OUT:-/tmp/opencode/t0w_control_smoke}"
else
  if [ "$CONFIRM" != "1" ]; then
    echo "Refusing the T0w control without CONFIRM=1." >&2
    echo "Use SMOKE=1 for a pipeline check." >&2
    exit 2
  fi
  EVAL_INTERVAL="${EVAL_INTERVAL:-2048}"
  EVAL_EPISODES="${EVAL_EPISODES:-20}"
  SAVE_INTERVAL="${SAVE_INTERVAL:-2048}"
fi

for path in "$RESUME" "$H"; do
  if [ ! -f "$path" ]; then
    echo "missing required checkpoint: $path" >&2
    exit 1
  fi
done

if [ "$SMOKE" != "1" ] && [ -f "$OUT/policy_final.pth" ]; then
  echo "refusing to overwrite existing checkpoint: $OUT/policy_final.pth" >&2
  echo "move it aside or set OUT to a new directory" >&2
  exit 1
fi

mkdir -p "$OUT"
echo "T0w no-V_T control: seed=$SEED total_steps=$TOTAL_STEPS episodes=$EPISODES smoke=$SMOKE"
echo "resume: $RESUME"
echo "h1: $H"
echo "task feature: zero / off"
echo "out: $OUT"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] continue actor1 with zero task feature"
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

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${EPISODES}ep deterministic evaluation"
"$PYTHON" -u nav/scripts/evaluate_v6.py \
  --checkpoint "$OUT/policy_final.pth" \
  --h-checkpoint "$H" \
  --num-episodes "$EPISODES" \
  --seed "$EVAL_SEED" \
  --device cuda \
  --out "$OUT/eval_100ep_final.json" \
  > "$OUT/eval_100ep.log" 2>&1

if [ "$SMOKE" != "1" ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] apply matched-control rule"
  "$PYTHON" -u nav/scripts/summarize_v6_nav_t0w_control.py \
    --control "$OUT/eval_100ep_final.json" \
    --warmstart "$WARMSTART" \
    --baseline outputs/v6/independent/nav_stageb_repair_16k/seed0/eval_100ep_final.json \
    --out "$OUT/diagnostic_decision.json" \
    > "$OUT/diagnostic_decision.log" 2>&1
  echo "  decision: $OUT/diagnostic_decision.json"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done: $OUT"
