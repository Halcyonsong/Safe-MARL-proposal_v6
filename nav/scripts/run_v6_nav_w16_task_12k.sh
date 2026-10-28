#!/usr/bin/env bash
# V6 navigation W16 task warm-start: 16k -> 28k, task channel on.
#
# Procedure-level task follow-up to the A-only 28k budget extension.  The
# base is the seed-0 16k checkpoint from the same 28k run, so the comparison
# to the from-scratch 24k/28k checkpoints has the same seed and 16k prefix.
# No C16/C24 control is run; the T0w 16k+4k no-V_T arm remains the channel
# attribution evidence.
#
# Full run:
#   CONFIRM=1 bash nav/scripts/run_v6_nav_w16_task_12k.sh
# Smoke:
#   SMOKE=1 bash nav/scripts/run_v6_nav_w16_task_12k.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/xiang/miniconda3/envs/safe_marl/bin/python}"
SEED="${SEED:-0}"
CONFIG="${CONFIG:-configs/v6/v6_nav_w16_task_12k.yaml}"
RESUME="${RESUME:-outputs/v6/independent/nav_stageb_repair_28k/seed0/policy_step_16384.pth}"
H="${H:-outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl}"
VT="${VT:-outputs/v6/independent/t0w_vt/v6_task_value.pkl}"
OUT="${OUT:-outputs/v6/independent/nav_w16_task_12k/seed${SEED}}"
TOTAL_STEPS="${TOTAL_STEPS:-28672}"
EPISODES="${EPISODES:-100}"
EVAL_SEED="${EVAL_SEED:-10000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-4096}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
SAVE_INTERVAL="${SAVE_INTERVAL:-4096}"
SMOKE="${SMOKE:-0}"
CONFIRM="${CONFIRM:-0}"

if [ "$SMOKE" = "1" ]; then
  TOTAL_STEPS="${SMOKE_TOTAL_STEPS:-16896}"
  EPISODES=1
  EVAL_INTERVAL=128
  EVAL_EPISODES=1
  SAVE_INTERVAL=0
  OUT="${SMOKE_OUT:-/tmp/opencode/w16_task_smoke}"
  rm -rf "$OUT"
else
  if [ "$CONFIRM" != "1" ]; then
    echo "Refusing the W16 task warm-start without CONFIRM=1." >&2
    echo "Use SMOKE=1 for a pipeline check." >&2
    exit 2
  fi
  if [ -f "$OUT/policy_final.pth" ]; then
    echo "refusing to overwrite existing checkpoint: $OUT/policy_final.pth" >&2
    exit 1
  fi
fi

for path in "$RESUME" "$H" "$VT"; do
  if [ ! -f "$path" ]; then
    echo "missing required checkpoint: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUT"
echo "W16 task warm-start: seed=$SEED total_steps=$TOTAL_STEPS smoke=$SMOKE"
echo "resume: $RESUME"
echo "h: $H"
echo "V_T: $VT"
echo "out: $OUT"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] continue 16k checkpoint with frozen V_T"
"$PYTHON" -u nav/scripts/train_v6.py \
  --config "$CONFIG" \
  --out "$OUT" \
  --resume "$RESUME" \
  --task-value-checkpoint "$VT" \
  --freeze-task-value \
  --zero-init-task-block \
  --seed "$SEED" \
  --device cuda \
  --total-steps "$TOTAL_STEPS" \
  --eval-interval "$EVAL_INTERVAL" \
  --eval-episodes "$EVAL_EPISODES" \
  --eval-seed "$EVAL_SEED" \
  --save-interval "$SAVE_INTERVAL" \
  > "$OUT/train.log" 2>&1

if [ "$SMOKE" = "1" ]; then
  echo "smoke training done; evaluating one episode"
  "$PYTHON" -u nav/scripts/evaluate_v6.py \
    --checkpoint "$OUT/policy_final.pth" --h-checkpoint "$H" \
    --num-episodes "$EPISODES" --seed "$EVAL_SEED" --device cuda \
    --out "$OUT/eval_smoke.json" > "$OUT/eval_smoke.log" 2>&1
  echo "smoke done: $OUT"
  exit 0
fi

if [ "${SKIP_EVAL:-0}" = "1" ]; then
  echo "SKIP_EVAL=1: training done; formal evaluation deferred"
  exit 0
fi

for step in 24576 28672; do
  if [ "$step" = "28672" ]; then
    TAG="eval_100ep_final"
  else
    TAG="eval_100ep_step${step}"
  fi
  CKPT="$OUT/policy_step_${step}.pth"
  if [ ! -f "$CKPT" ]; then
    echo "missing W16 checkpoint: $CKPT" >&2
    exit 1
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${EPISODES}ep evaluation step $step"
  CHECKPOINT="$CKPT" H="$H" OUT="$OUT" TAG="$TAG" \
    PARALLEL=3 PYTHON="$PYTHON" \
    bash nav/scripts/eval_v6_nav_100ep_chunked.sh \
    > "$OUT/eval_${TAG}.log" 2>&1
done

CONTROL="outputs/v6/independent/nav_stageb_repair_28k/seed${SEED}"
"$PYTHON" -u nav/scripts/summarize_v6_nav_w16.py \
  --w16-24k "$OUT/eval_100ep_step24576.json" \
  --w16-28k "$OUT/eval_100ep_final.json" \
  --control-24k "$CONTROL/eval_100ep_step24576.json" \
  --control-28k "$CONTROL/eval_100ep_final.json" \
  --out "$OUT/screen_decision.json" \
  > "$OUT/screen_decision.log" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] W16 task warm-start done: $OUT"
