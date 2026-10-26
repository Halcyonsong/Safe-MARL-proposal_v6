#!/usr/bin/env bash
# V6 navigation C1s: no-training h2 seed-0 swap on actor1 seeds 1 and 2.
#
# Evaluation only.  No actor training, no h training, no gate change.  The
# pre-registered protocol is in
# outputs/v6/independent/nav/STAGE_C1S_H2_ACTOR_SEED_SWAP.md.
#
# Full run:
#   CONFIRM=1 bash nav/scripts/run_v6_nav_c1s_h2_actor_seed_swap.sh
# Smoke (pipeline only, no claim):
#   SMOKE=1 bash nav/scripts/run_v6_nav_c1s_h2_actor_seed_swap.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
H="${H:-outputs/v6/independent/h_c0/seed0/v6_h_c0.pkl}"
BASELINE_ROOT="${BASELINE_ROOT:-outputs/v6/independent/nav_stageb_repair_16k}"
OUT_ROOT="${OUT_ROOT:-outputs/v6/independent/nav_c1s_h2_actor_seed_swap}"
EVAL_SEED="${EVAL_SEED:-10000}"
SMOKE="${SMOKE:-0}"
CONFIRM="${CONFIRM:-0}"

if [ "$SMOKE" = "1" ]; then
  SEEDS="${SMOKE_SEEDS:-1}"
  EPISODES="${SMOKE_EPISODES:-2}"
  OUT_ROOT="${SMOKE_OUT_ROOT:-/tmp/opencode/c1s_smoke}"
else
  if [ "$CONFIRM" != "1" ]; then
    echo "Refusing the C1s full evaluation without CONFIRM=1." >&2
    echo "Use SMOKE=1 for a pipeline check." >&2
    exit 2
  fi
  SEEDS="${SEEDS:-1 2}"
  EPISODES="${EPISODES:-100}"
fi

if [ ! -f "$H" ]; then
  echo "missing h2 checkpoint: $H" >&2
  exit 1
fi

for seed in $SEEDS; do
  ckpt="$BASELINE_ROOT/seed${seed}/policy_final.pth"
  baseline="$BASELINE_ROOT/seed${seed}/eval_100ep_final.json"
  if [ ! -f "$ckpt" ]; then
    echo "missing actor1 checkpoint: $ckpt" >&2
    exit 1
  fi
  if [ ! -f "$baseline" ]; then
    echo "missing h1 baseline: $baseline" >&2
    exit 1
  fi
done

if [ "$SMOKE" != "1" ]; then
  for seed in $SEEDS; do
    if [ -f "$OUT_ROOT/seed${seed}/eval_100ep_final.json" ]; then
      echo "refusing to overwrite existing result:" \
        "$OUT_ROOT/seed${seed}/eval_100ep_final.json" >&2
      echo "move it aside or set OUT_ROOT to a new directory" >&2
      exit 1
    fi
  done
fi

mkdir -p "$OUT_ROOT"
echo "C1s h2 actor-seed swap: seeds='$SEEDS' episodes=$EPISODES smoke=$SMOKE"
echo "h2: $H"
echo "out: $OUT_ROOT"

for seed in $SEEDS; do
  ckpt="$BASELINE_ROOT/seed${seed}/policy_final.pth"
  out="$OUT_ROOT/seed${seed}"
  mkdir -p "$out"
  if [ "$SMOKE" = "1" ] && [ -f "$out/eval_100ep_final.json" ]; then
    rm -f "$out/eval_100ep_final.json"
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] evaluate actor1 seed=$seed with h2"
  "$PYTHON" -u nav/scripts/evaluate_v6.py \
    --checkpoint "$ckpt" \
    --h-checkpoint "$H" \
    --num-episodes "$EPISODES" \
    --seed "$EVAL_SEED" \
    --device cuda \
    --out "$out/eval_100ep_final.json" \
    > "$out/eval_100ep.log" 2>&1
  echo "  done: $out/eval_100ep_final.json"
done

if [ "$SMOKE" != "1" ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] apply C1s consistency rule"
  "$PYTHON" -u nav/scripts/summarize_v6_nav_c1s.py \
    --out-root "$OUT_ROOT" \
    --baseline-root "$BASELINE_ROOT" \
    --out "$OUT_ROOT/decision.json" \
    > "$OUT_ROOT/decision.log" 2>&1
  echo "  decision: $OUT_ROOT/decision.json"
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done: $OUT_ROOT"
