#!/usr/bin/env bash
# V6 navigation Stage C0: h2-only temporal-calibration screen.
#
# For each seed:
#   1. collect actor1 stochastic rollouts with the frozen repaired h1;
#   2. merge with the fixed random coverage used by h1 (50/50 timesteps);
#   3. train h2 from scratch with the same target/arch/loss/calibration;
#   4. calibrate the gate on h2 validation;
#   5. evaluate h1 vs h2 offline on held-out actor1 rows.
#
# This runner does NOT train actor2 and does NOT run a closed-loop policy
# evaluation.  Full C0 requires CONFIRM=1.  Smoke mode is explicit:
#   SMOKE=1 SEEDS=0 bash nav/scripts/run_v6_nav_c0_h2.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
EPISODES="${EPISODES:-}"
EVAL_EPISODES="${EVAL_EPISODES:-}"
EPOCHS="${EPOCHS:-}"
BASE_SEED="${BASE_SEED:-30000}"
EVAL_BASE_SEED="${EVAL_BASE_SEED:-31000}"
SMOKE="${SMOKE:-0}"
CONFIRM="${CONFIRM:-0}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

DATA_ROOT="${DATA_ROOT:-outputs/v6/independent/data_c0_h2}"
H_ROOT="${H_ROOT:-outputs/v6/independent/h_c0}"
EVAL_ROOT="${EVAL_ROOT:-outputs/v6/independent/nav_c0_eval}"
LOG_ROOT="${LOG_ROOT:-outputs/v6/independent/nav_c0/logs}"

H1="${H1:-outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl}"
H1_CALIB="${H1_CALIB:-outputs/v6/independent/h_stageb_repair/gate_calibration.json}"
RANDOM_DATA="${RANDOM_DATA:-outputs/v6/independent/data_random_rebuilt}"
ACTOR_DIR="${ACTOR_DIR:-outputs/v6/independent/nav_stageb_repair_16k}"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/v6/v6_nav_c0_h2.yaml}"
COLLECT_CONFIG="${COLLECT_CONFIG:-configs/v6/v6_nav_stageb_repair_16k.yaml}"

if [ "$SMOKE" = "1" ]; then
  EPISODES="${EPISODES:-5}"
  EVAL_EPISODES="${EVAL_EPISODES:-5}"
  EPOCHS="${EPOCHS:-1}"
  if [ "$EPISODES" -gt 5 ]; then EPISODES=5; fi
  if [ "$EVAL_EPISODES" -gt 5 ]; then EVAL_EPISODES=5; fi
else
  EPISODES="${EPISODES:-100}"
  EVAL_EPISODES="${EVAL_EPISODES:-101}"
  EPOCHS="${EPOCHS:-40}"
  if [ "$CONFIRM" != "1" ]; then
    echo "Refusing full C0 without CONFIRM=1." >&2
    echo "Use SMOKE=1 for a pipeline check, or CONFIRM=1 for the full screen." >&2
    exit 2
  fi
fi

mkdir -p "$DATA_ROOT" "$H_ROOT" "$EVAL_ROOT" "$LOG_ROOT"

echo "C0 h2 screen: seeds='$SEEDS' train_episodes=$EPISODES eval_episodes=$EVAL_EPISODES epochs=$EPOCHS smoke=$SMOKE"
for SEED in $SEEDS; do
  SEED_BASE=$((BASE_SEED + 1000 * SEED))
  EVAL_SEED_BASE=$((EVAL_BASE_SEED + 1000 * SEED))
  ACTOR1_DIR="$DATA_ROOT/seed${SEED}_actor1"
  EVAL_DIR="$DATA_ROOT/seed${SEED}_eval"
  MERGED_DIR="$DATA_ROOT/seed${SEED}_merged"
  H_OUT="$H_ROOT/seed${SEED}"
  LOG="$LOG_ROOT/seed${SEED}.log"
  ACTOR1="$ACTOR_DIR/seed${SEED}/policy_final.pth"

  echo "=== seed $SEED (train seeds $SEED_BASE..$((SEED_BASE + EPISODES - 1)), eval seeds $EVAL_SEED_BASE..$((EVAL_SEED_BASE + EVAL_EPISODES - 1))) ==="
  {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] seed=$SEED collect actor1 training data"
    "$PYTHON" -u nav/scripts/collect_v6_h_data.py \
      --config "$COLLECT_CONFIG" \
      --out "$ACTOR1_DIR" \
      --episodes "$EPISODES" --steps 256 --seed "$SEED_BASE" \
      --policy-checkpoint "$ACTOR1" --h-checkpoint "$H1" \
      --policy-device "$POLICY_DEVICE" --stochastic-actions \
      --horizon 32 --margin-scale 0.2

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] seed=$SEED collect held-out actor1 evaluation data (deterministic)"
    "$PYTHON" -u nav/scripts/collect_v6_h_data.py \
      --config "$COLLECT_CONFIG" \
      --out "$EVAL_DIR" \
      --episodes "$EVAL_EPISODES" --steps 256 --seed "$EVAL_SEED_BASE" \
      --val-ratio 0.01 --test-ratio 1.0 \
      --policy-checkpoint "$ACTOR1" --h-checkpoint "$H1" \
      --policy-device "$POLICY_DEVICE" \
      --horizon 32 --margin-scale 0.2

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] seed=$SEED merge actor1 + random"
    "$PYTHON" -u nav/scripts/merge_v6_c0_data.py \
      --actor1-dir "$ACTOR1_DIR" \
      --random-dir "$RANDOM_DATA" \
      --out "$MERGED_DIR"

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] seed=$SEED train h2"
    EXTRA=()
    if [ "$SMOKE" = "1" ]; then
      EXTRA=(--max-rows 64 --batch-size 32 --grad-eval-rows 8
             --grad-max-rows 1)
    fi
    "$PYTHON" -u nav/scripts/train_v6_official_h.py \
      --config "$TRAIN_CONFIG" \
      --train-data "$MERGED_DIR/train_data.npz" \
      --val-data "$MERGED_DIR/val_data.npz" \
      --test-data "$MERGED_DIR/test_data.npz" \
      --out "$H_OUT" --seed "$SEED" --epochs "$EPOCHS" \
      --tag v6_h_c0 "${EXTRA[@]}"

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] seed=$SEED calibrate gate"
    "$PYTHON" -u nav/scripts/calibrate_v6_nav_gate.py \
      --checkpoint "$H_OUT/v6_h_c0.pkl" \
      --data "$MERGED_DIR/val_data.npz" \
      --out "$H_OUT/gate_calibration.json"

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] seed=$SEED evaluate h1 vs h2 on held-out actor1 data"
    "$PYTHON" -u nav/scripts/evaluate_v6_nav_c0_h2.py \
      --data "$EVAL_DIR/test_data.npz" \
      --h1 "$H1" --h1-calibration "$H1_CALIB" \
      --h2 "$H_OUT/v6_h_c0.pkl" \
      --h2-calibration "$H_OUT/gate_calibration.json" \
      --out "$EVAL_ROOT/seed${SEED}.json"

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] seed=$SEED complete"
  } > "$LOG" 2>&1

  tail -n 3 "$LOG"
done

echo "C0 complete. Evaluations: $EVAL_ROOT/seed*.json"
