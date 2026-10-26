#!/usr/bin/env bash
# Safety-oriented V6 actor screen.
#
# All variants use the same frozen V6 h, from-scratch actor (except the
# explicitly marked fine-tune variants), no task channel, and no QP/projection.
# The screen trains briefly and evaluates 20 episodes; only promising variants
# should be promoted to 100 episodes.
#
# Usage:
#   VARIANTS="cost barrier cbf" bash scripts/run_v6_safety_screen.sh
#   VARIANTS="ft_cost ft_barrier" bash scripts/run_v6_safety_screen.sh
#   FULL=1 VARIANTS="barrier" bash scripts/run_v6_safety_screen.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/nav"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/v6/v6_nav_h_only_16k.yaml}"
H_CHECKPOINT="${H_CHECKPOINT:-$ROOT/outputs/v6/independent/h/v6_h.pkl}"
H_ONLY_FINAL="${H_ONLY_FINAL:-$ROOT/outputs/v6/independent/single_seed_16k/h_only/policy_final.pth}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/v6/independent/safety_opt}"
VARIANTS="${VARIANTS:-cost barrier cbf}"
SEED="${SEED:-0}"
TRAIN_STEPS="${TRAIN_STEPS:-16384}"
FT_STEPS="${FT_STEPS:-8192}"
SCREEN_EPISODES="${SCREEN_EPISODES:-20}"
EVAL_SEED="${EVAL_SEED:-10000}"
DEVICE="${DEVICE:-auto}"
FULL="${FULL:-0}"

mkdir -p "$OUT_ROOT"
"$PYTHON" scripts/check_v6_independence.py

run_variant() {
  local name="$1"
  shift
  local run_dir="$OUT_ROOT/${name}_s${SEED}"
  local resume_args=()
  local total_steps="$TRAIN_STEPS"

  if [[ "$name" == ft_* ]]; then
    if [ ! -f "$H_ONLY_FINAL" ]; then
      echo "missing h-only final checkpoint: $H_ONLY_FINAL" >&2
      return 1
    fi
    resume_args=(--resume "$H_ONLY_FINAL")
    total_steps=$((16384 + FT_STEPS))
  fi

  mkdir -p "$run_dir"
  echo "=== V6 safety screen variant=$name seed=$SEED ==="
  "$PYTHON" scripts/train_v6.py \
    --config "$CONFIG" \
    --h-checkpoint "$H_CHECKPOINT" \
    --seed "$SEED" \
    --total-steps "$total_steps" \
    --rollout-steps 128 \
    --eval-interval 0 \
    --eval-episodes 20 \
    --eval-seed "$EVAL_SEED" \
    --save-interval 0 \
    --skip-final-eval \
    --deterministic \
    --device "$DEVICE" \
    --out "$run_dir" \
    "${resume_args[@]}" \
    "$@" 2>&1 | tee "$run_dir/train.log"

  "$PYTHON" scripts/evaluate_v6.py \
    --checkpoint "$run_dir/policy_final.pth" \
    --h-checkpoint "$H_CHECKPOINT" \
    --num-episodes "$SCREEN_EPISODES" \
    --seed "$EVAL_SEED" \
    --device "$DEVICE" \
    --out "$run_dir/eval_${SCREEN_EPISODES}ep_final.json" \
    2>&1 | tee "$run_dir/eval_${SCREEN_EPISODES}ep.log"

  if [ "$FULL" = "1" ]; then
    "$PYTHON" scripts/evaluate_v6.py \
      --checkpoint "$run_dir/policy_final.pth" \
      --h-checkpoint "$H_CHECKPOINT" \
      --num-episodes 100 \
      --seed "$EVAL_SEED" \
      --device "$DEVICE" \
      --out "$run_dir/eval_100ep_final.json" \
      2>&1 | tee "$run_dir/eval_100ep.log"
  fi
}

for variant in $VARIANTS; do
  case "$variant" in
    cost)
      run_variant cost \
        --unsafe-penalty 40.0 --cost-actor-coef 1.0
      ;;
    cost20)
      run_variant cost20 \
        --unsafe-penalty 20.0 --cost-actor-coef 1.0
      ;;
    barrier)
      run_variant barrier \
        --h-barrier-coef 1.0 --h-barrier-margin 0.05 \
        --unsafe-penalty 10.0 --cost-actor-coef 0.5
      ;;
    cbf)
      run_variant cbf \
        --cbf-coef 0.2 --unsafe-penalty 10.0 --cost-actor-coef 0.5
      ;;
    ft_cost)
      run_variant ft_cost \
        --unsafe-penalty 40.0 --cost-actor-coef 1.0
      ;;
    ft_barrier)
      run_variant ft_barrier \
        --h-barrier-coef 1.0 --h-barrier-margin 0.05 \
        --unsafe-penalty 10.0 --cost-actor-coef 1.0
      ;;
    ft_both)
      run_variant ft_both \
        --h-barrier-coef 1.0 --h-barrier-margin 0.05 \
        --unsafe-penalty 40.0 --cost-actor-coef 1.0
      ;;
    *)
      echo "unknown variant: $variant" >&2
      exit 2
      ;;
  esac
done

"$PYTHON" scripts/summarize_v6_safety.py \
  --root "$OUT_ROOT" \
  --out "$OUT_ROOT/screen_summary.json" \
  2>&1 | tee "$OUT_ROOT/logs/screen_summary.log"
