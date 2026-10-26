#!/usr/bin/env bash
# V6 Ant V_T transfer diagnostic: base sanity -> V_T -> 1M warm-start ->
# matched no-V_T control -> 100ep det/stoch -> pre-registered decision.
#
# Single seed, diagnostic only.  The promoted Ant 3-seed claim is unchanged.
# Full run:
#   CONFIRM=1 bash nav/scripts/run_v6_ant_vt_transfer.sh
# Pipeline smoke (not a claim):
#   SMOKE=1 bash nav/scripts/run_v6_ant_vt_transfer.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${ROOT}/nav${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

BASE="${BASE:-outputs/v6/independent/ant/mappo_h_pred_gate_10m/seed0/policy_010000000.pth}"
H="${H:-outputs/v6/independent/ant/h_predictive/ant_h.pkl}"
ROOT_OUT="${ROOT_OUT:-outputs/v6/independent/ant_vt}"
WARM_OUT="${WARM_OUT:-outputs/v6/independent/ant_vt_warmstart_1m/seed0}"
CTRL_OUT="${CTRL_OUT:-outputs/v6/independent/ant_vt_control_1m/seed0}"
SEED="${SEED:-0}"
EPISODES="${EPISODES:-100}"
COLLECT_EPISODES="${COLLECT_EPISODES:-100}"
START_STEP="${START_STEP:-10000000}"
END_STEP="${END_STEP:-11000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250000}"
SMOKE="${SMOKE:-0}"
CONFIRM="${CONFIRM:-0}"

if [ "$SMOKE" = "1" ]; then
  ROOT_OUT="${SMOKE_OUT:-/tmp/opencode/ant_vt_smoke_run}/vt"
  WARM_OUT="${SMOKE_OUT:-/tmp/opencode/ant_vt_smoke_run}/warm"
  CTRL_OUT="${SMOKE_OUT:-/tmp/opencode/ant_vt_smoke_run}/control"
  EPISODES=1
  COLLECT_EPISODES=3
  END_STEP=$((START_STEP + 16384))
  EVAL_INTERVAL=0
else
  if [ "$CONFIRM" != "1" ]; then
    echo "Refusing the Ant V_T transfer run without CONFIRM=1." >&2
    echo "Use SMOKE=1 for a pipeline check." >&2
    exit 2
  fi
fi

for path in "$BASE" "$H"; do
  if [ ! -f "$path" ]; then
    echo "missing required checkpoint: $path" >&2
    exit 1
  fi
done

if [ "$SMOKE" != "1" ]; then
  for path in "$WARM_OUT/policy_final.pth" "$CTRL_OUT/policy_final.pth"; do
    if [ -f "$path" ]; then
      echo "refusing to overwrite existing checkpoint: $path" >&2
      exit 1
    fi
  done
fi

mkdir -p "$ROOT_OUT" "$WARM_OUT" "$CTRL_OUT"
CHAIN_LOG="$ROOT_OUT/chain.log"
if [ "$SMOKE" = "1" ]; then
  CHAIN_LOG="$ROOT_OUT/chain_smoke.log"
fi
{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Ant V_T transfer start smoke=$SMOKE"
  echo "base: $BASE"
  echo "h: $H"
  echo "start_step: $START_STEP end_step: $END_STEP"
} | tee -a "$CHAIN_LOG"

# 1. Base h-only sanity: re-evaluate the existing 10M seed-0 checkpoint.
BASE_EVAL="$ROOT_OUT/base_eval"
mkdir -p "$BASE_EVAL"
if [ ! -f "$BASE_EVAL/det_${EPISODES}ep.json" ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] base deterministic eval" \
    | tee -a "$CHAIN_LOG"
  "$PYTHON" -u nav/scripts/evaluate_v6_ant_mappo.py \
    --checkpoint "$BASE" --h-checkpoint "$H" \
    --episodes "$EPISODES" --seed 20000 --device cpu \
    --out "$BASE_EVAL/det_${EPISODES}ep.json" \
    > "$BASE_EVAL/det_${EPISODES}ep.log" 2>&1
fi
if [ ! -f "$BASE_EVAL/stoch_${EPISODES}ep.json" ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] base stochastic eval" \
    | tee -a "$CHAIN_LOG"
  "$PYTHON" -u nav/scripts/evaluate_v6_ant_mappo.py \
    --checkpoint "$BASE" --h-checkpoint "$H" \
    --episodes "$EPISODES" --seed 20000 --device cpu --stochastic \
    --out "$BASE_EVAL/stoch_${EPISODES}ep.json" \
    > "$BASE_EVAL/stoch_${EPISODES}ep.log" 2>&1
fi
"$PYTHON" - "$BASE_EVAL/det_${EPISODES}ep.json" \
  "$BASE_EVAL/stoch_${EPISODES}ep.json" <<'PY' | tee -a "$CHAIN_LOG"
import json, sys
det = json.load(open(sys.argv[1]))['metrics']
sto = json.load(open(sys.argv[2]))['metrics']
print('base sanity:', {
    'det_safe': det['safe_episode'],
    'det_forward': det['forward_return'],
    'det_cost_fraction': det['cost_fraction'],
    'stoch_safe': sto['safe_episode'],
    'stoch_forward': sto['forward_return'],
})
if det['safe_episode'] < 0.90 or det['cost_fraction'] > 0.01:
    raise SystemExit('base deterministic sanity failed; stop')
if sto['safe_episode'] < 0.60:
    raise SystemExit('base stochastic sanity failed; stop')
PY

# 2. V_T data collection.
DATA="$ROOT_OUT/data/ant_h10m_seed0_${COLLECT_EPISODES}ep.npz"
mkdir -p "$(dirname "$DATA")"
if [ ! -f "$DATA" ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] collecting V_T data" \
    | tee -a "$CHAIN_LOG"
  "$PYTHON" -u nav/scripts/collect_v6_ant_task_value_data.py \
    --checkpoint "$BASE" --h-checkpoint "$H" \
    --episodes "$COLLECT_EPISODES" --seed 30000 --device cpu \
    --out "$DATA" >> "$CHAIN_LOG" 2>&1
fi

# 3. V_T offline training + quality gate.
VT="$ROOT_OUT/ant_task_value.pkl"
if [ ! -f "$VT" ]; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] training V_T" \
    | tee -a "$CHAIN_LOG"
  "$PYTHON" -u nav/scripts/train_v6_ant_task_value.py \
    --data "$DATA" --out "$VT" \
    --epochs 40 --hidden 128 128 --device cpu \
    >> "$CHAIN_LOG" 2>&1
fi
"$PYTHON" - "$ROOT_OUT/summary.json" <<'PY' | tee -a "$CHAIN_LOG"
import json, sys
s = json.load(open(sys.argv[1]))
g = s['metrics']['quality_gate']
print('V_T quality gate:', g)
if not g['passed']:
    raise SystemExit(
        'V_T quality gate failed; fix V_T offline once, then stop this run')
PY

# 4. Warm-start and matched control, same base/steps/seeds.
COMMON=(
  --resume "$BASE"
  --h-checkpoint "$H"
  --total-steps "$END_STEP"
  --eval-interval "$EVAL_INTERVAL"
  --eval-episodes 20
  --eval-seed 20000
  --save-interval "$EVAL_INTERVAL"
  --skip-final-eval
  --device cpu
)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] warm-start 1M" | tee -a "$CHAIN_LOG"
"$PYTHON" -u nav/scripts/train_v6_ant_mappo.py \
  --config configs/v6/v6_ant_vt_warmstart_1m.yaml \
  --out "$WARM_OUT" \
  --task-value-checkpoint "$VT" \
  "${COMMON[@]}" > "$WARM_OUT/train.log" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] matched no-V_T control 1M" \
  | tee -a "$CHAIN_LOG"
"$PYTHON" -u nav/scripts/train_v6_ant_mappo.py \
  --config configs/v6/v6_ant_vt_control_1m.yaml \
  --out "$CTRL_OUT" \
  "${COMMON[@]}" > "$CTRL_OUT/train.log" 2>&1

# 5. Formal 100ep deterministic + stochastic evaluation for both arms.
for arm in warm control; do
  if [ "$arm" = "warm" ]; then
    ARM_OUT="$WARM_OUT"
  else
    ARM_OUT="$CTRL_OUT"
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${arm} deterministic eval" \
    | tee -a "$CHAIN_LOG"
  "$PYTHON" -u nav/scripts/evaluate_v6_ant_mappo.py \
    --checkpoint "$ARM_OUT/policy_final.pth" --h-checkpoint "$H" \
    --episodes "$EPISODES" --seed 20000 --device cpu \
    --out "$ARM_OUT/eval_${EPISODES}ep_deterministic.json" \
    > "$ARM_OUT/eval_${EPISODES}ep_deterministic.log" 2>&1
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ${arm} stochastic eval" \
    | tee -a "$CHAIN_LOG"
  "$PYTHON" -u nav/scripts/evaluate_v6_ant_mappo.py \
    --checkpoint "$ARM_OUT/policy_final.pth" --h-checkpoint "$H" \
    --episodes "$EPISODES" --seed 20000 --device cpu --stochastic \
    --out "$ARM_OUT/eval_${EPISODES}ep_stochastic.json" \
    > "$ARM_OUT/eval_${EPISODES}ep_stochastic.log" 2>&1
done

# 6. Pre-registered decision.
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] applying Ant V_T rule" \
  | tee -a "$CHAIN_LOG"
"$PYTHON" -u nav/scripts/summarize_v6_ant_vt_warmstart.py \
  --control-det "$CTRL_OUT/eval_${EPISODES}ep_deterministic.json" \
  --control-stoch "$CTRL_OUT/eval_${EPISODES}ep_stochastic.json" \
  --warmstart-det "$WARM_OUT/eval_${EPISODES}ep_deterministic.json" \
  --warmstart-stoch "$WARM_OUT/eval_${EPISODES}ep_stochastic.json" \
  --base-det "$BASE_EVAL/det_${EPISODES}ep.json" \
  --out "$ROOT_OUT/decision.json" \
  >> "$CHAIN_LOG" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Ant V_T transfer done" \
  | tee -a "$CHAIN_LOG"
