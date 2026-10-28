#!/usr/bin/env bash
# V6 Ant V_T-gradient pilot: 250k warm-start vs matched no-V_T control.
#
# The existing 1-D A_T pilot failed because the actor received only a value
# scalar.  This pilot adds the unit-normalized state gradient:
#   task feature = [A_T, grad_s V_T]   (32-D for the 31-D Ant observation)
# and runs a short 250k continuation from the same 10M seed-0 base, with the
# same h/cost/reward/optimizer and a matched no-V_T control.
#
# The Ant simulator path needs the mappo_lag conda env (mujoco_py,
# tensorboardX); the nav safe_marl env does not provide them.  Example:
#   PYTHON=/home/xiang/miniconda3/envs/mappo_lag/bin/python \
#     SMOKE=1 bash nav/scripts/run_v6_ant_vt_grad_pilot.sh
#
# Smoke:
#   SMOKE=1 bash nav/scripts/run_v6_ant_vt_grad_pilot.sh
# Full (diagnostic, single seed):
#   CONFIRM=1 bash nav/scripts/run_v6_ant_vt_grad_pilot.sh
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
VT="${VT:-outputs/v6/independent/ant_vt/ant_task_value.pkl}"
SEED="${SEED:-0}"
START_STEP="${START_STEP:-10000000}"
END_STEP="${END_STEP:-10250000}"
EPISODES="${EPISODES:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-250000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
ROOT_OUT="${ROOT_OUT:-outputs/v6/independent/ant_vt_grad_250k}"
WARM_OUT="${WARM_OUT:-${ROOT_OUT}/seed0}"
CTRL_OUT="${CTRL_OUT:-outputs/v6/independent/ant_vt_grad_control_250k/seed0}"
SMOKE="${SMOKE:-0}"
CONFIRM="${CONFIRM:-0}"

if [ "$SMOKE" = "1" ]; then
  ROOT_OUT="${SMOKE_OUT:-/tmp/opencode/ant_vt_grad_smoke}/vt"
  WARM_OUT="${SMOKE_OUT:-/tmp/opencode/ant_vt_grad_smoke}/warm"
  CTRL_OUT="${SMOKE_OUT:-/tmp/opencode/ant_vt_grad_smoke}/control"
  END_STEP=$((START_STEP + 2048))
  EPISODES=1
  EVAL_INTERVAL=0
  EVAL_EPISODES=1
  rm -rf "${SMOKE_OUT:-/tmp/opencode/ant_vt_grad_smoke}"
else
  if [ "$CONFIRM" != "1" ]; then
    echo "Refusing the Ant V_T-gradient pilot without CONFIRM=1." >&2
    echo "Use SMOKE=1 for a pipeline check." >&2
    exit 2
  fi
fi

for path in "$BASE" "$H" "$VT"; do
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
CHAIN_LOG="${ROOT_OUT}/chain.log"
if [ "$SMOKE" = "1" ]; then
  CHAIN_LOG="${ROOT_OUT}/chain_smoke.log"
fi
{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Ant V_T-gradient pilot smoke=$SMOKE"
  echo "base: $BASE"
  echo "h: $H"
  echo "vt: $VT"
  echo "start_step: $START_STEP end_step: $END_STEP"
  echo "warm: $WARM_OUT"
  echo "control: $CTRL_OUT"
} | tee -a "$CHAIN_LOG"

COMMON=(
  --resume "$BASE"
  --h-checkpoint "$H"
  --total-steps "$END_STEP"
  --eval-interval "$EVAL_INTERVAL"
  --eval-episodes "$EVAL_EPISODES"
  --eval-seed 20000
  --save-interval "$EVAL_INTERVAL"
  --skip-final-eval
  --device cpu
  --seed "$SEED"
)

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] warm-start A_T_grad" | tee -a "$CHAIN_LOG"
"$PYTHON" -u nav/scripts/train_v6_ant_mappo.py \
  --config configs/v6/v6_ant_vt_grad_250k.yaml \
  --out "$WARM_OUT" \
  --task-value-checkpoint "$VT" \
  --freeze-task-value \
  "${COMMON[@]}" > "$WARM_OUT/train.log" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] matched no-V_T control" | tee -a "$CHAIN_LOG"
"$PYTHON" -u nav/scripts/train_v6_ant_mappo.py \
  --config configs/v6/v6_ant_vt_grad_control_250k.yaml \
  --out "$CTRL_OUT" \
  "${COMMON[@]}" > "$CTRL_OUT/train.log" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] 100ep evaluations" | tee -a "$CHAIN_LOG"
for arm in warm control; do
  if [ "$arm" = "warm" ]; then
    ARM_OUT="$WARM_OUT"
    TASK_ARGS=(--task-value-checkpoint "$VT")
  else
    ARM_OUT="$CTRL_OUT"
    TASK_ARGS=()
  fi
  "$PYTHON" -u nav/scripts/evaluate_v6_ant_mappo.py \
    --checkpoint "$ARM_OUT/policy_final.pth" --h-checkpoint "$H" \
    --episodes "$EPISODES" --seed 20000 --device cpu \
    "${TASK_ARGS[@]}" \
    --out "$ARM_OUT/eval_${EPISODES}ep_deterministic.json" \
    > "$ARM_OUT/eval_${EPISODES}ep_deterministic.log" 2>&1
  "$PYTHON" -u nav/scripts/evaluate_v6_ant_mappo.py \
    --checkpoint "$ARM_OUT/policy_final.pth" --h-checkpoint "$H" \
    --episodes "$EPISODES" --seed 20000 --device cpu --stochastic \
    "${TASK_ARGS[@]}" \
    --out "$ARM_OUT/eval_${EPISODES}ep_stochastic.json" \
    > "$ARM_OUT/eval_${EPISODES}ep_stochastic.log" 2>&1
done

"$PYTHON" - "$WARM_OUT/eval_${EPISODES}ep_deterministic.json" \
  "$CTRL_OUT/eval_${EPISODES}ep_deterministic.json" \
  "$WARM_OUT/eval_${EPISODES}ep_stochastic.json" \
  "$CTRL_OUT/eval_${EPISODES}ep_stochastic.json" \
  "$EPISODES" "$SMOKE" "$CHAIN_LOG" <<'PY'
import json, sys

paths = sys.argv[1:5]
episodes = int(sys.argv[5])
smoke = sys.argv[6] == '1'
log_path = sys.argv[7]
rows = {}
for name, path in zip(('warm_det', 'ctrl_det', 'warm_stoch', 'ctrl_stoch'), paths):
    rows[name] = json.load(open(path))['metrics']

def fmt(row):
    return (f"safe {row['safe_episode']:.3f} "
            f"forward {row['forward_return']:.1f} "
            f"return {row['episode_return']:.1f} "
            f"cost {row['cost_fraction']:.5f}")

lines = [f'{name}: {fmt(row)}' for name, row in rows.items()]
if not smoke:
    wd, cd = rows['warm_det'], rows['ctrl_det']
    task_gain = (wd['forward_return'] >= 1.05 * cd['forward_return']
                 or wd['episode_return'] >= 1.05 * cd['episode_return'])
    safety_ok = (wd['safe_episode'] >= cd['safe_episode'] - 0.03
                 and wd['cost_fraction'] <= cd['cost_fraction'] + 0.005)
    lines.append(f'task_gain={task_gain} safety_not_worse={safety_ok} '
                 f'vt_grad_supported={task_gain and safety_ok}')
else:
    lines.append('SMOKE only; no decision.')
text = '\n'.join(lines)
print(text)
with open(log_path, 'a') as stream:
    stream.write(text + '\n')
PY

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Ant V_T-gradient pilot done" | tee -a "$CHAIN_LOG"
