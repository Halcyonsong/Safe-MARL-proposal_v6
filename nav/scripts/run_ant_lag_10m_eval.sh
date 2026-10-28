#!/usr/bin/env bash
# Evaluate the Ant 10M from-scratch upstream MAPPO-Lag baseline.
#
# 100 deterministic + 100 stochastic episodes, eval seeds 20000..20099, using
# the V6-compatible metric adapter.  The checkpoint is the final model; there
# is no checkpoint selection.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/xiang/miniconda3/envs/mappo_lag/bin/python}"
SEED="${TRAIN_SEED:-0}"
SEED_DIR="${SEED_DIR:-${ROOT}/outputs/v6/independent/ant/mappo_lag_10m/seed${SEED}}"
DEVICE="${DEVICE:-cuda}"
EPISODES="${EPISODES:-100}"
EVAL_SEED="${EVAL_SEED:-20000}"

export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${ROOT}/nav${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

if [[ ! -d "${SEED_DIR}/models" ]]; then
  echo "missing models directory: ${SEED_DIR}/models" >&2
  exit 2
fi

for mode in deterministic stochastic; do
  extra=()
  if [[ "${mode}" == "stochastic" ]]; then
    extra=(--stochastic)
  fi
  out="${SEED_DIR}/eval_${EPISODES}ep_${mode}.json"
  echo "evaluating ${mode} -> ${out}"
  "${PYTHON}" nav/scripts/evaluate_ant_upstream_lag.py \
    --models-dir "${SEED_DIR}/models" \
    --num-episodes "${EPISODES}" \
    --seed "${EVAL_SEED}" \
    --training-seed "${SEED}" \
    --device "${DEVICE}" \
    "${extra[@]}" \
    --out "${out}" \
    > "${SEED_DIR}/eval_${EPISODES}ep_${mode}.log" 2>&1
done
echo "done: ${SEED_DIR}/eval_${EPISODES}ep_{deterministic,stochastic}.json"
