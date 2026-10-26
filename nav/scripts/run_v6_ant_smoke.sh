#!/usr/bin/env bash
# End-to-end V6 Ant smoke: V6 h data -> frozen h -> from-scratch MAPPO.
# Requires the mappo_lag Python environment and MuJoCo 2.1.0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${ROOT}/nav${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

OUT="${OUT:-${ROOT}/outputs/v6/independent/ant/smoke}"
DATA="${OUT}/h_data"
H="${OUT}/h"
POLICY="${OUT}/h_only"
SEED="${SEED:-0}"

mkdir -p "${OUT}"
"${PYTHON}" nav/scripts/collect_v6_ant_h_data.py \
  --out "${DATA}" --episodes 4 --episode-limit 100 --horizon 16 \
  --aug-rows 1000 --perturb-rows 500 --seed "${SEED}"

"${PYTHON}" nav/scripts/train_v6_ant_h.py \
  --data-dir "${DATA}" --out "${H}" --epochs 5 --batch-size 128 \
  --hidden 64,64 --seed "${SEED}" --device cpu

"${PYTHON}" nav/scripts/train_v6_ant_mappo.py \
  --config configs/v6/v6_ant_h_base.yaml \
  --out "${POLICY}" --h-checkpoint "${H}/ant_h.pkl" \
  --total-steps 1024 --num-envs 1 --rollout-steps 256 --episode-limit 200 \
  --eval-interval 512 --eval-episodes 2 --eval-seed 20000 \
  --save-interval 0 --device cpu

printf '\nV6 Ant smoke complete: %s\n' "${OUT}"
