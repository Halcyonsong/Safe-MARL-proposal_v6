#!/usr/bin/env bash
# Train MAPPO-Lagrangian baseline (pure task reward + official collision cost).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/nav"

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-100000}"
OUT_ROOT="${OUT_ROOT:-../outputs/v5/baselines/mappo_lag}"
OMP="${OMP_NUM_THREADS:-2}"
MKL="${MKL_NUM_THREADS:-2}"

for seed in ${SEEDS}; do
  out="${OUT_ROOT}/seed${seed}"
  mkdir -p "${out}"
  echo "launching MAPPO-Lag seed ${seed} -> ${out}"
  OMP_NUM_THREADS="${OMP}" MKL_NUM_THREADS="${MKL}" \
    "${PYTHON}" scripts/train_mappo_lag_official.py \
      --config configs/baselines/mappo_lag_double_integrator.yaml \
      --seed "${seed}" \
      --total-steps "${STEPS}" \
      --rollout-steps 256 \
      --eval-interval 16384 \
      --eval-episodes 10 \
      --save-interval 16384 \
      --device cpu \
      --out "${out}" \
      > "${out}/train.log" 2>&1 &
done
wait
echo "finished MAPPO-Lag under ${OUT_ROOT}"
