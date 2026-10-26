#!/usr/bin/env bash
# Train pure MAPPO baselines on the official GCBF+ DoubleIntegrator.
#
# Variants:
#   pure         pure task reward, no collision penalty (default)
#   same_reward  V5 collision-penalized reward, still no h/correction/cost/QP
#
# Usage:
#   SEEDS="0 1 2" STEPS=100000 VARIANTS="pure" \
#     bash scripts/run_mappo_baseline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/nav"

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-100000}"
VARIANTS="${VARIANTS:-pure}"
OUT_ROOT="${OUT_ROOT:-../outputs/v5/baselines}"
OMP="${OMP_NUM_THREADS:-2}"
MKL="${MKL_NUM_THREADS:-2}"

for variant in ${VARIANTS}; do
  case "${variant}" in
    pure)
      config="configs/baselines/mappo_double_integrator_pure.yaml"
      out_variant="${OUT_ROOT}/mappo_pure"
      ;;
    same_reward|same-reward)
      config="configs/baselines/mappo_double_integrator_same_reward.yaml"
      out_variant="${OUT_ROOT}/mappo_same_reward"
      ;;
    *)
      echo "unknown MAPPO variant: ${variant}" >&2
      exit 2
      ;;
  esac
  for seed in ${SEEDS}; do
    out="${out_variant}/seed${seed}"
    mkdir -p "${out}"
    echo "launching ${variant} seed ${seed} -> ${out}"
    OMP_NUM_THREADS="${OMP}" MKL_NUM_THREADS="${MKL}" \
      "${PYTHON}" scripts/train_mappo_official.py \
        --config "${config}" \
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
  echo "finished ${variant} under ${out_variant}"
done
