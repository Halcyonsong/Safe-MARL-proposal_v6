#!/usr/bin/env bash
# Train controlled V5 h(s) ablation baselines under the same 4096 -> 8192 -> 16384
# schedule as the current main V5 run.
#
# Variants:
#   no_h       same V5 architecture/reward/correction/cost, h/grad zeroed and
#              all h-based losses disabled
#   task_only  no h, no correction, no cost critic; task PPO with the same
#              collision-penalized reward
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/nav"

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
OUT_ROOT="${OUT_ROOT:-../outputs/v5/h_ablation}"
OMP="${OMP_NUM_THREADS:-2}"
MKL="${MKL_NUM_THREADS:-2}"

train_variant() {
  local variant="$1"
  local seed="$2"
  local config="configs/v5/qp_free_double_integrator_${variant}.yaml"
  local out="${OUT_ROOT}/${variant}/seed${seed}"
  mkdir -p "${out}"
  : > "${out}/train.log"

  # Stage 1: 4096 from random init.
  OMP_NUM_THREADS="${OMP}" MKL_NUM_THREADS="${MKL}" \
    "${PYTHON}" scripts/train_qp_free.py \
      --config "${config}" --seed "${seed}" --total-steps 4096 \
      --rollout-steps 128 --eval-interval 0 --save-interval 0 \
      --skip-final-eval --device cpu --out "${out}" \
      >> "${out}/train.log" 2>&1

  # Stage 2: resume to 8192.
  OMP_NUM_THREADS="${OMP}" MKL_NUM_THREADS="${MKL}" \
    "${PYTHON}" scripts/train_qp_free.py \
      --config "${config}" --seed "${seed}" --total-steps 8192 \
      --rollout-steps 128 --eval-interval 0 --save-interval 0 \
      --skip-final-eval --device cpu --out "${out}" \
      --resume "${out}/policy_final.pth" \
      >> "${out}/train.log" 2>&1

  # Stage 3: resume to 16384.
  OMP_NUM_THREADS="${OMP}" MKL_NUM_THREADS="${MKL}" \
    "${PYTHON}" scripts/train_qp_free.py \
      --config "${config}" --seed "${seed}" --total-steps 16384 \
      --rollout-steps 128 --eval-interval 0 --save-interval 0 \
      --skip-final-eval --device cpu --out "${out}" \
      --resume "${out}/policy_final.pth" \
      >> "${out}/train.log" 2>&1

  echo "finished ${variant} seed ${seed}" >> "${out}/train.log"
}

for variant in ${VARIANTS:-no_h task_only}; do
  for seed in ${SEEDS}; do
    train_variant "${variant}" "${seed}" &
  done
  wait
done

echo "all ablations finished under ${OUT_ROOT}"
