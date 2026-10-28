#!/usr/bin/env bash
# MAPPO fixed-penalty fairness sweep at the matched 16k navigation budget.
#
# New arms: collision_penalty 1 and 100.  The existing penalty 10
# (`mappo_same_reward`) and penalty 0 (`mappo_pure`) rows are the endpoints
# and are not rerun.  Protocol: 16,384 steps, seeds 0/1/2, 100 deterministic
# episodes on eval seeds 10000..10099, final checkpoint.
#
# Usage (from anywhere):
#   bash nav/scripts/run_mappo_penalty_sweep_16k.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-/home/xiang/miniconda3/envs/safe_marl/bin/python}"
SEEDS="${SEEDS:-0 1 2}"
STEPS="${STEPS:-16384}"
VARIANTS="${VARIANTS:-penalty1 penalty100}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/v6/independent/nav_baselines_16k}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] MAPPO penalty sweep: variants=${VARIANTS} seeds=${SEEDS} steps=${STEPS}"

# ---- train variants concurrently (each variant runs its 3 seeds in parallel)
pids=()
for variant in ${VARIANTS}; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] train ${variant}"
  SEEDS="${SEEDS}" STEPS="${STEPS}" VARIANTS="${variant}" \
    OUT_ROOT="${OUT_ROOT}" PYTHON="${PYTHON}" \
    bash nav/scripts/run_mappo_baseline.sh \
    > "${LOG_DIR}/penalty_sweep_train_${variant}.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do
  wait "${p}" || echo "warning: a MAPPO training job exited nonzero" >&2
done

missing=()
for variant in ${VARIANTS}; do
  for seed in ${SEEDS}; do
    [[ -s "${OUT_ROOT}/mappo_${variant}/seed${seed}/policy_final.pth" ]] || \
      missing+=("${variant}:${seed}")
  done
done
if (( ${#missing[@]} > 0 )); then
  echo "incomplete MAPPO training: ${missing[*]}" >&2
  exit 3
fi
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] training complete"

# ---- evaluate sequentially to avoid oversubscribing CPU/JAX chunks ---------
for variant in ${VARIANTS}; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] eval ${variant}"
  SEEDS="${SEEDS}" VARIANTS="${variant}" OUT_ROOT="${OUT_ROOT}" \
    PYTHON="${PYTHON}" \
    bash nav/scripts/evaluate_mappo_baseline.sh \
    > "${LOG_DIR}/penalty_sweep_eval_${variant}.log" 2>&1
done

missing_eval=()
for variant in ${VARIANTS}; do
  for seed in ${SEEDS}; do
    [[ -s "${OUT_ROOT}/mappo_${variant}/seed${seed}/eval_100ep_policy_final.json" ]] || \
      missing_eval+=("${variant}:${seed}")
  done
done
if (( ${#missing_eval[@]} > 0 )); then
  echo "incomplete MAPPO evaluation: ${missing_eval[*]}" >&2
  exit 4
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] MAPPO penalty sweep complete: ${OUT_ROOT}"
