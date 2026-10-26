#!/usr/bin/env bash
# Evaluate MAPPO-Lagrangian checkpoints on the 100 fixed V5 scenes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/nav"

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
CHECKPOINT="${CHECKPOINT:-policy_final.pth}"
OUT_ROOT="${OUT_ROOT:-../outputs/v5/baselines/mappo_lag}"
OMP="${OMP_NUM_THREADS:-2}"
MKL="${MKL_NUM_THREADS:-2}"

for seed in ${SEEDS}; do
  out="${OUT_ROOT}/seed${seed}"
  chunks="${out}/chunks_$(basename "${CHECKPOINT}" .pth)"
  rm -rf "${chunks}"
  mkdir -p "${chunks}"
  for i in 0 1 2 3 4 5 6 7 8 9; do
    (
      OMP_NUM_THREADS="${OMP}" MKL_NUM_THREADS="${MKL}" \
      "${PYTHON}" scripts/evaluate_mappo_lag_official.py \
        --checkpoint "${out}/${CHECKPOINT}" \
        --num-episodes 10 \
        --seed $((10000 + i * 10)) \
        --device cpu \
        --out "${chunks}/chunk_${i}.json" \
        > "${chunks}/chunk_${i}.log" 2>&1
    ) &
  done
  wait
  "${PYTHON}" scripts/combine_eval_chunks.py \
    --pattern "${chunks}/chunk_*.json" \
    --out "${out}/eval_100ep_$(basename "${CHECKPOINT}" .pth).json" \
    > "${out}/combine_$(basename "${CHECKPOINT}" .pth).log" 2>&1
  echo "evaluated MAPPO-Lag seed ${seed} (${CHECKPOINT})"
done
"${PYTHON}" scripts/aggregate_qp_free.py \
  --pattern "${OUT_ROOT}/seed*/eval_100ep_$(basename "${CHECKPOINT}" .pth).json" \
  --out "${OUT_ROOT}/aggregate_100ep_$(basename "${CHECKPOINT}" .pth).json" \
  > "${OUT_ROOT}/aggregate_$(basename "${CHECKPOINT}" .pth).log" 2>&1
echo "aggregated MAPPO-Lag (${CHECKPOINT})"
