#!/usr/bin/env bash
# Evaluate the h-ablation checkpoints on the same 100 fixed scenes used by the
# main V5 protocol: seeds 10000..10099, deterministic actors.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/nav"

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
OUT_ROOT="${OUT_ROOT:-../outputs/v5/h_ablation}"
OMP="${OMP_NUM_THREADS:-2}"
MKL="${MKL_NUM_THREADS:-2}"

for variant in ${VARIANTS:-no_h task_only}; do
  for seed in ${SEEDS}; do
    out="${OUT_ROOT}/${variant}/seed${seed}"
    chunks="${out}/chunks"
    rm -rf "${chunks}"
    mkdir -p "${chunks}"
    for i in 0 1 2 3 4 5 6 7 8 9; do
      (
        OMP_NUM_THREADS="${OMP}" MKL_NUM_THREADS="${MKL}" \
        "${PYTHON}" scripts/evaluate_qp_free.py \
          --checkpoint "${out}/policy_final.pth" \
          --num-episodes 10 --seed $((10000 + i * 10)) --device cpu \
          --out "${chunks}/chunk_${i}.json" \
          > "${chunks}/chunk_${i}.log" 2>&1
      ) &
    done
    wait
    "${PYTHON}" scripts/combine_eval_chunks.py \
      --pattern "${chunks}/chunk_*.json" \
      --out "${out}/eval_100ep.json" \
      > "${out}/combine.log" 2>&1
    echo "evaluated ${variant} seed ${seed}"
  done
  "${PYTHON}" scripts/aggregate_qp_free.py \
    --pattern "${OUT_ROOT}/${variant}/seed*/eval_100ep.json" \
    --out "${OUT_ROOT}/${variant}/aggregate_100ep.json" \
    > "${OUT_ROOT}/${variant}/aggregate.log" 2>&1
  echo "aggregated ${variant}"
done
