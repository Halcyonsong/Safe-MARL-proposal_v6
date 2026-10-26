#!/usr/bin/env bash
# Evaluate the official centralized CBF-QP formulation with OSQP fallback.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/nav"

PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-../outputs/v5/baselines/centralized_cbf_qp}"
EPISODES="${EPISODES:-100}"
CHUNK="${CHUNK:-10}"
SEED="${SEED:-10000}"
ALPHA="${ALPHA:-1.0}"
OMP="${OMP_NUM_THREADS:-2}"
MKL="${MKL_NUM_THREADS:-2}"

mkdir -p "${OUT_DIR}/chunks"
rm -f "${OUT_DIR}"/chunks/chunk_*.json
n_chunks=$((EPISODES / CHUNK))
for i in $(seq 0 $((n_chunks - 1))); do
  (
    OMP_NUM_THREADS="${OMP}" MKL_NUM_THREADS="${MKL}" \
    "${PYTHON}" scripts/evaluate_cbf_qp_official.py \
      --num-episodes "${CHUNK}" \
      --seed $((SEED + i * CHUNK)) \
      --alpha "${ALPHA}" \
      --out "${OUT_DIR}/chunks/chunk_${i}.json" \
      > "${OUT_DIR}/chunks/chunk_${i}.log" 2>&1
  ) &
done
wait
"${PYTHON}" scripts/combine_eval_chunks.py \
  --pattern "${OUT_DIR}/chunks/chunk_*.json" \
  --out "${OUT_DIR}/eval_${EPISODES}ep.json" \
  > "${OUT_DIR}/combine.log" 2>&1
echo "evaluated centralized CBF-QP on ${EPISODES} fixed scenes"
