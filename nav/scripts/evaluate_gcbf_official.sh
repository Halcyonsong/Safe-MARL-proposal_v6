#!/usr/bin/env bash
# Evaluate an official GCBF/GCBF+ checkpoint on 100 fixed scene keys.
# GCBF+ inference is actor-only; the QP dependency is import-only for this path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/nav"

PYTHON="${PYTHON:-python}"
PATH_MODEL="${PATH_MODEL:-third_party/gcbfplus/pretrained/DoubleIntegrator/gcbf+}"
STEP="${STEP:-1000}"
OUT_DIR="${OUT_DIR:-../outputs/v5/baselines/gcbfplus}"
EPISODES="${EPISODES:-100}"
CHUNK="${CHUNK:-10}"
SEED="${SEED:-10000}"
OMP="${OMP_NUM_THREADS:-2}"
MKL="${MKL_NUM_THREADS:-2}"

mkdir -p "${OUT_DIR}/chunks"
rm -f "${OUT_DIR}"/chunks/chunk_*.json
n_chunks=$((EPISODES / CHUNK))
for i in $(seq 0 $((n_chunks - 1))); do
  (
    OMP_NUM_THREADS="${OMP}" MKL_NUM_THREADS="${MKL}" \
    "${PYTHON}" scripts/evaluate_gcbf_official.py \
      --path "${PATH_MODEL}" \
      --step "${STEP}" \
      --num-episodes "${CHUNK}" \
      --seed $((SEED + i * CHUNK)) \
      --out "${OUT_DIR}/chunks/chunk_${i}.json" \
      > "${OUT_DIR}/chunks/chunk_${i}.log" 2>&1
  ) &
done
wait
"${PYTHON}" scripts/combine_eval_chunks.py \
  --pattern "${OUT_DIR}/chunks/chunk_*.json" \
  --out "${OUT_DIR}/eval_${EPISODES}ep.json" \
  > "${OUT_DIR}/combine.log" 2>&1
echo "evaluated ${PATH_MODEL} on ${EPISODES} fixed scenes"
