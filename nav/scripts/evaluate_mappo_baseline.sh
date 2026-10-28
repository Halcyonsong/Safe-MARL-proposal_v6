#!/usr/bin/env bash
# Evaluate pure MAPPO checkpoints on the same 100 fixed scenes used by V5.
# Scenes: seeds 10000..10099, deterministic actors.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/nav"

PYTHON="${PYTHON:-python}"
SEEDS="${SEEDS:-0 1 2}"
VARIANTS="${VARIANTS:-pure}"
CHECKPOINT="${CHECKPOINT:-policy_final.pth}"
OUT_ROOT="${OUT_ROOT:-../outputs/v5/baselines}"
CHUNK="${CHUNK:-20}"
EPISODES="${EPISODES:-100}"
OMP="${OMP_NUM_THREADS:-2}"
MKL="${MKL_NUM_THREADS:-2}"

for variant in ${VARIANTS}; do
  case "${variant}" in
    pure)
      out_variant="${OUT_ROOT}/mappo_pure"
      ;;
    same_reward|same-reward)
      out_variant="${OUT_ROOT}/mappo_same_reward"
      ;;
    penalty1|penalty_1)
      out_variant="${OUT_ROOT}/mappo_penalty1"
      ;;
    penalty100|penalty_100)
      out_variant="${OUT_ROOT}/mappo_penalty100"
      ;;
    *)
      echo "unknown MAPPO variant: ${variant}" >&2
      exit 2
      ;;
  esac
  for seed in ${SEEDS}; do
    out="${out_variant}/seed${seed}"
    chunks="${out}/chunks_$(basename "${CHECKPOINT}" .pth)"
    rm -rf "${chunks}"
    mkdir -p "${chunks}"
    n_chunks=$((EPISODES / CHUNK))
    for i in $(seq 0 $((n_chunks - 1))); do
      (
        OMP_NUM_THREADS="${OMP}" MKL_NUM_THREADS="${MKL}" \
        "${PYTHON}" scripts/evaluate_mappo_official.py \
          --checkpoint "${out}/${CHECKPOINT}" \
          --num-episodes "${CHUNK}" \
          --seed $((10000 + i * CHUNK)) \
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
    echo "evaluated ${variant} seed ${seed} (${CHECKPOINT})"
  done
  "${PYTHON}" scripts/aggregate_qp_free.py \
    --pattern "${out_variant}/seed*/eval_100ep_$(basename "${CHECKPOINT}" .pth).json" \
    --out "${out_variant}/aggregate_100ep_$(basename "${CHECKPOINT}" .pth).json" \
    > "${out_variant}/aggregate_$(basename "${CHECKPOINT}" .pth).log" 2>&1
  echo "aggregated ${variant} (${CHECKPOINT})"
done
