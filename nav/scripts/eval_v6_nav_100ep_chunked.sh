#!/usr/bin/env bash
# Formal 100-episode deterministic evaluation for V6 navigation checkpoints,
# split into fixed scene chunks so long CPU evaluations can run in parallel.
#
# Usage:
#   CHECKPOINT=... H=... OUT=... TAG=... bash nav/scripts/eval_v6_nav_100ep_chunked.sh
#
# Environment:
#   CHECKPOINT  policy checkpoint (required)
#   H           V6 h checkpoint (required)
#   OUT         seed output directory (required)
#   TAG         chunk-file prefix (default: eval_100ep_final)
#   EPISODES    total episodes (default: 100)
#   CHUNK       episodes per chunk (default: 20)
#   PARALLEL    concurrent chunk processes (default: 5, never above 5)
#   START_SEED  first eval scene seed (default: 10000)
#   DEVICE      torch device for the actor (default: cuda)
#   NUM_AGENTS / NUM_OBSTACLES / AREA_SIZE / MAX_STEP
#               optional evaluation-only density overrides; if unset the
#               checkpoint config is used unchanged.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
: "${CHECKPOINT:?set CHECKPOINT}"
: "${H:?set H}"
: "${OUT:?set OUT}"
TAG="${TAG:-eval_100ep_final}"
EPISODES="${EPISODES:-100}"
CHUNK="${CHUNK:-20}"
PARALLEL="${PARALLEL:-5}"
START_SEED="${START_SEED:-10000}"
DEVICE="${DEVICE:-cuda}"

if (( PARALLEL > 5 )); then
  echo "PARALLEL=${PARALLEL} is above the safe cap of 5" >&2
  exit 2
fi
if (( EPISODES % CHUNK != 0 )); then
  echo "EPISODES=${EPISODES} must be divisible by CHUNK=${CHUNK}" >&2
  exit 2
fi

CHUNK_DIR="${OUT}/chunks_${TAG}"
mkdir -p "${CHUNK_DIR}"
N_CHUNKS=$(( EPISODES / CHUNK ))

ENV_ARGS=()
[[ -n "${NUM_AGENTS:-}" ]] && ENV_ARGS+=(--num-agents "${NUM_AGENTS}")
[[ -n "${NUM_OBSTACLES:-}" ]] && ENV_ARGS+=(--num-obstacles "${NUM_OBSTACLES}")
[[ -n "${AREA_SIZE:-}" ]] && ENV_ARGS+=(--area-size "${AREA_SIZE}")
[[ -n "${MAX_STEP:-}" ]] && ENV_ARGS+=(--max-step "${MAX_STEP}")

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export PYTHONDONTWRITEBYTECODE=1

run_chunk() {
  local idx="$1"
  local seed=$(( START_SEED + idx * CHUNK ))
  "${PYTHON}" -u nav/scripts/evaluate_v6.py \
    --checkpoint "${CHECKPOINT}" \
    --h-checkpoint "${H}" \
    --num-episodes "${CHUNK}" --seed "${seed}" \
    --device "${DEVICE}" \
    "${ENV_ARGS[@]}" \
    --out "${CHUNK_DIR}/chunk_${idx}.json" \
    > "${CHUNK_DIR}/chunk_${idx}.log" 2>&1
}

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] eval ${EPISODES}ep ${CHECKPOINT} -> ${OUT}"
pids=()
for i in $(seq 0 $(( N_CHUNKS - 1 ))); do
  run_chunk "${i}" &
  pids+=("$!")
  if (( ${#pids[@]} >= PARALLEL )); then
    wait -n || true
    live=()
    for p in "${pids[@]}"; do
      kill -0 "${p}" 2>/dev/null && live+=("${p}")
    done
    pids=("${live[@]}")
  fi
done
for p in "${pids[@]}"; do
  wait "${p}" || true
done

# Retry any chunk that failed (e.g. a transient JAX crash), one at a time.
missing=()
for i in $(seq 0 $(( N_CHUNKS - 1 ))); do
  [[ -s "${CHUNK_DIR}/chunk_${i}.json" ]] || missing+=("$i")
done
for i in "${missing[@]}"; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] retry chunk ${i}" >&2
  run_chunk "${i}"
done
for i in $(seq 0 $(( N_CHUNKS - 1 ))); do
  if [[ ! -s "${CHUNK_DIR}/chunk_${i}.json" ]]; then
    echo "chunk ${i} is still missing; refusing to combine" >&2
    exit 3
  fi
done

"${PYTHON}" nav/scripts/combine_eval_chunks.py \
  --pattern "${CHUNK_DIR}/chunk_*.json" \
  --out "${OUT}/${TAG}.json" \
  > "${OUT}/combine_${TAG}.log" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] wrote ${OUT}/${TAG}.json"
