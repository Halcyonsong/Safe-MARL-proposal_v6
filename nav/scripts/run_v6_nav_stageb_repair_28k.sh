#!/usr/bin/env bash
# Stage-B repair actor, 28k budget extension (A cell only).
#
# Trains seeds 0/1/2 from scratch with the repaired V6 h and the frozen
# progress gate, then runs the formal 100-episode deterministic evaluation on
# (a) the 28,672-step final checkpoint (primary endpoint) and (b) the same-run
# 16,384-step checkpoint (paired budget control).  No checkpoint selection.
#
# Preconditions:
#   - outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl
#   - outputs/v6/independent/h_stageb_repair/gate_calibration.json
#   - nav/configs/v6/v6_nav_stageb_repair_28k.yaml
#
# Usage:
#   bash nav/scripts/run_v6_nav_stageb_repair_28k.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/v6/v6_nav_stageb_repair_28k.yaml}"
H="${H:-outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl}"
CALIB="${CALIB:-outputs/v6/independent/h_stageb_repair/gate_calibration.json}"
BASE="${BASE:-outputs/v6/independent/nav_stageb_repair_28k}"
TRAIN_PARALLEL="${TRAIN_PARALLEL:-3}"
EVAL_PARALLEL="${EVAL_PARALLEL:-5}"
SEEDS=(0 1 2)
LOG_DIR="${BASE}/logs"
mkdir -p "${LOG_DIR}"

if [[ ! -f "${H}" ]]; then
  echo "missing repaired h checkpoint: ${H}" >&2
  exit 1
fi
if [[ ! -f "${CALIB}" ]]; then
  echo "missing gate calibration: ${CALIB}" >&2
  exit 1
fi

# ---- training -------------------------------------------------------------
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] train 28k seeds ${SEEDS[*]}, parallel=${TRAIN_PARALLEL}"
pids=()
for seed in "${SEEDS[@]}"; do
  out="${BASE}/seed${seed}"
  mkdir -p "${out}"
  if [[ -f "${out}/policy_final.pth" ]]; then
    echo "skip completed training: seed${seed}"
    continue
  fi
  (
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] train seed=${seed}"
    "${PYTHON}" -u nav/scripts/train_v6.py \
      --config "${CONFIG}" --out "${out}" --seed "${seed}" --device cuda \
      > "${out}/train.log" 2>&1
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] train done seed=${seed}"
  ) > "${LOG_DIR}/train_seed${seed}.log" 2>&1 &
  pids+=("$!")
  if (( ${#pids[@]} >= TRAIN_PARALLEL )); then
    wait -n || echo "warning: a training job exited nonzero" >&2
    live=()
    for p in "${pids[@]}"; do
      kill -0 "${p}" 2>/dev/null && live+=("${p}")
    done
    pids=("${live[@]}")
  fi
done
for p in "${pids[@]}"; do
  wait "${p}" || echo "warning: a training job exited nonzero" >&2
done

missing_train=()
for seed in "${SEEDS[@]}"; do
  out="${BASE}/seed${seed}"
  [[ -f "${out}/policy_final.pth" ]] || missing_train+=("seed${seed}:final")
  [[ -f "${out}/policy_step_16384.pth" ]] || missing_train+=("seed${seed}:step16384")
done
if (( ${#missing_train[@]} > 0 )); then
  echo "incomplete training runs: ${missing_train[*]}" >&2
  exit 3
fi
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] training complete"

# ---- formal evaluation (fixed checkpoints only) ---------------------------
eval_one() {
  local seed="$1" ckpt="$2" tag="$3"
  local out="${BASE}/seed${seed}"
  if [[ -s "${out}/${tag}.json" ]]; then
    echo "skip completed eval: ${tag} seed${seed}"
    return
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] eval ${tag} seed${seed}"
  CHECKPOINT="${ckpt}" H="${H}" OUT="${out}" TAG="${tag}" \
    PARALLEL="${EVAL_PARALLEL}" PYTHON="${PYTHON}" \
    bash nav/scripts/eval_v6_nav_100ep_chunked.sh \
    > "${LOG_DIR}/eval_${tag}_seed${seed}.log" 2>&1
}

for seed in "${SEEDS[@]}"; do
  eval_one "${seed}" "${BASE}/seed${seed}/policy_final.pth" "eval_100ep_final"
  eval_one "${seed}" "${BASE}/seed${seed}/policy_step_16384.pth" "eval_100ep_step16384"
done

missing_eval=()
for seed in "${SEEDS[@]}"; do
  out="${BASE}/seed${seed}"
  [[ -s "${out}/eval_100ep_final.json" ]] || missing_eval+=("seed${seed}:final")
  [[ -s "${out}/eval_100ep_step16384.json" ]] || missing_eval+=("seed${seed}:step16384")
done
if (( ${#missing_eval[@]} > 0 )); then
  echo "incomplete evaluations: ${missing_eval[*]}" >&2
  exit 4
fi

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] 28k A run complete: ${BASE}"
