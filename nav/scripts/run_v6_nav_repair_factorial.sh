#!/usr/bin/env bash
# P1 repaired-h 2x2 factorial orchestration: train the three missing cells
# (B/C/D) for seeds 0/1/2, then run the formal 100-episode deterministic
# evaluation for every cell and seed.
#
# A = repaired h ON, gate ON is the existing nav_stageb_repair_16k row and is
# not retrained here.  See run_v6_nav_repair_factorial_arm.sh for cell details.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
H="${H:-outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl}"
BASE="${BASE:-outputs/v6/independent/nav_repair_factorial}"
TRAIN_PARALLEL="${TRAIN_PARALLEL:-3}"
EVAL_PARALLEL="${EVAL_PARALLEL:-5}"
LOG_DIR="${BASE}/logs"
mkdir -p "${LOG_DIR}"

ARMS=(B_h_on_gate_off C_h_off_gate_on D_h_off_gate_off)
SEEDS=(0 1 2)

if [[ ! -f "${H}" ]]; then
  echo "missing repaired h checkpoint: ${H}" >&2
  exit 1
fi

# ---- training -------------------------------------------------------------
pending=()
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="${BASE}/${arm}/seed${seed}"
    if [[ -f "${out}/policy_final.pth" ]]; then
      echo "skip completed training: ${arm} seed${seed}"
    else
      pending+=("${arm}:${seed}")
    fi
  done
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] training ${#pending[@]} runs, parallel=${TRAIN_PARALLEL}"
pids=()
for job in "${pending[@]}"; do
  arm="${job%%:*}"
  seed="${job##*:}"
  ARM="${arm}" SEED="${seed}" H="${H}" BASE="${BASE}" PYTHON="${PYTHON}" \
    bash nav/scripts/run_v6_nav_repair_factorial_arm.sh \
    > "${LOG_DIR}/train_${arm}_seed${seed}.log" 2>&1 &
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
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    [[ -f "${BASE}/${arm}/seed${seed}/policy_final.pth" ]] || \
      missing_train+=("${arm}:${seed}")
  done
done
if (( ${#missing_train[@]} > 0 )); then
  echo "incomplete training runs: ${missing_train[*]}" >&2
  exit 3
fi
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] training complete"

# ---- formal evaluation ----------------------------------------------------
for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    out="${BASE}/${arm}/seed${seed}"
    if [[ -s "${out}/eval_100ep_final.json" ]]; then
      echo "skip completed eval: ${arm} seed${seed}"
      continue
    fi
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] eval ${arm} seed${seed}"
    CHECKPOINT="${out}/policy_final.pth" \
      H="${H}" OUT="${out}" TAG="eval_100ep_final" \
      PARALLEL="${EVAL_PARALLEL}" PYTHON="${PYTHON}" \
      bash nav/scripts/eval_v6_nav_100ep_chunked.sh \
      > "${LOG_DIR}/eval_${arm}_seed${seed}.log" 2>&1
  done
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] P1 factorial complete: ${BASE}"
