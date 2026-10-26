#!/usr/bin/env bash
# Stage-B repair actor: train from scratch with the repaired V6 h, then run
# the formal 100-episode deterministic evaluation.
#
# Preconditions:
#   - outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl
#   - outputs/v6/independent/h_stageb_repair/gate_calibration.json
#   - nav/configs/v6/v6_nav_stageb_repair_16k.yaml has the repaired h path and
#     the calibrated task_gate_high
#
# Seed 0 is run by default.  Seeds 1/2 are not part of the seed-0 repair gate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SEED="${SEED:-0}"
CONFIG="${CONFIG:-configs/v6/v6_nav_stageb_repair_16k.yaml}"
H="${H:-outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl}"
CALIB="${CALIB:-outputs/v6/independent/h_stageb_repair/gate_calibration.json}"
OUT="${OUT:-outputs/v6/independent/nav_stageb_repair_16k/seed${SEED}}"

if [[ ! -f "${H}" ]]; then
  echo "missing repaired h checkpoint: ${H}" >&2
  exit 1
fi
if [[ ! -f "${CALIB}" ]]; then
  echo "missing gate calibration: ${CALIB}" >&2
  exit 1
fi
mkdir -p "${OUT}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] train Stage-B repair actor seed=${SEED}"
"${PYTHON}" -u nav/scripts/train_v6.py \
  --config "${CONFIG}" --out "${OUT}" --seed "${SEED}" --device cuda \
  > "${OUT}/train.log" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] formal 100ep deterministic eval"
"${PYTHON}" -u nav/scripts/evaluate_v6.py \
  --checkpoint "${OUT}/policy_final.pth" \
  --h-checkpoint "${H}" \
  --num-episodes 100 --seed 10000 --device cuda \
  --out "${OUT}/eval_100ep_final.json" \
  > "${OUT}/eval_100ep.log" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done: ${OUT}"
