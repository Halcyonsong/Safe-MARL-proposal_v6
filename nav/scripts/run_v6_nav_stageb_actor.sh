#!/usr/bin/env bash
# Stage B actor: train from scratch with the frozen Stage-B h, then run the
# formal 100-episode deterministic evaluation.
#
# Preconditions:
#   - outputs/v6/independent/h_stageb/v6_h_stageb.pkl exists
#   - outputs/v6/independent/h_stageb/gate_calibration.json exists
#   - nav/configs/v6/v6_nav_stageb_16k.yaml has the calibrated task_gate_high
#
# Seed 0 is run by default.  Seeds 1/2 are only for after seed 0 passes the
# safety and task guardrails.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SEED="${SEED:-0}"
CONFIG="${CONFIG:-configs/v6/v6_nav_stageb_16k.yaml}"
H="${H:-outputs/v6/independent/h_stageb/v6_h_stageb.pkl}"
OUT="${OUT:-outputs/v6/independent/nav_stageb_16k/seed${SEED}}"

if [[ ! -f "${H}" ]]; then
  echo "missing Stage-B h checkpoint: ${H}" >&2
  exit 1
fi
if [[ ! -f "outputs/v6/independent/h_stageb/gate_calibration.json" ]]; then
  echo "missing gate calibration; run run_v6_nav_stageb_h.sh first" >&2
  exit 1
fi
mkdir -p "${OUT}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] train Stage B actor seed=${SEED}"
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
