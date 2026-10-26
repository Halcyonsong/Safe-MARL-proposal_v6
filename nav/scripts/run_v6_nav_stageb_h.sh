#!/usr/bin/env bash
# Stage B: train the new V6 navigation h from scratch and calibrate its gate.
#
# Run only after outputs/v6/independent/data_stageb/summary.json exists.  This
# script does not train the actor and does not edit any config; the calibrated
# gate values are inspected first and then written into
# nav/configs/v6/v6_nav_stageb_16k.yaml by hand.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
DATA="${DATA:-outputs/v6/independent/data_stageb}"
OUT="${OUT:-outputs/v6/independent/h_stageb}"

if [[ ! -f "${DATA}/summary.json" ]]; then
  echo "missing ${DATA}/summary.json; Stage B collection is not complete" >&2
  exit 1
fi
mkdir -p "${OUT}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] train Stage B h"
"${PYTHON}" -u nav/scripts/train_v6_official_h.py \
  --config configs/v6/v6_nav_h_stageb.yaml \
  --train-data "${DATA}/train_data.npz" \
  --val-data "${DATA}/val_data.npz" \
  --test-data "${DATA}/test_data.npz" \
  --out "${OUT}" \
  > "${OUT}/train.log" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] calibrate Stage B gate"
"${PYTHON}" -u nav/scripts/calibrate_v6_nav_gate.py \
  --checkpoint "${OUT}/v6_h_stageb.pkl" \
  --data "${DATA}/val_data.npz" \
  --out "${OUT}/gate_calibration.json" \
  > "${OUT}/gate_calibration.log" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done"
