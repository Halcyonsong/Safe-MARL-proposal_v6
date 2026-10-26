#!/usr/bin/env bash
# Diagnostic control for Stage B: same Stage-B h, same actor budget, but the
# task gate is disabled (`high <= low`).  This separates "the new h features
# are harmful" from "the gate suppresses too much progress".  It is not a
# threshold sweep and is not a promoted arm.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SEED="${SEED:-0}"
H="${H:-outputs/v6/independent/h_stageb/v6_h_stageb.pkl}"
OUT="${OUT:-outputs/v6/independent/nav_stageb_gateoff_16k/seed${SEED}}"

if [[ ! -f "${H}" ]]; then
  echo "missing Stage-B h checkpoint: ${H}" >&2
  exit 1
fi
mkdir -p "${OUT}"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] train Stage-B gate-off control seed=${SEED}"
"${PYTHON}" -u nav/scripts/train_v6.py \
  --config configs/v6/v6_nav_stageb_16k.yaml --out "${OUT}" \
  --seed "${SEED}" --device cuda --task-gate-high 0.0 \
  > "${OUT}/train.log" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] 100ep deterministic eval"
"${PYTHON}" -u nav/scripts/evaluate_v6.py \
  --checkpoint "${OUT}/policy_final.pth" \
  --h-checkpoint "${H}" \
  --num-episodes 100 --seed 10000 --device cuda \
  --out "${OUT}/eval_100ep_final.json" \
  > "${OUT}/eval_100ep.log" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done: ${OUT}"
