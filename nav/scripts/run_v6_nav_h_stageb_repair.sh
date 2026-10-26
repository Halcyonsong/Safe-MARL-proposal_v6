#!/usr/bin/env bash
# Train the repaired V6 navigation Stage-B h after the Ant CPU runs finish.
#
# The Ant 10M multi-seed runner uses CPU; the repaired h trainer uses JAX CPU.
# Waiting for the Ant runner (including its 100-episode final evaluations)
# avoids starving those formal runs.  The nav actor/gate-off control uses the
# GPU and does not conflict.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
# train_v6_official_h.py resolves relative config paths against nav/, while
# ``--out`` is resolved against the repository root.
CONFIG="${CONFIG:-configs/v6/v6_nav_h_stageb_repair.yaml}"
OUT="${OUT:-outputs/v6/independent/h_stageb_repair}"
LOG="${LOG:-outputs/v6/independent/nav_h_stageb_repair.log}"
SEED="${SEED:-0}"

while pgrep -f 'run_v6_ant_multiseed_overnight.sh' >/dev/null 2>&1; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for Ant runner" >> "${LOG}"
  sleep 60
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] train repaired Stage-B h seed=${SEED}" >> "${LOG}"
"${PYTHON}" -u nav/scripts/train_v6_official_h.py \
  --config "${CONFIG}" --out "${OUT}" --seed "${SEED}" >> "${LOG}" 2>&1
rc=$?
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] repaired h finished rc=${rc}" >> "${LOG}"
exit ${rc}
