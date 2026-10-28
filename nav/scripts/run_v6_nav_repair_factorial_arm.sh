#!/usr/bin/env bash
# P1 repaired-h 2x2 factorial: train one missing cell from scratch.
#
# A (reference, already complete): repaired h ON,  task gate ON
# B: repaired h ON,  task gate OFF
# C: repaired h OFF (actor h/grad channels zeroed), task gate ON
# D: repaired h OFF (actor h/grad channels zeroed), task gate OFF
#
# The h network is the same frozen repaired checkpoint in every cell; C/D only
# remove h/grad_h from the actor input.  The gate (when ON) still uses the
# repaired h's h_next, so C is "gate-only, no h actor input".
#
# Protocol: 16,384 steps, seed 0/1/2, fixed unsafe penalty 10, actor from
# scratch.  No official h, cbf.pkl, QP, u_ref, or V5/V4 artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SEED="${SEED:-0}"
ARM="${ARM:?set ARM to B_h_on_gate_off, C_h_off_gate_on, or D_h_off_gate_off}"
H="${H:-outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl}"
BASE="${BASE:-outputs/v6/independent/nav_repair_factorial}"
OUT="${OUT:-${BASE}/${ARM}/seed${SEED}}"
CONFIG="${CONFIG:-configs/v6/v6_nav_stageb_repair_16k.yaml}"

if [[ ! -f "${H}" ]]; then
  echo "missing repaired h checkpoint: ${H}" >&2
  exit 1
fi
if [[ -f "${OUT}/policy_final.pth" ]]; then
  echo "refusing to overwrite completed run: ${OUT}/policy_final.pth" >&2
  exit 2
fi
mkdir -p "${OUT}"

case "${ARM}" in
  B_h_on_gate_off)
    ARM_ARGS=(--task-gate-high 0.0)
    ;;
  C_h_off_gate_on)
    ARM_ARGS=(--no-use-h-features)
    ;;
  D_h_off_gate_off)
    ARM_ARGS=(--no-use-h-features --task-gate-high 0.0)
    ;;
  *)
    echo "unknown ARM=${ARM}" >&2
    exit 2
    ;;
esac

cat > "${OUT}/command.txt" <<EOF
ARM=${ARM}
SEED=${SEED}
CONFIG=${CONFIG}
H=${H}
TOTAL_STEPS=16384
UNSAFE_PENALTY=10.0
EXTRA_ARGS=${ARM_ARGS[*]}
EOF

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] train P1 ${ARM} seed=${SEED}"
"${PYTHON}" -u nav/scripts/train_v6.py \
  --config "${CONFIG}" \
  --out "${OUT}" \
  --seed "${SEED}" \
  --device cuda \
  --total-steps 16384 \
  --unsafe-penalty 10.0 \
  --h-checkpoint "${H}" \
  "${ARM_ARGS[@]}" \
  > "${OUT}/train.log" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done: ${OUT}/policy_final.pth"
