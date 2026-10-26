#!/usr/bin/env bash
# Official Ant V5 chain: learned cost+survival h -> frozen h -> QP-free policy.
#
# Required:
#   DATA=/path/to/safety_data.npz
# Optional:
#   TASK_CKPT=/path/to/mappo_actor_dir   # task-policy prior
#   PYTHON=/path/to/python
#   DEVICE=cuda
#
# The dataset must contain official cost labels and terminal labels; see
# nav/v5/README.md.  This script has no QP solver, QP label, or expert u_ref.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/nav"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
DATA="${DATA:-$ROOT/outputs/v5/official/ant/learned_h/safety_data.npz}"
H_OUT="${H_OUT:-$ROOT/outputs/v5/official/ant/learned_h}"
POLICY_OUT="${POLICY_OUT:-$ROOT/outputs/v5/official/ant/stage2_policy}"
TASK_CKPT="${TASK_CKPT:-$ROOT/outputs/v5/ant/baselines/lag_ft_2m_models}"

if [[ ! -f "$DATA" ]]; then
  echo "missing safety dataset: $DATA" >&2
  echo "See nav/v5/README.md for the .npz schema." >&2
  exit 2
fi

"$PYTHON" scripts/train_learned_h.py \
  --config configs/v5/learned_h_ant.yaml \
  --data "$DATA" \
  --out "$H_OUT" \
  --device "$DEVICE"

"$PYTHON" scripts/train_ant_qp_free.py \
  --out "$POLICY_OUT" \
  --ckpt "$TASK_CKPT" \
  --safety-h-checkpoint "$H_OUT/learned_h.pt" \
  --freeze-task \
  --gate-on-wall \
  --correction-radius 0.03 \
  --gate-margin 0.10 \
  --gate-tau 0.05 \
  --device "$DEVICE" \
  --seed 0 \
  --updates 20 \
  --rollout-steps 1024

echo "official learned-h V5 stage-2 output: $POLICY_OUT"
