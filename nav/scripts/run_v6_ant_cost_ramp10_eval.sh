#!/usr/bin/env bash
# Wait for the cost-only ramp-to-1.0 control run, then evaluate its final
# checkpoint on the matched 100-episode deterministic/stochastic protocol.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${ROOT}/nav${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
DIR="${DIR:-${ROOT}/outputs/v6/independent/ant/mappo_cost_ramp10_10m/seed0}"
while pgrep -f "train_v6_ant_mappo.py --config configs/v6/v6_ant_cost_ramp10_10m.yaml" >/dev/null; do
  date -u +"waiting cost_ramp10 %Y-%m-%dT%H:%M:%SZ"
  sleep 120
done
CKPT="$DIR/policy_final.pth"
if [ ! -f "$CKPT" ]; then
  echo "missing checkpoint $CKPT" >&2
  exit 1
fi
"$PYTHON" nav/scripts/evaluate_v6_ant_mappo.py \
  --checkpoint "$CKPT" --episodes 100 --seed 20000 --device cpu \
  --out "$DIR/eval_100ep_deterministic.json"
"$PYTHON" nav/scripts/evaluate_v6_ant_mappo.py \
  --checkpoint "$CKPT" --episodes 100 --seed 20000 --device cpu --stochastic \
  --out "$DIR/eval_100ep_stochastic.json"
echo "cost_ramp10 final evaluation complete"
