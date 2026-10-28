#!/usr/bin/env bash
# Ant 10M from-scratch MAPPO-Lagrangian external baseline (upstream trainer).
#
# This is NOT a V6 main-method run.  It trains the upstream mappo_lagr recipe
# from randomly initialized weights for 10M environment steps and copies the
# final actor checkpoints into the V6 output tree for the V6-compatible
# evaluation adapter.
#
# Usage:
#   TRAIN_SEED=0 STEPS=10000000 bash nav/scripts/run_ant_lag_10m.sh
#   STEPS=32000 EXPERIMENT=v6_lag_smoke bash nav/scripts/run_ant_lag_10m.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/xiang/miniconda3/envs/mappo_lag/bin/python}"
V2="${SAFE_MAMUJOCO_ROOT:-${ROOT}/../Safe-MARL-proposal_v2}"
PKG="${V2}/third_party/mappo_lag/MAPPO-Lagrangian"
TRAIN_DIR="${PKG}/mappo_lagrangian/scripts"

TRAIN_SEED="${TRAIN_SEED:-0}"
STEPS="${STEPS:-10000000}"
EXPERIMENT="${EXPERIMENT:-v6_lag_10m_seed${TRAIN_SEED}}"
OUT="${OUT:-${ROOT}/outputs/v6/independent/ant/mappo_lag_10m/seed${TRAIN_SEED}}"
N_ROLLOUT="${N_ROLLOUT:-16}"
N_THREADS="${N_THREADS:-4}"
RUN_ROOT="${TRAIN_DIR}/results/mujoco/Ant-v2/mappo_lagr/${EXPERIMENT}"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing to train: upstream experiment dir already exists: ${RUN_ROOT}" >&2
  echo "choose a new EXPERIMENT name; this runner never resumes or overwrites." >&2
  exit 2
fi
if [[ ! -d "${PKG}" ]]; then
  echo "upstream MAPPO-Lag package not found: ${PKG}" >&2
  exit 2
fi

mkdir -p "${OUT}"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PKG}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

cat > "${OUT}/command.txt" <<EOF
TRAIN_SEED=${TRAIN_SEED}
STEPS=${STEPS}
EXPERIMENT=${EXPERIMENT}
N_ROLLOUT=${N_ROLLOUT}
N_THREADS=${N_THREADS}
RUN_ROOT=${RUN_ROOT}
EOF

echo "launching upstream Ant MAPPO-Lag: seed=${TRAIN_SEED} steps=${STEPS}" \
  "experiment=${EXPERIMENT} rollout_threads=${N_ROLLOUT}"
cd "${ROOT}"
"${PYTHON}" nav/scripts/train_ant_lag_upstream.py \
  --env_name mujoco \
  --algorithm_name mappo_lagr \
  --experiment_name "${EXPERIMENT}" \
  --scenario Ant-v2 \
  --agent_conf 2x4 \
  --agent_obsk 1 \
  --lr 9e-5 \
  --critic_lr 5e-3 \
  --std_x_coef 1 \
  --std_y_coef 5e-1 \
  --seed "${TRAIN_SEED}" \
  --n_training_threads "${N_THREADS}" \
  --n_rollout_threads "${N_ROLLOUT}" \
  --num_mini_batch 40 \
  --episode_length 1000 \
  --num_env_steps "${STEPS}" \
  --ppo_epoch 5 \
  --use_value_active_masks \
  --add_center_xy \
  --use_state_agent \
  --safety_bound 0.2 \
  --lamda_lagr 0.78 \
  --lagrangian_coef_rate 1e-7 \
  > "${OUT}/train.log" 2>&1

RUN_DIR="$(find "${RUN_ROOT}" -maxdepth 1 -type d -name 'run*' | sort -V | tail -1)"
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}/models" ]]; then
  echo "training finished but no models directory found under ${RUN_ROOT}" >&2
  exit 3
fi

rm -rf "${OUT}/models"
cp -r "${RUN_DIR}/models" "${OUT}/models"
cp "${RUN_DIR}/logs/summary.json" "${OUT}/upstream_summary.json" 2>/dev/null || true
printf '%s\n' "${RUN_DIR}" > "${OUT}/upstream_run_dir.txt"
echo "done: copied ${RUN_DIR}/models -> ${OUT}/models"
