#!/usr/bin/env bash
# V6 Ant protocol: V6-only h data -> frozen 9-D h -> from-scratch MAPPO arms.
# The corrected base configs match the upstream V5 Ant recipe (normalized
# 31-D centralized state, ValueNorm, 16 envs, 10M steps).  No V5/V4 weights
# or official h are used.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${ROOT}/nav${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

ROOT_OUT="${ROOT_OUT:-${ROOT}/outputs/v6/independent/ant}"
DATA="${DATA:-${ROOT_OUT}/h_data}"
H="${H:-${ROOT_OUT}/h}"
H_VALIDATION="${H_VALIDATION:-${ROOT_OUT}/h_validation}"
SEEDS="${SEEDS:-0}"
STEPS="${STEPS:-10000000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
ARMS="${ARMS:-task_only}"
SKIP_DATA="${SKIP_DATA:-}"
SKIP_H="${SKIP_H:-}"
SKIP_H_VALIDATION="${SKIP_H_VALIDATION:-}"
H_VALIDATION_EPISODES="${H_VALIDATION_EPISODES:-50}"

# A task-only/cost-only run does not need to rebuild/validate h.  When h_only
# is in ARMS, keep the full data -> h -> validation chain by default.
if [[ " ${ARMS} " != *" h_only "* ]]; then
  SKIP_DATA="${SKIP_DATA:-1}"
  SKIP_H="${SKIP_H:-1}"
  SKIP_H_VALIDATION="${SKIP_H_VALIDATION:-1}"
else
  SKIP_DATA="${SKIP_DATA:-0}"
  SKIP_H="${SKIP_H:-0}"
  SKIP_H_VALIDATION="${SKIP_H_VALIDATION:-0}"
fi

mkdir -p "${ROOT_OUT}"
if [[ "${SKIP_DATA}" != "1" ]]; then
  "${PYTHON}" nav/scripts/collect_v6_ant_h_data.py \
    --out "${DATA}" --episodes "${H_EPISODES:-200}" --episode-limit 1000 \
    --horizon 32 --aug-rows "${AUG_ROWS:-20000}" \
    --perturb-rows "${PERTURB_ROWS:-10000}" --seed "${H_SEED:-0}"
fi
if [[ "${SKIP_H}" != "1" ]]; then
  "${PYTHON}" nav/scripts/train_v6_ant_h.py \
    --data-dir "${DATA}" --out "${H}" --epochs "${H_EPOCHS:-40}" \
    --hidden 128,128 --seed "${H_SEED:-0}" --device "${DEVICE:-auto}"
fi
if [[ "${SKIP_H_VALIDATION}" != "1" ]]; then
  echo "validating frozen Ant h on fresh V6 rollouts before MAPPO"
  "${PYTHON}" nav/scripts/validate_v6_ant_h.py \
    --h-checkpoint "${H}/ant_h.pkl" --out "${H_VALIDATION}" \
    --episodes "${H_VALIDATION_EPISODES}" --horizon 32 \
    --device "${DEVICE:-auto}"
fi

for seed in ${SEEDS}; do
  for arm in ${ARMS}; do
    case "${arm}" in
      h_only)
        config="configs/v6/v6_ant_h_p01_10m.yaml"
        h_args=(--h-checkpoint "${H}/ant_h.pkl")
        out_name="mappo_h_p01_10m"
        ;;
      cost_only)
        config="configs/v6/v6_ant_cost_p01_10m.yaml"
        h_args=()
        out_name="mappo_cost_p01_10m"
        ;;
      task_only)
        config="configs/v6/v6_ant_task_base.yaml"
        h_args=()
        out_name="mappo_task_base"
        ;;
      *)
        echo "unknown Ant arm: ${arm}" >&2
        exit 2
        ;;
    esac
    out="${ROOT_OUT}/${out_name}/seed${seed}"
    mkdir -p "${out}"
    echo "launching Ant ${arm} seed ${seed} -> ${out}"
    "${PYTHON}" nav/scripts/train_v6_ant_mappo.py \
      --config "${config}" --out "${out}" "${h_args[@]}" \
      --seed "${seed}" --total-steps "${STEPS}" \
      --eval-interval "${EVAL_INTERVAL}" \
      --eval-episodes "${EVAL_EPISODES}" \
      --eval-seed "$((20000 + seed * 100))" \
      --device "${DEVICE:-auto}"
  done
done
