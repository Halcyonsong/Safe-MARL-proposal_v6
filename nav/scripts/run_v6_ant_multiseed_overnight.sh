#!/usr/bin/env bash
# Overnight V6 Ant 10M multi-seed protocol.
#
# Runs the two matched arms from the seed-0 retention check:
#   - predictive h + safety-gated forward reward
#   - cost-only control with the same scheduled cost penalty
#
# Seeds 1 and 2 only.  Seed 0 already exists and is not overwritten.  All arms
# use 10M steps, 8 envs, 500k eval interval, 20 training-eval episodes, and the
# protocol eval seed 20000 + 100*seed.  After training, each final checkpoint
# is evaluated for 100 deterministic and 100 stochastic episodes.
#
# The four runs are independent and launched concurrently.  Ant uses CPU only,
# so this does not compete with the Stage-B nav actor for the GPU.  Nice keeps
# interactive/nav work responsive.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${ROOT}/nav${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

OUT_ROOT="${OUT_ROOT:-${ROOT}/outputs/v6/independent/ant}"
H_PRED="${H_PRED:-${OUT_ROOT}/h_predictive/ant_h.pkl}"
STEPS="${STEPS:-10000000}"
NICE="${NICE:-5}"
SEEDS="${SEEDS:-1 2}"

mkdir -p "${OUT_ROOT}"

run_one() {
  local arm="$1" seed="$2"
  local config out_name h_checkpoint log eval_seed
  local eval_seed=$((20000 + seed * 100))

  case "${arm}" in
    pred_gate)
      config="configs/v6/v6_ant_h_pred_gate_2m.yaml"
      out_name="mappo_h_pred_gate_10m"
      h_checkpoint="${H_PRED}"
      ;;
    cost_only)
      config="configs/v6/v6_ant_cost_ramp10_10m.yaml"
      out_name="mappo_cost_ramp10_10m"
      h_checkpoint=""
      ;;
    *)
      echo "unknown arm: ${arm}" >&2
      return 2
      ;;
  esac

  local out="${OUT_ROOT}/${out_name}/seed${seed}"
  log="${OUT_ROOT}/train_${arm}_10m_seed${seed}.log"
  mkdir -p "${out}"
  rm -f "${out}/FAILED"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launch ${arm} seed=${seed} eval_seed=${eval_seed} out=${out}"

  local h_args=()
  if [[ -n "${h_checkpoint}" ]]; then
    h_args=(--h-checkpoint "${h_checkpoint}")
  fi

  nice -n "${NICE}" "${PYTHON}" -u nav/scripts/train_v6_ant_mappo.py \
    --config "${config}" --out "${out}" \
    --seed "${seed}" --total-steps "${STEPS}" \
    --eval-interval 500000 --eval-episodes 20 \
    --eval-seed "${eval_seed}" --device cpu \
    "${h_args[@]}" > "${log}" 2>&1
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    echo "train failed: ${arm} seed=${seed} rc=${rc}" >&2
    touch "${out}/FAILED"
    return ${rc}
  fi

  if [[ ! -f "${out}/policy_final.pth" ]]; then
    echo "missing final checkpoint: ${out}" >&2
    touch "${out}/FAILED"
    return 1
  fi

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] evaluate det ${arm} seed=${seed}"
  nice -n "${NICE}" "${PYTHON}" -u nav/scripts/evaluate_v6_ant_mappo.py \
    --checkpoint "${out}/policy_final.pth" "${h_args[@]}" \
    --episodes 100 --seed "${eval_seed}" --device cpu \
    --out "${out}/eval_100ep_deterministic.json" \
    > "${out}/eval_100ep_deterministic.log" 2>&1 || {
      echo "det eval failed: ${arm} seed=${seed}" >&2
      touch "${out}/FAILED"
      return 1
    }

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] evaluate stoch ${arm} seed=${seed}"
  nice -n "${NICE}" "${PYTHON}" -u nav/scripts/evaluate_v6_ant_mappo.py \
    --checkpoint "${out}/policy_final.pth" "${h_args[@]}" \
    --episodes 100 --seed "${eval_seed}" --device cpu --stochastic \
    --out "${out}/eval_100ep_stochastic.json" \
    > "${out}/eval_100ep_stochastic.log" 2>&1 || {
      echo "stoch eval failed: ${arm} seed=${seed}" >&2
      touch "${out}/FAILED"
      return 1
    }

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] complete ${arm} seed=${seed}"
}

pids=()
for seed in ${SEEDS}; do
  run_one pred_gate "${seed}" &
  pids+=("$!")
  run_one cost_only "${seed}" &
  pids+=("$!")
done

echo "launched ${#pids[@]} runs; pids=${pids[*]}"
failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ ${failed} -ne 0 ]]; then
  echo "one or more Ant runs failed; inspect FAILED markers" >&2
  exit 1
fi
echo "all Ant multi-seed runs and 100-episode evaluations complete"
