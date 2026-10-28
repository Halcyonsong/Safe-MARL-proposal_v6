#!/usr/bin/env bash
# V6 navigation density screen: seed-0, 20 episodes, 4 key conditions.
#
# Conditions: default 8 agents/8 obstacles, 4 obstacles, 16 obstacles,
# 16 agents.  Policies: A@28k, A@16k (paired budget control), H0@16k.
# This is evaluation-only; no retraining.
set -u

ROOT="/home/xiang/workspace/python-project/Safe-MARL-proposal_v6"
cd "$ROOT"

PYTHON="${PYTHON:-/home/xiang/miniconda3/envs/safe_marl/bin/python}"
OUT_ROOT="${OUT_ROOT:-outputs/v6/independent/nav_density/seed0}"
EPISODES="${EPISODES:-20}"
START_SEED="${START_SEED:-30000}"
POLICIES="${POLICIES:-A28 A16 H0}"
CONDITIONS="${CONDITIONS:-8x8 4obs 16obs 16agents}"
LOG="$OUT_ROOT/density_screen.log"
mkdir -p "$OUT_ROOT"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

resolve_policy() {
  local policy="$1"
  case "$policy" in
    A28)
      CKPT="outputs/v6/independent/nav_stageb_repair_28k/seed0/policy_final.pth"
      H="outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl"
      ;;
    A16)
      CKPT="outputs/v6/independent/nav_stageb_repair_28k/seed0/policy_step_16384.pth"
      H="outputs/v6/independent/h_stageb_repair/v6_h_stageb_repair.pkl"
      ;;
    H0)
      CKPT="outputs/v6/independent/nav_h_gate_16k/seed0/policy_final.pth"
      H="outputs/v6/independent/h/v6_h.pkl"
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_condition() {
  local condition="$1"
  NUM_AGENTS=8
  NUM_OBSTACLES=8
  case "$condition" in
    8x8) ;;
    4obs) NUM_OBSTACLES=4 ;;
    16obs) NUM_OBSTACLES=16 ;;
    16agents) NUM_AGENTS=16 ;;
    *) return 1 ;;
  esac
}

run_one() {
  local policy="$1" condition="$2"
  resolve_policy "$policy" || { say "unknown policy $policy"; return 1; }
  resolve_condition "$condition" || { say "unknown condition $condition"; return 1; }
  local out="$OUT_ROOT/$policy/$condition"
  local tag="density_20ep_${condition}"
  if [ -s "$out/${tag}.json" ]; then
    say "skip existing $out/${tag}.json"
    return 0
  fi
  mkdir -p "$out"
  say "start $policy $condition agents=$NUM_AGENTS obstacles=$NUM_OBSTACLES"
  CHECKPOINT="$CKPT" H="$H" OUT="$out" TAG="$tag" \
    EPISODES="$EPISODES" CHUNK="$EPISODES" PARALLEL=1 \
    START_SEED="$START_SEED" NUM_AGENTS="$NUM_AGENTS" \
    NUM_OBSTACLES="$NUM_OBSTACLES" PYTHON="$PYTHON" \
    timeout 7200 bash nav/scripts/eval_v6_nav_100ep_chunked.sh \
    > "$out/eval.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    say "ERROR $policy $condition rc=$rc (see $out/eval.log)"
    return 1
  fi
  say "done $policy $condition -> $out/${tag}.json"
  return 0
}

say "density screen start policies='$POLICIES' conditions='$CONDITIONS'"
for policy in $POLICIES; do
  for condition in $CONDITIONS; do
    run_one "$policy" "$condition" || true
  done
done
say "density screen complete"
