#!/usr/bin/env bash
# Reproducible V6 h-only formal run: 3 training seeds, fixed 100-episode eval.
#
# This runner intentionally uses only V6-owned data/h/checkpoints.  It trains
# the actor from scratch, uses the final checkpoint (not a 20-episode "best"
# checkpoint), and evaluates every seed on the same eval seed range.
#
# Usage:
#   bash scripts/run_v6_formal_3seed.sh
#   REUSE_SEED0=0 bash scripts/run_v6_formal_3seed.sh   # retrain seed 0 too
#
# Existing complete seed directories are skipped, so the script is resumable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/nav"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/v6/v6_nav_h_only_16k.yaml}"
H_CHECKPOINT="${H_CHECKPOINT:-$ROOT/outputs/v6/independent/h/v6_h.pkl}"
OUT="${OUT:-$ROOT/outputs/v6/independent/formal_3seed/h_only}"
SEEDS="${SEEDS:-0 1 2}"
TRAIN_STEPS="${TRAIN_STEPS:-16384}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
EVAL_SEED="${EVAL_SEED:-10000}"
DEVICE="${DEVICE:-auto}"
REUSE_SEED0="${REUSE_SEED0:-1}"
SOURCE_SEED0="${SOURCE_SEED0:-$ROOT/outputs/v6/independent/single_seed_16k/h_only}"
TRAIN_EXTRA="${TRAIN_EXTRA:-}"
export REUSE_SEED0 SOURCE_SEED0 TRAIN_EXTRA
read -r -a TRAIN_EXTRA_ARGS <<< "$TRAIN_EXTRA"

mkdir -p "$OUT"

"$PYTHON" scripts/check_v6_independence.py

"$PYTHON" - "$OUT" "$CONFIG" "$H_CHECKPOINT" "$SEEDS" \
  "$TRAIN_STEPS" "$EVAL_EPISODES" "$EVAL_SEED" <<'PY'
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

out, config, h_checkpoint, seeds, steps, episodes, eval_seed = sys.argv[1:]
root = Path.cwd().parent


def sha256(path: Path):
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


data = root / 'outputs/v6/independent/data'
payload = {
    'schema': 'v6-formal-3seed.v1',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'purpose': ('reproducible h-only baseline; frozen V6 h, from-scratch '
                'actor, no task channel, final checkpoint, fixed eval seeds'),
    'config': str(Path(config).resolve()),
    'config_sha256': sha256(Path(config).resolve()),
    'h_checkpoint': str(Path(h_checkpoint).resolve()),
    'h_sha256': sha256(Path(h_checkpoint).resolve()),
    'data_sha256': {
        name: sha256(data / name)
        for name in ('train_merged.npz', 'val_data.npz', 'test_data.npz')
    },
    'seeds': [int(v) for v in seeds.split()],
    'train_steps': int(steps),
    'eval_episodes': int(episodes),
    'eval_seed': int(eval_seed),
    'python': sys.version,
    'platform': platform.platform(),
    'versions': {name: version(name)
                 for name in ('torch', 'jax', 'jaxlib', 'numpy')},
    'reuse_seed0': os.environ.get('REUSE_SEED0', '1'),
    'source_seed0': os.environ.get('SOURCE_SEED0'),
    'train_extra': os.environ.get('TRAIN_EXTRA', ''),
}
Path(out, 'manifest.json').write_text(json.dumps(payload, indent=2) + '\n')
print(json.dumps(payload, indent=2))
PY

for seed in $SEEDS; do
  RUN_DIR="$OUT/seed${seed}"
  mkdir -p "$RUN_DIR"
  EVAL_FILE="$RUN_DIR/eval_${EVAL_EPISODES}ep_final.json"

  if [ "$seed" = "0" ] && [ "$REUSE_SEED0" = "1" ] \
      && [ ! -f "$RUN_DIR/policy_final.pth" ] \
      && [ -f "$SOURCE_SEED0/policy_final.pth" ]; then
    echo "=== reuse V6 h-only seed0 from $SOURCE_SEED0 ==="
    cp "$SOURCE_SEED0/policy_final.pth" "$RUN_DIR/policy_final.pth"
    if [ -f "$SOURCE_SEED0/eval_${EVAL_EPISODES}ep_final.json" ]; then
      cp "$SOURCE_SEED0/eval_${EVAL_EPISODES}ep_final.json" "$EVAL_FILE"
    fi
    if [ -f "$SOURCE_SEED0/summary.json" ]; then
      cp "$SOURCE_SEED0/summary.json" "$RUN_DIR/summary.json"
    fi
    printf '%s\n' "reused_from=$SOURCE_SEED0" > "$RUN_DIR/reuse.txt"
  fi

  if [ ! -f "$RUN_DIR/policy_final.pth" ]; then
    echo "=== V6 formal h-only train seed=$seed ==="
    "$PYTHON" scripts/train_v6.py \
      --config "$CONFIG" \
      --h-checkpoint "$H_CHECKPOINT" \
      --seed "$seed" \
      --total-steps "$TRAIN_STEPS" \
      --rollout-steps 128 \
      --eval-interval 0 \
      --eval-episodes 20 \
      --eval-seed "$EVAL_SEED" \
      --save-interval 0 \
      --skip-final-eval \
      --deterministic \
      --device "$DEVICE" \
      --out "$RUN_DIR" \
      "${TRAIN_EXTRA_ARGS[@]}" 2>&1 | tee "$RUN_DIR/train.log"
  fi

  if [ ! -f "$EVAL_FILE" ]; then
    echo "=== V6 formal 100ep eval seed=$seed ==="
    "$PYTHON" scripts/evaluate_v6.py \
      --checkpoint "$RUN_DIR/policy_final.pth" \
      --h-checkpoint "$H_CHECKPOINT" \
      --num-episodes "$EVAL_EPISODES" \
      --seed "$EVAL_SEED" \
      --device "$DEVICE" \
      --out "$EVAL_FILE" 2>&1 | tee "$RUN_DIR/eval_${EVAL_EPISODES}ep.log"
  fi
done

"$PYTHON" scripts/summarize_v6_formal.py \
  --dir "$OUT" \
  --out "$OUT/aggregate_${EVAL_EPISODES}ep.json"
