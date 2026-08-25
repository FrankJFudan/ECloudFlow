#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MANIFEST_JSON OUTPUT_DIR" >&2
  exit 2
fi

MANIFEST="$(realpath "$1")"
OUTPUT="$(realpath -m "$2")"
SHARD_DIR="$(dirname "$MANIFEST")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export ECLOUDFLOW_REQUIRE_NCCL=1

ecloudflow doctor --server --dataset "$MANIFEST" --output-dir "$OUTPUT/doctor"
python -m pytest tests/distributed/test_nccl_training.py -q

ecloudflow train +experiment=h100_smoke \
  --output-dir "$OUTPUT/initial" \
  --max-steps 2 \
  --override "data.manifest=$MANIFEST" \
  --override "data.shard_dir=$SHARD_DIR"

checkpoint="$OUTPUT/initial/checkpoints/last.ckpt"
test -f "$checkpoint"

ecloudflow train +experiment=h100_smoke \
  --output-dir "$OUTPUT/resumed" \
  --resume-from "$checkpoint" \
  --max-steps 3 \
  --override "data.manifest=$MANIFEST" \
  --override "data.shard_dir=$SHARD_DIR" \
  --override trainer.max_epochs=2

test -f "$OUTPUT/resumed/checkpoints/last.ckpt"
echo "Four-GPU train/checkpoint/resume acceptance completed: $OUTPUT"
