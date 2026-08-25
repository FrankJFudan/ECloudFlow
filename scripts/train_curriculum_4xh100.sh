#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 MANIFEST_JSON OUTPUT_ROOT" >&2
  exit 2
fi

MANIFEST="$1"
OUTPUT_ROOT="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -A STEPS=(
  [1]="${ECLOUDFLOW_STAGE1_STEPS:-100000}"
  [2]="${ECLOUDFLOW_STAGE2_STEPS:-300000}"
  [3]="${ECLOUDFLOW_STAGE3_STEPS:-500000}"
  [4]="${ECLOUDFLOW_STAGE4_STEPS:-100000}"
)

previous=""
for stage in 1 2 3 4; do
  output="$OUTPUT_ROOT/stage${stage}"
  args=(
    --manifest "$MANIFEST"
    --output "$output"
    --stage "$stage"
    --max-steps "${STEPS[$stage]}"
  )
  if [[ -n "$previous" ]]; then
    args+=(--init-from "$previous")
  fi
  bash "$SCRIPT_DIR/train_4xh100.sh" "${args[@]}"
  previous="$output/checkpoints/last.ckpt"
  if [[ ! -f "$previous" ]]; then
    echo "stage $stage did not publish $previous" >&2
    exit 1
  fi
done

echo "Curriculum completed. Final checkpoint: $previous"
