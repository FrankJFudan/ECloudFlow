#!/usr/bin/env bash
set -euo pipefail

MANIFEST=""
OUTPUT=""
STAGE="3"
INIT_FROM=""
RESUME_FROM=""
MAX_STEPS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --init-from) INIT_FROM="$2"; shift 2 ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$MANIFEST" || ! -f "$MANIFEST" ]]; then
  echo "--manifest must name an existing ECloudFlow manifest.json" >&2
  exit 2
fi
if [[ -z "$OUTPUT" ]]; then
  echo "--output is required" >&2
  exit 2
fi
if [[ ! "$STAGE" =~ ^[1-4]$ ]]; then
  echo "--stage must be 1, 2, 3, or 4" >&2
  exit 2
fi
if [[ -n "$INIT_FROM" && -n "$RESUME_FROM" ]]; then
  echo "--init-from and --resume-from are mutually exclusive" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$(realpath "$MANIFEST")"
SHARD_DIR="$(dirname "$MANIFEST")"
OUTPUT="$(realpath -m "$OUTPUT")"

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

ecloudflow doctor --server --dataset "$MANIFEST" --output-dir "$OUTPUT/doctor"

cmd=(
  ecloudflow train
  +experiment=pdbbind_large
  "train=stage${STAGE}"
  --output-dir "$OUTPUT"
  --override "data.manifest=$MANIFEST"
  --override "data.shard_dir=$SHARD_DIR"
)
if [[ -n "$INIT_FROM" ]]; then
  cmd+=(--init-from "$(realpath "$INIT_FROM")")
fi
if [[ -n "$RESUME_FROM" ]]; then
  cmd+=(--resume-from "$(realpath "$RESUME_FROM")")
fi
if [[ -n "$MAX_STEPS" ]]; then
  cmd+=(--max-steps "$MAX_STEPS")
fi

"${cmd[@]}"
