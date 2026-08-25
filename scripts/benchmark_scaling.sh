#!/usr/bin/env bash
set -euo pipefail

CONFIG="experiment=h100_large"
OUTPUT="runs/scaling"
STEPS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

pushd "$REPO_ROOT" >/dev/null
export ECLOUDFLOW_BENCHMARK_CONFIG="$CONFIG"
"$PYTHON_BIN" - <<'PY'
import os

from ecloudflow.config import load_config
from ecloudflow.training.benchmark import benchmark_hashes

config = os.environ["ECLOUDFLOW_BENCHMARK_CONFIG"]
resolved = load_config([config if config.startswith(("+", "~")) else f"+{config}"])
hashes = benchmark_hashes(
    config, resolved.benchmark, resolved.data, app_config=resolved
)
print(f"git={hashes['git']}")
print(f"config={hashes['config']}")
print(f"data={hashes['data']}")
print(f"data_source={hashes['data_source']}")
PY
# Set ECLOUDFLOW_RUN_NCCL=1 to execute the real multi-process NCCL path.
if [[ "${ECLOUDFLOW_RUN_NCCL:-0}" == "1" ]]; then
  for count in 1 2 4; do
    run_dir="$OUTPUT/dev${count}"
    if [[ "$count" -eq 1 ]]; then
      cmd=(
        torchrun
        --standalone
        --nproc_per_node=1
        -m
        ecloudflow.cli.main
        benchmark
        --config
        "$CONFIG"
        --output-dir
        "$run_dir"
        --devices
        1
      )
    elif [[ "$count" -eq 2 ]]; then
      cmd=(
        torchrun
        --standalone
        --nproc_per_node=2
        -m
        ecloudflow.cli.main
        benchmark
        --config
        "$CONFIG"
        --output-dir
        "$run_dir"
        --devices
        2
      )
    else
      cmd=(
        torchrun
        --standalone
        --nproc_per_node=4
        -m
        ecloudflow.cli.main
        benchmark
        --config
        "$CONFIG"
        --output-dir
        "$run_dir"
        --devices
        4
      )
    fi
    if [[ -n "$STEPS" ]]; then
      cmd+=(--steps "$STEPS")
    fi
    "${cmd[@]}"
  done
  "$PYTHON_BIN" - "$OUTPUT" <<'PY'
import sys
from pathlib import Path

from ecloudflow.training.benchmark import merge_scaling_reports

root = Path(sys.argv[1])
sources = [root / f"dev{count}" / "scaling.json" for count in (1, 2, 4)]
paths = merge_scaling_reports(sources, root)
for path in paths:
    print(path)
PY
else
  cmd=(
    "$PYTHON_BIN"
    -m
    ecloudflow.cli.main
    benchmark
    --config
    "$CONFIG"
    --output-dir
    "$OUTPUT"
    --devices
    1
    --devices
    2
    --devices
    4
    --dry-run
  )
  if [[ -n "$STEPS" ]]; then
    cmd+=(--steps "$STEPS")
  fi
  "${cmd[@]}"
fi
popd >/dev/null
