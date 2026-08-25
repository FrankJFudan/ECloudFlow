#!/usr/bin/env bash
set -euo pipefail

CONFIG="experiment=h100_smoke"
OUTPUT="runs/h100-smoke"
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
# Set ECLOUDFLOW_RUN_NCCL=1 to execute the real four-rank NCCL path.
if [[ "${ECLOUDFLOW_RUN_NCCL:-0}" == "1" ]]; then
  # The explicit flag is required because NCCL is a server-only validation.
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
    "$OUTPUT"
    --devices
    4
  )
else
  # Local smoke uses deterministic synthetic work and never claims GPU results.
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
    --dry-run
  )
fi
if [[ -n "$STEPS" ]]; then
  cmd+=(--steps "$STEPS")
fi
"${cmd[@]}"
popd >/dev/null
