#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-ecloudflow-h100}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda env update --name "$ENV_NAME" --file environment-h100.yml --prune
else
  conda env create --name "$ENV_NAME" --file environment-h100.yml
fi

conda run --name "$ENV_NAME" \
  python -m pip install --requirement requirements/h100-cu128.txt
conda run --name "$ENV_NAME" \
  python -m pip install --no-deps --editable .
conda run --name "$ENV_NAME" python -m pip check
conda run --name "$ENV_NAME" ecloudflow doctor --server

echo "Environment '$ENV_NAME' is ready. Activate it with: conda activate $ENV_NAME"
