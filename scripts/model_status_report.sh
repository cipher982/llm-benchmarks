#!/bin/bash
# Thin wrapper around the lifecycle CLI.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

cd "$REPO_ROOT"

PYTHONPATH=api uv run python -m api.llm_bench.model_lifecycle.cli "$@"
