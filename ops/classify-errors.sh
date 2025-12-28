#!/bin/bash
# Quick script to run LLM error classification
# Usage:
#   ./classify-errors.sh              # Classify up to 200 errors
#   ./classify-errors.sh --max 500    # Classify up to 500 errors
#   ./classify-errors.sh --all        # Classify all unclassified errors

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check API keys
if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: Need ANTHROPIC_API_KEY or OPENAI_API_KEY"
    exit 1
fi

# Parse arguments
MAX_ROLLUPS=""
BATCH_SIZE=50
USE_OPENAI=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            MAX_ROLLUPS=""
            shift
            ;;
        --max)
            MAX_ROLLUPS="--max-rollups $2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE=$2
            shift 2
            ;;
        --use-openai)
            USE_OPENAI="--use-openai"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--all] [--max N] [--batch-size N] [--use-openai]"
            exit 1
            ;;
    esac
done

# Default to 200 if not specified
if [ -z "$MAX_ROLLUPS" ]; then
    MAX_ROLLUPS="--max-rollups 200"
fi

echo "Running LLM error classification..."
uv run env PYTHONPATH=api python -m llm_bench.ops.llm_error_classifier \
    --batch-size "$BATCH_SIZE" \
    $MAX_ROLLUPS \
    $USE_OPENAI
