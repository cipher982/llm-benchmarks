#!/bin/bash

# Script to add Claude Sonnet 4.5 to the LLM Benchmarks database
# 
# Prerequisites:
# - mongosh installed
# - MONGODB_URI and MONGODB_DB environment variables set
#
# Usage:
#   1. First, find the exact model ID from Anthropic's documentation
#   2. Update the MODEL_ID variable below with the correct value
#   3. Ensure MONGODB_URI and MONGODB_DB are set in your environment
#   4. Run this script: bash scripts/add_sonnet_4.5.sh

set -e  # Exit on error

# ====================================
# CONFIGURATION - UPDATE THESE VALUES
# ====================================

# TODO: Replace with the actual model ID from Anthropic's docs
# Common possibilities based on their naming pattern:
#   - claude-4-sonnet-20250929
#   - claude-4-5-sonnet-20250929
#   - claude-sonnet-4-5-20250929
MODEL_ID="claude-4-sonnet-20250929"  # <-- VERIFY AND UPDATE THIS

PROVIDER="anthropic"
ENABLED="true"

# ====================================
# VALIDATION
# ====================================

echo "=========================================="
echo "Add Claude Sonnet 4.5 to LLM Benchmarks"
echo "=========================================="
echo ""

# Check if mongosh is installed
if ! command -v mongosh &> /dev/null; then
    echo "❌ Error: mongosh is not installed"
    echo "   Install it from: https://www.mongodb.com/docs/mongodb-shell/install/"
    exit 1
fi

# Check if required environment variables are set
if [ -z "$MONGODB_URI" ]; then
    echo "❌ Error: MONGODB_URI environment variable is not set"
    echo "   Example: export MONGODB_URI='mongodb+srv://user:pass@cluster.mongodb.net'"
    exit 1
fi

if [ -z "$MONGODB_DB" ]; then
    echo "⚠️  Warning: MONGODB_DB not set, using default: llm-bench"
    MONGODB_DB="llm-bench"
fi

echo "Configuration:"
echo "  Provider:  $PROVIDER"
echo "  Model ID:  $MODEL_ID"
echo "  Enabled:   $ENABLED"
echo "  Database:  $MONGODB_DB"
echo ""

# Warn about model ID
if [[ "$MODEL_ID" == "claude-4-sonnet-20250929" ]]; then
    echo "⚠️  WARNING: You are using the default model ID"
    echo "   Please verify this is correct from Anthropic's documentation:"
    echo "   https://docs.anthropic.com/en/docs/models-overview"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# ====================================
# ADD MODEL TO DATABASE
# ====================================

echo ""
echo "Adding model to database..."

PROVIDER=$PROVIDER MODEL_ID=$MODEL_ID ENABLED=$ENABLED \
    mongosh "$MONGODB_URI/$MONGODB_DB" scripts/seed_model.js

if [ $? -eq 0 ]; then
    echo "✅ Model added successfully!"
else
    echo "❌ Failed to add model"
    exit 1
fi

# ====================================
# VERIFY
# ====================================

echo ""
echo "Verifying model was added..."

QUERY="db.models.find({provider: '$PROVIDER', model_id: '$MODEL_ID'}).pretty()"
mongosh "$MONGODB_URI/$MONGODB_DB" --eval "$QUERY"

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Ensure ANTHROPIC_API_KEY is set in your environment"
echo "2. Test with: python api/bench_headless.py --providers anthropic --limit 1"
echo "3. Check for errors in the errors_cloud collection"
echo ""
echo "To list all enabled Anthropic models:"
echo "  mongosh \"\$MONGODB_URI/\$MONGODB_DB\" --eval 'db.models.find({provider: \"anthropic\", enabled: true}).pretty()'"
echo ""