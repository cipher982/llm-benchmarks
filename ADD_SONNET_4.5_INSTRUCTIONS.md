# Adding Claude Sonnet 4.5 to LLM Benchmarks

## Overview
This document explains how to add the new Claude Sonnet 4.5 model to your LLM benchmarking system.

## Current Status
- **Release Date**: September 29, 2025
- **Model Type**: Cloud API-based (not for local deployment)
- **Provider**: Anthropic

## Model Identifier
⚠️ **IMPORTANT**: You need to verify the exact model ID from Anthropic's documentation.

Based on existing naming patterns, it's likely one of:
- `claude-4-sonnet-20250929`
- `claude-4-5-sonnet-20250929`
- `claude-4-sonnet-20250514`

Check: https://docs.anthropic.com/en/docs/models-overview

## How to Add the Model

### Prerequisites
- Access to your MongoDB instance
- `mongosh` installed on your machine
- Environment variables set:
  - `MONGODB_URI`: e.g., `mongodb+srv://user:pass@cluster.mongodb.net`
  - `MONGODB_DB`: e.g., `llm-bench` or `llmbench_staging`

### Method 1: Add to Database (Recommended for Production)

1. **Verify the correct model ID** from Anthropic's API documentation

2. **Seed the model into MongoDB**:
   ```bash
   # Replace MODEL_ID with the actual model identifier
   PROVIDER=anthropic MODEL_ID=claude-4-sonnet-20250929 mongosh "$MONGODB_URI/$MONGODB_DB" scripts/seed_model.js
   ```

3. **Verify it was added**:
   ```bash
   mongosh "$MONGODB_URI/$MONGODB_DB" --eval 'db.models.find({provider: "anthropic", enabled: true}).pretty()'
   ```

### Method 2: Update Configuration File

Edit `/workspace/cloud/models.json` and add the new model to the `anthropic` array:

```json
"anthropic": [
    "claude-2.1",
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
    "claude-3-sonnet-20240229",
    "claude-3-5-sonnet-20240620",
    "claude-3-7-sonnet-20250219",
    "claude-3-opus-20240229",
    "claude-4-sonnet-20250929"  // <-- ADD THIS (verify exact ID)
]
```

**Note**: This method updates the reference configuration. The actual benchmarking system reads from MongoDB.

### Method 3: Enqueue a Job (Optional)

If you want to immediately queue a benchmark job:
```bash
PROVIDER=anthropic MODEL=claude-4-sonnet-20250929 IGNORE_FRESHNESS=true mongosh "$MONGODB_URI/$MONGODB_DB" scripts/enqueue_job.js
```

## Testing the Integration

### 1. Test with headless benchmark (local, no Docker):
```bash
python api/bench_headless.py --providers anthropic --limit 1 --fresh-minutes 0
```

### 2. Check for errors:
```bash
mongosh "$MONGODB_URI/$MONGODB_DB" --eval 'db.errors_cloud.find({provider: "anthropic"}).sort({ts: -1}).limit(5).pretty()'
```

### 3. View results:
```bash
mongosh "$MONGODB_URI/$MONGODB_DB" --eval 'db.metrics_cloud_staging.find({provider: "anthropic", model_name: "claude-4-sonnet-20250929"}).sort({gen_ts: -1}).limit(1).pretty()'
```

## Database Schema

Models are stored with the following structure:
```javascript
{
  provider: "anthropic",
  model_id: "claude-4-sonnet-20250929",
  enabled: true,
  added_at: ISODate("2025-09-30T...")
}
```

## Code Files Involved

1. **`/workspace/api/llm_bench/models_db.py`**: Loads models from MongoDB
2. **`/workspace/cloud/models.json`**: Reference configuration
3. **`/workspace/scripts/seed_model.js`**: Script to add models to DB
4. **`/workspace/api/llm_bench/cloud/providers/anthropic.py`**: Anthropic API integration

## Required Environment Variables

Ensure you have:
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `MONGODB_URI`: MongoDB connection string
- `MONGODB_DB`: Database name
- `MONGODB_COLLECTION_MODELS`: Collection name (default: `models`)

## Next Steps

1. ✅ Find the exact model ID from Anthropic docs
2. ✅ Ensure `ANTHROPIC_API_KEY` is set in your environment
3. ✅ Run the seed_model.js script with the correct model ID
4. ✅ Test with a benchmark run
5. ✅ Monitor for any errors

## Troubleshooting

**If the model doesn't appear in benchmarks:**
- Check it's marked as `enabled: true` in the database
- Verify the model_id matches exactly what Anthropic's API expects
- Check logs for API errors (might indicate wrong model ID)

**If you get authentication errors:**
- Verify `ANTHROPIC_API_KEY` is set correctly
- Check the API key has access to the new model

**If the model runs but fails:**
- The model ID might be incorrect
- Check Anthropic's API status/changelog for any breaking changes