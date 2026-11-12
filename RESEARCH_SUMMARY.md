# Research Summary: Adding Models to LLM Benchmarks Database

**Date**: September 30, 2025  
**Topic**: How models are added to the database & adding Claude Sonnet 4.5

---

## Executive Summary

✅ **Yes, this is fixable via code!** You can add new models to the database using:
1. MongoDB scripts (recommended)
2. Python script (created for you)
3. Direct MongoDB commands

You may need to be on your server locally only to ensure you have access to the MongoDB instance and can run the seeding scripts.

---

## How the System Works

### Database Architecture

The LLM Benchmarks system uses **MongoDB** to manage model configurations:

- **Database**: `MONGODB_DB` (default: `llm-bench`)
- **Collection**: `MONGODB_COLLECTION_MODELS` (default: `models`)
- **Document Structure**:
  ```javascript
  {
    provider: "anthropic",
    model_id: "claude-3-5-sonnet-20240620",
    enabled: true,
    added_at: ISODate("2024-06-20T...")
  }
  ```

### How Models Are Loaded

The Python code (`/workspace/api/llm_bench/models_db.py`) queries MongoDB for enabled models:

```python
def load_provider_models() -> Dict[str, List[str]]:
    # Connects to MongoDB
    # Queries: db.models.find({"enabled": True})
    # Returns: {provider: [model_ids]}
```

This means:
- **Runtime**: Models are loaded from MongoDB
- **Configuration**: `/workspace/cloud/models.json` is a reference file (not used by runtime)
- **Control**: Enable/disable models by changing the `enabled` field in MongoDB

---

## Current Anthropic Models in System

From `/workspace/cloud/models.json`:

```json
"anthropic": [
    "claude-2.1",
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
    "claude-3-sonnet-20240229",
    "claude-3-5-sonnet-20240620",
    "claude-3-7-sonnet-20250219",  // This is the newest in your system
    "claude-3-opus-20240229"
]
```

**Note**: Claude 3.7 Sonnet (released Feb 2025) is already in the system.

---

## About Claude Sonnet 4.5

### Release Information
- **Announced**: September 29, 2025
- **Type**: Cloud API only (no local deployment)
- **Improvements**: Coding, finance, cybersecurity, long-duration autonomous work

### Model Naming Pattern

Anthropic uses date-based model identifiers following the pattern:
```
claude-{family}-{variant}-{YYYYMMDD}
```

Examples:
- `claude-3-5-sonnet-20240620` (June 20, 2024)
- `claude-3-7-sonnet-20250219` (Feb 19, 2025)
- `claude-3-5-haiku-20241022` (Oct 22, 2024)

### Likely Model ID for Sonnet 4.5

⚠️ **UNCONFIRMED** - Based on naming patterns, likely one of:
- `claude-4-sonnet-20250929` (most likely)
- `claude-4-5-sonnet-20250929`
- `claude-sonnet-4-5-20250929`

**YOU MUST VERIFY** the exact model ID from:
- Anthropic's API documentation: https://docs.anthropic.com/en/docs/models-overview
- Your Anthropic console/dashboard
- Or test via their API

---

## How to Add Claude Sonnet 4.5

### Prerequisites

1. **Access to MongoDB**:
   ```bash
   export MONGODB_URI="mongodb+srv://user:pass@cluster.mongodb.net"
   export MONGODB_DB="llm-bench"  # or llmbench_staging
   ```

2. **Tools installed**:
   - Either `mongosh` (for shell script)
   - Or `pymongo` (for Python script)

3. **Anthropic API Key**:
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

4. **Model ID**: Get the exact model identifier from Anthropic

---

### Method 1: Using the Python Script (Recommended)

I created a helper script for you at `/workspace/scripts/add_model_to_db.py`:

```bash
# First, verify the model ID from Anthropic's docs
# Then run:

python scripts/add_model_to_db.py \
  --provider anthropic \
  --model-id claude-4-sonnet-20250929

# To list existing models:
python scripts/add_model_to_db.py --list --provider anthropic
```

**Features**:
- ✅ Checks for existing models
- ✅ Confirms before updating
- ✅ Verifies the insertion
- ✅ Shows next steps
- ✅ Can list all models

---

### Method 2: Using the Shell Script

I created a bash script at `/workspace/scripts/add_sonnet_4.5.sh`:

```bash
# Edit the script to set the correct MODEL_ID
# Then run:

bash scripts/add_sonnet_4.5.sh
```

---

### Method 3: Using the Original MongoDB Script

The existing seed script:

```bash
PROVIDER=anthropic \
MODEL_ID=claude-4-sonnet-20250929 \
mongosh "$MONGODB_URI/$MONGODB_DB" scripts/seed_model.js
```

---

### Method 4: Direct MongoDB Command

```bash
mongosh "$MONGODB_URI/$MONGODB_DB" --eval '
db.models.insertOne({
  provider: "anthropic",
  model_id: "claude-4-sonnet-20250929",
  enabled: true,
  added_at: new Date()
})
'
```

---

## Testing the Integration

### 1. Verify the model was added:

```bash
mongosh "$MONGODB_URI/$MONGODB_DB" --eval '
db.models.find({
  provider: "anthropic",
  model_id: "claude-4-sonnet-20250929"
}).pretty()
'
```

### 2. Run a test benchmark:

```bash
python api/bench_headless.py \
  --providers anthropic \
  --limit 1 \
  --fresh-minutes 0
```

### 3. Check for errors:

```bash
mongosh "$MONGODB_URI/$MONGODB_DB" --eval '
db.errors_cloud.find({
  provider: "anthropic",
  model_name: /claude-4/
}).sort({ts: -1}).limit(5).pretty()
'
```

### 4. View results:

```bash
mongosh "$MONGODB_URI/$MONGODB_DB" --eval '
db.metrics_cloud_staging.find({
  provider: "anthropic",
  model_name: "claude-4-sonnet-20250929"
}).sort({gen_ts: -1}).limit(1).pretty()
'
```

---

## Files I Created for You

1. **`/workspace/ADD_SONNET_4.5_INSTRUCTIONS.md`**
   - Complete step-by-step guide
   - Troubleshooting tips
   - Environment setup

2. **`/workspace/scripts/add_model_to_db.py`**
   - Python script to add models
   - List existing models
   - Interactive confirmation
   - Error handling

3. **`/workspace/scripts/add_sonnet_4.5.sh`**
   - Bash script specifically for Sonnet 4.5
   - Validation checks
   - Verification steps

4. **`/workspace/RESEARCH_SUMMARY.md`** (this file)
   - Complete research findings
   - All methods to add models

---

## Key Files in the Codebase

### Runtime Code
- `/workspace/api/llm_bench/models_db.py` - Loads models from MongoDB
- `/workspace/api/llm_bench/cloud/providers/anthropic.py` - Anthropic API integration
- `/workspace/api/bench_headless.py` - Runs benchmarks

### Configuration
- `/workspace/cloud/models.json` - Reference only (not used at runtime)
- Environment variables - The actual configuration source

### Scripts
- `/workspace/scripts/seed_model.js` - MongoDB script to add single model
- `/workspace/scripts/add_model_to_db.py` - **NEW**: Python helper script
- `/workspace/scripts/add_sonnet_4.5.sh` - **NEW**: Bash helper script

---

## Important Notes

### Do You Need Local Server Access?

**Maybe** - You need access to:
1. ✅ **MongoDB instance**: To add the model to the database
2. ✅ **Environment variables**: To set `MONGODB_URI`, `ANTHROPIC_API_KEY`, etc.
3. ❌ **Not needed**: Local model files (Anthropic is cloud API only)
4. ❌ **Not needed**: Special hardware/GPU (for API-based models)

**If you can**:
- Connect to the MongoDB instance from anywhere → Can add the model remotely
- Set environment variables on the server → Can run benchmarks

### Cloud vs Local Benchmarks

This system has two types:
- **Cloud benchmarks**: Test API-based models (OpenAI, Anthropic, etc.) - This is what you want
- **Local benchmarks**: Test locally hosted models (Llama, etc.) - Not relevant here

Anthropic models are **cloud benchmarks** - they just need API access.

---

## Next Steps

1. **Get the exact model ID**:
   - Check https://docs.anthropic.com/en/docs/models-overview
   - Or use Anthropic's API to list available models
   - The date suffix is usually the release date (YYYYMMDD)

2. **Ensure you have MongoDB access**:
   ```bash
   mongosh "$MONGODB_URI/$MONGODB_DB" --eval 'db.models.countDocuments()'
   ```

3. **Add the model** using one of the methods above

4. **Test it** with a single benchmark run

5. **Monitor** the first few runs for any issues

---

## Troubleshooting

### "Model not found" errors
- Check the model_id exactly matches what Anthropic expects
- Verify with a direct API call first

### "Authentication failed"
- Check `ANTHROPIC_API_KEY` is set correctly
- Verify the key has access to the new model

### Model not appearing in benchmarks
- Check `enabled: true` in database
- Verify `fresh-minutes` isn't skipping it
- Check the scheduler picked it up

### API rate limits
- Anthropic has rate limits per model
- May need to adjust `SLEEP_SECONDS` between runs

---

## Questions to Answer

Before proceeding, you should:

1. ✅ **Verify the model ID**: What is the exact identifier from Anthropic?
2. ✅ **Check API access**: Can you call this model via Anthropic's API?
3. ✅ **Confirm MongoDB access**: Can you connect and write to the database?
4. ✅ **Test environment**: Are all environment variables set correctly?

---

## Summary

**The good news**: This is 100% fixable via code. The system is designed to support adding new models easily through the database.

**What you need**:
1. The correct model ID from Anthropic
2. Access to your MongoDB instance
3. Your Anthropic API key

**What you do**:
1. Run one of the scripts I created
2. Test with a benchmark
3. Monitor for any issues

**No code changes needed** - everything is configuration-driven through MongoDB!