# Adding New Models to LLM Benchmarks

**Quick Navigation**: 🚀 [Quick Start](#quick-start) | 📚 [Documentation](#documentation) | 🔧 [Tools](#tools) | ❓ [FAQ](#faq)

---

## 🎯 Goal

Add **Claude Sonnet 4.5** (or any new model) to your LLM Benchmarks database.

## ✅ Answer: Is This Fixable Via Code?

**YES!** The system is designed for this. You can add models through:
- MongoDB scripts
- Python scripts (recommended)
- Direct database commands

**No code changes needed** - it's all configuration-driven.

---

## 🚀 Quick Start

### The Fastest Way (3 commands)

```bash
# 1. Find the correct model ID
python scripts/test_anthropic_models.py --test-model claude-4-sonnet-20250929

# 2. Add it to the database
python scripts/add_model_to_db.py --provider anthropic --model-id claude-4-sonnet-20250929

# 3. Test it
python api/bench_headless.py --providers anthropic --limit 1
```

**Done!** 🎉

For more details, see [QUICK_START.md](QUICK_START.md)

---

## 📚 Documentation

I've created comprehensive documentation for you:

| File | Purpose | When to Use |
|------|---------|-------------|
| **[QUICK_START.md](QUICK_START.md)** | 30-second guide | You want to add the model NOW |
| **[RESEARCH_SUMMARY.md](RESEARCH_SUMMARY.md)** | Complete research & all methods | You want to understand everything |
| **[ADD_SONNET_4.5_INSTRUCTIONS.md](ADD_SONNET_4.5_INSTRUCTIONS.md)** | Step-by-step guide | You want detailed instructions |
| **[FILES_CREATED.md](FILES_CREATED.md)** | Index of new files | You want to see what was created |

---

## 🔧 Tools

I've created three utility scripts:

### 1. Test Anthropic Models ⭐ START HERE

**Purpose**: Find the correct model ID and verify API access

```bash
python scripts/test_anthropic_models.py --test-model claude-4-sonnet-20250929
```

**This answers**: "What's the correct model ID?"

### 2. Add Model to Database ⭐ RECOMMENDED

**Purpose**: Add any model to MongoDB

```bash
python scripts/add_model_to_db.py --provider anthropic --model-id <MODEL_ID>
```

**Features**: Duplicate checking, verification, list models, interactive

### 3. Add Sonnet 4.5 (Bash)

**Purpose**: Bash alternative for adding Sonnet 4.5

```bash
bash scripts/add_sonnet_4.5.sh
```

**Note**: Edit the script first to set the correct model ID

---

## 🔍 Research Findings

### How Models Are Stored

- **Database**: MongoDB
- **Collection**: `models` (configurable)
- **Structure**:
  ```javascript
  {
    provider: "anthropic",
    model_id: "claude-4-sonnet-20250929",
    enabled: true,
    added_at: ISODate(...)
  }
  ```

### How Models Are Loaded

The Python code (`api/llm_bench/models_db.py`) queries MongoDB:

```python
db.models.find({"enabled": True})
```

This means:
- ✅ Add models = add to MongoDB
- ✅ Enable/disable = update `enabled` field
- ✅ Changes are immediate (no restart needed)

### Current Anthropic Models in Your System

From your database/config:
- `claude-3-7-sonnet-20250219` ← **Newest currently**
- `claude-3-5-sonnet-20240620`
- `claude-3-5-haiku-20241022`
- `claude-3-opus-20240229`
- Others...

---

## 🎯 About Claude Sonnet 4.5

### Release Info
- **Date**: September 29, 2025
- **Type**: Cloud API only
- **Capabilities**: Improved coding, finance, cybersecurity, long-duration tasks

### Model Naming Pattern

Anthropic uses: `claude-{family}-{variant}-{YYYYMMDD}`

Examples:
- `claude-3-5-sonnet-20240620` (June 20, 2024)
- `claude-3-7-sonnet-20250219` (Feb 19, 2025)

### Likely Model ID

⚠️ **Must verify from Anthropic docs!**

Based on pattern, likely:
- `claude-4-sonnet-20250929` (most likely)
- `claude-4-5-sonnet-20250929`
- `claude-sonnet-4-5-20250929`

**Verify at**: https://docs.anthropic.com/en/docs/models-overview

---

## ❓ FAQ

### Q: Do I need to be on my server locally?

**A**: Only if:
- Your MongoDB is local-only (no remote access)
- You need to set environment variables on the server

If MongoDB is accessible remotely, you can add models from anywhere.

### Q: Will this require code changes?

**A**: No! The system is configuration-driven through MongoDB.

### Q: What if I use the wrong model ID?

**A**: The benchmark will fail with an API error. Just update it in MongoDB:

```bash
python scripts/add_model_to_db.py --provider anthropic --model-id CORRECT_ID
```

### Q: How do I know if it worked?

**A**: Run a test benchmark:

```bash
python api/bench_headless.py --providers anthropic --limit 1
```

Check for errors in the `errors_cloud` collection.

### Q: Can I add multiple models at once?

**A**: Yes! Run the script multiple times:

```bash
python scripts/add_model_to_db.py --provider anthropic --model-id model-1
python scripts/add_model_to_db.py --provider anthropic --model-id model-2
```

### Q: How do I list what's already in the database?

**A**: 

```bash
python scripts/add_model_to_db.py --list --provider anthropic
```

---

## 📋 Prerequisites

Before you start, ensure you have:

### Environment Variables
```bash
export MONGODB_URI="mongodb+srv://..."
export MONGODB_DB="llm-bench"
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Tools Installed
- Python 3.9+
- `pymongo` package: `pip install pymongo`
- `anthropic` package: `pip install anthropic`
- OR `mongosh` for bash scripts

### Access
- MongoDB read/write access
- Valid Anthropic API key
- Correct model ID from Anthropic docs

---

## 🎓 Step-by-Step Tutorial

### Step 1: Verify Environment

```bash
# Check MongoDB connection
mongosh "$MONGODB_URI/$MONGODB_DB" --eval 'db.runCommand({ping: 1})'

# Check Anthropic API key
echo $ANTHROPIC_API_KEY
```

### Step 2: Find the Correct Model ID

```bash
# Test a likely model ID
python scripts/test_anthropic_models.py --test-model claude-4-sonnet-20250929

# If it fails, try variations or check Anthropic docs
```

### Step 3: Add to Database

```bash
# Add the model
python scripts/add_model_to_db.py \
  --provider anthropic \
  --model-id claude-4-sonnet-20250929
```

### Step 4: Verify

```bash
# List all Anthropic models
python scripts/add_model_to_db.py --list --provider anthropic
```

### Step 5: Test

```bash
# Run a single benchmark
python api/bench_headless.py --providers anthropic --limit 1 --fresh-minutes 0
```

### Step 6: Check Results

```bash
# Check for errors
mongosh "$MONGODB_URI/$MONGODB_DB" --eval '
db.errors_cloud.find({provider: "anthropic"}).sort({ts: -1}).limit(5).pretty()
'

# Check for successful results
mongosh "$MONGODB_URI/$MONGODB_DB" --eval '
db.metrics_cloud_staging.find({
  provider: "anthropic",
  model_name: /claude-4/
}).sort({gen_ts: -1}).limit(1).pretty()
'
```

---

## 🔗 Key Files in Codebase

### Runtime
- `api/llm_bench/models_db.py` - Loads models from MongoDB
- `api/llm_bench/cloud/providers/anthropic.py` - Anthropic integration
- `api/bench_headless.py` - Benchmark runner

### Configuration
- `cloud/models.json` - Reference file (NOT used at runtime)
- Environment variables - Actual configuration

### Scripts (Original)
- `scripts/seed_model.js` - MongoDB script to add models
- `scripts/enqueue_job.js` - Queue a benchmark job

### Scripts (NEW - Created for You)
- `scripts/add_model_to_db.py` ⭐ - Python helper
- `scripts/test_anthropic_models.py` ⭐ - Test API & find model IDs
- `scripts/add_sonnet_4.5.sh` - Bash helper

---

## 🎉 Summary

### What You Learned

1. ✅ Models are stored in **MongoDB** (not code)
2. ✅ Adding models = **updating the database**
3. ✅ The system is **configuration-driven**
4. ✅ Changes are **immediate** (no restart needed)

### What You Got

1. 📚 **4 documentation files** explaining everything
2. 🔧 **3 utility scripts** to make it easy
3. 🎯 **Step-by-step instructions** for success

### What You Need to Do

1. 🔍 **Find the correct model ID** (use `test_anthropic_models.py`)
2. ➕ **Add it to MongoDB** (use `add_model_to_db.py`)
3. ✅ **Test it** (run a benchmark)
4. 🎊 **Done!**

---

## 🚀 Ready to Start?

Pick your starting point:

- **Just want to do it**: [QUICK_START.md](QUICK_START.md)
- **Want to understand first**: [RESEARCH_SUMMARY.md](RESEARCH_SUMMARY.md)
- **Want step-by-step guide**: [ADD_SONNET_4.5_INSTRUCTIONS.md](ADD_SONNET_4.5_INSTRUCTIONS.md)
- **Want to see what's available**: [FILES_CREATED.md](FILES_CREATED.md)

Or just run:
```bash
python scripts/test_anthropic_models.py
```

Good luck! 🎯