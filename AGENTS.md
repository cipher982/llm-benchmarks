# LLM Benchmarks API Service

This is the backend code that has a Docker container with a scheduler and makes calls to benchmark providers across cloud and local services. Cloud providers represent ~90% of the effort and value; local model benchmarking has been deprioritized.

## Quick Reference

| Need | Command |
|------|---------|
| **Check errors** | `mongosh "$MONGODB_URI" --eval "db.errors_cloud.find({ts: {\$gte: new Date(Date.now()-24*60*60*1000)}}).limit(10)"` |
| **Deploy clifford** | Via Coolify API (see Coolify section) |
| **Add model** | `mongosh "$MONGODB_URI" --eval "db.models.insertOne({provider: 'openai', model_id: 'gpt-4o', enabled: true, created_at: new Date()})"` |
| **Disable model** | `mongosh "$MONGODB_URI" --eval "db.models.updateOne({provider: 'X', model_id: 'Y'}, {\$set: {enabled: false, disabled_reason: 'Reason', disabled_at: new Date()}})"` |
| **Check logs** | `ssh clifford 'docker logs $(docker ps -qf "name=llm-bench-service") --tail 100'` |

---

## Repository Structure

### Nested Repository Structure

**Note:** `~/git/llmbench/` is NOT a git repository itself. It contains two separate repos:
- `~/git/llmbench/llm-benchmarks/` - Benchmark runner (this repo)
- `~/git/llmbench/llm-benchmarks-dashboard/` - Dashboard UI (Next.js)

Always `cd` into the specific subdirectory before git operations.

### Key Files

| File | Purpose |
|------|---------|
| `api/bench_headless.py` | Main benchmark runner (daemon mode) |
| `api/bench_simple_runner.py` | Lightweight runner for EC2 (HTTP ingest) |
| `api/llm_bench/cloud/providers/` | Provider implementations |
| `api/llm_bench/models_db.py` | Model loading from MongoDB |
| `REASONING_MODELS.md` | OpenAI o1/o3/o4 support documentation |

---

## Architecture

### Two Deployment Instances

| Instance | Location | Providers | Connection |
|----------|----------|-----------|------------|
| **clifford** | VPS | anthropic, cerebras, deepinfra, fireworks, groq, openai, together, vertex | Direct MongoDB |
| **ml-tuner-demo-server** | AWS EC2 | bedrock only | HTTP Ingest Bridge |

**Why separate?**
- Bedrock requires AWS IAM role (EC2 instance profile)
- MongoDB only accessible via Tailscale (not from AWS)
- HTTP ingest bridge allows EC2 to POST results to clifford

### HTTP Ingest Bridge

1. **bench-ingest API**: Deployed on `clifford` at `https://bench-ingest.drose.io`
   - Receives benchmark results via HTTPS
   - Writes to MongoDB
   - FastAPI service

2. **bench_simple_runner.py**: Lightweight runner on EC2
   - Reuses provider logic
   - POSTs results instead of direct MongoDB writes

### Runner Configuration

**clifford** (`bench_headless.py`):
```bash
# Environment variables
BENCHMARK_PROVIDERS=anthropic,cerebras,deepinfra,fireworks,groq,openai,together,vertex
FRESH_MINUTES=180  # Run every 3 hours
MONGODB_URI=mongodb://...
```

**EC2** (`bench_simple_runner.py`):
```bash
# Environment variables
INGEST_API_URL=https://bench-ingest.drose.io/ingest
INGEST_API_KEY=xxx
BENCHMARK_MODELS=us.anthropic.claude-3-5-sonnet-...,us.meta.llama...
```

---

## MongoDB Collections

| Collection | Purpose | Key Fields |
|------------|---------|------------|
| `models` | Enabled models catalog | provider, model_id, enabled, deprecated |
| `metrics_cloud_v2` | Successful benchmark runs | provider, model_name, run_ts, tokens_per_second |
| `errors_cloud` | Failed runs | provider, model_name, ts, message, stage |
| `provider_catalog` | Models discovered from provider APIs | provider, model_id, first_seen_at, in_our_db |
| `openrouter_catalog` | OpenRouter model listings | (deprecated in favor of provider_catalog) |
| `deprecation_snapshots` | Historical baselines for deprecated models | provider, model_id, stats |
| `model_status` | Lifecycle tracking | provider, model_id, status, last_run_at |

**MongoDB URI:** Check Coolify env vars for `MONGODB_URI` or ask user for connection string.

**Database name:** `llm-bench` (configurable via `MONGODB_DB`)

---

## Adding Models to the Database

Models are stored in the `models` collection. The scheduler reads enabled models from this collection.

### Model Document Schema

```javascript
{
  provider: "openai",            // Provider name
  model_id: "gpt-4o",           // The actual API model ID
  enabled: true,                // Set false to disable without deleting
  deprecated: false,            // For lifecycle tracking
  created_at: ISODate(),        // When added
  // Optional fields for disabled models:
  disabled_reason: "...",       // Why it was disabled
  disabled_at: ISODate()        // When disabled
}
```

### Provider-Specific ID Rules

#### ⚠️ Bedrock (CRITICAL)

AWS Bedrock requires specific model ID formats. Getting this wrong causes silent failures.

| Model Type | Required Format | Example |
|------------|-----------------|---------|
| Anthropic Claude (all) | `us.anthropic.*` | `us.anthropic.claude-opus-4-5-20251101-v1:0` |
| Meta Llama 3.2+ | `us.meta.*` | `us.meta.llama4-maverick-17b-instruct-v1:0` |
| Meta Llama 3.1 and older | `meta.*` | `meta.llama3-1-70b-instruct-v1:0` |
| Mistral | `mistral.*` | `mistral.mistral-large-2402-v1:0` |
| Cohere | `cohere.*` | `cohere.command-r-plus-v1:0` |
| Amazon Nova | `amazon.*` | `amazon.nova-pro-v1:0` |

**Why `us.` prefix?** Newer Anthropic/Meta models require cross-region inference profiles. The `us.` prefix uses these automatically. Non-prefixed IDs fail with "Invocation with on-demand throughput isn't supported".

**DO NOT add:**
- Non-prefixed Anthropic IDs (e.g., `anthropic.claude-opus-4-5-*`) - will fail
- Image models (e.g., `amazon.nova-canvas-v1:0`) - not text, breaks streaming
- Vision models without testing first

#### ⚠️ OpenAI Reasoning Models

OpenAI's reasoning models (o1, o3, o4 series) require the Responses API (`/v1/responses`) instead of Chat Completions.

**Supported models:**
- o1, o1-pro
- o3, o3-mini, o3-pro
- o4, o4-mini

**Implementation:** Automatically detected by prefix. See `REASONING_MODELS.md` for details.

**Key differences:**
- Uses `max_output_tokens` parameter (not `max_tokens`)
- Includes reasoning tokens in `output_tokens` count
- May output no visible text (all tokens used for reasoning)

#### Model Types to Avoid

| Type | Pattern | Why |
|------|---------|-----|
| Embeddings | `*embedding*` | Not chat models |
| TTS/Audio | `tts-*`, `whisper-*`, `*-audio-*` | Not text generation |
| Image | `dall-e*`, `sora*` | Not text generation |
| Moderation | `*moderation*` | Not chat models |
| Guard models | `*-guard-*` | Output 1-2 tokens (unsuitable) |
| Compound/agentic | `*compound*` | Output 1000+ tokens (unsuitable) |

### Adding a New Model

```bash
mongosh "$MONGODB_URI" --eval '
db.models.insertOne({
  provider: "openai",
  model_id: "gpt-4o-2024-11-20",
  enabled: true,
  deprecated: false,
  created_at: new Date()
})
'
```

### Disabling a Model (preferred over deleting)

```bash
mongosh "$MONGODB_URI" --eval '
db.models.updateOne(
  { provider: "openai", model_id: "gpt-4o-2024-05-13" },
  { $set: { enabled: false, disabled_reason: "Superseded by newer version", disabled_at: new Date() } }
)
'
```

### Checking Current Models

```bash
# List enabled models for a provider
mongosh "$MONGODB_URI" --quiet --eval '
db.models.find({provider: "openai", enabled: true}).forEach(d => print(d.model_id))
'

# Count by provider
mongosh "$MONGODB_URI" --quiet --eval '
db.models.aggregate([
  {$match: {enabled: true}},
  {$group: {_id: "$provider", count: {$sum: 1}}},
  {$sort: {count: -1}}
]).forEach(printjson)
'

# Check disabled models with reasons
mongosh "$MONGODB_URI" --quiet --eval '
db.models.find({enabled: false, disabled_reason: {$exists: true}}).forEach(d => {
  print(d.provider + ":" + d.model_id + " → " + d.disabled_reason)
})
'
```

---

## Error Investigation

### Quick Error Check

```bash
# Errors in last 24 hours by provider
ssh clifford 'mongosh "mongodb://writer:PASSWORD@localhost:27017/llm-bench?authSource=llm-bench" --quiet --eval "
var yesterday = new Date(Date.now() - 24*60*60*1000);
db.errors_cloud.aggregate([
  {\$match: {ts: {\$gte: yesterday}}},
  {\$group: {_id: \"\$provider\", count: {\$sum: 1}}},
  {\$sort: {count: -1}}
]).forEach(printjson)
"'
```

### Error Analysis Patterns

```bash
# Errors by model (last 24h)
var yesterday = new Date(Date.now() - 24*60*60*1000);
db.errors_cloud.aggregate([
  {$match: {provider: "openai", ts: {$gte: yesterday}}},
  {$group: {_id: "$model_name", count: {$sum: 1}, sample_msg: {$first: "$message"}}},
  {$sort: {count: -1}}
]).forEach(e => print(e.count + "x " + e._id + ": " + e.sample_msg.substring(0, 100)))

# Success vs error ratio
var weekAgo = new Date(Date.now() - 7*24*60*60*1000);
var model = "gpt-4o";
var successes = db.metrics_cloud_v2.countDocuments({provider: "openai", model_name: model, run_ts: {$gte: weekAgo}});
var errors = db.errors_cloud.countDocuments({provider: "openai", model_name: model, ts: {$gte: weekAgo}});
print(model + ": " + successes + " successes, " + errors + " errors = " + (successes/(successes+errors)*100).toFixed(1) + "% success rate");

# Common error patterns
db.errors_cloud.aggregate([
  {$match: {ts: {$gte: new Date(Date.now() - 24*60*60*1000)}}},
  {$project: {
    provider: 1,
    error_type: {
      $cond: {
        if: {$regexMatch: {input: "$message", regex: /429|RateLimit|quota/i}},
        then: "Rate Limit",
        else: {$cond: {
          if: {$regexMatch: {input: "$message", regex: /output_tokens.*not within/}},
          then: "Token Count Mismatch",
          else: {$cond: {
            if: {$regexMatch: {input: "$message", regex: /404|NotFound/i}},
            then: "Model Not Found",
            else: "Other"
          }}
        }}
      }
    }
  }},
  {$group: {_id: "$error_type", count: {$sum: 1}}},
  {$sort: {count: -1}}
])
```

### Common Error Patterns

| Error Message | Cause | Solution |
|---------------|-------|----------|
| `ProfileNotFound: zh-ml-mlengineer` | AWS_PROFILE env var set on clifford | Remove AWS_PROFILE from clifford's env |
| `output_tokens X not within 10% of requested Y` | Guard/compound/thinking models | Disable unsuitable models |
| `tokens_per_second <= 0` | Empty response or instant completion | Edge case in validation, usually safe to ignore |
| `429 / quota exceeded` | Rate limit hit | Increase quota or reduce frequency |
| `This model is only supported in v1/responses` | Reasoning model using Chat API | Update to use Responses API |
| `Unsupported parameter: 'max_tokens'` | o1/o3/o4 models | Use `max_output_tokens` (handled automatically) |

---

## Provider Discovery System

**Sauron job:** `llm-bench-provider-discovery` (runs 07:00 UTC daily)

**Purpose:** Fetches model lists directly from provider APIs to discover new models.

**Providers:** Groq, Together, Cerebras, OpenAI, Anthropic, Fireworks, DeepInfra

**Output:**
- Stores in `provider_catalog` collection
- Emails report with add commands for new models
- Filters out non-chat models (embeddings, TTS, image, etc.)

**Checking catalog:**
```bash
ssh clifford 'mongosh "mongodb://..." --quiet --eval "
db.provider_catalog.aggregate([
  {\$group: {_id: \"\$provider\", count: {\$sum: 1}, last_seen: {\$max: \"\$last_seen_at\"}}}
]).toArray()
"'
```

**Location:** `~/git/sauron/sauron/jobs/llm_benchmarks/provider_discovery.py`

---

## Deployment Management

### Accessing Bedrock EC2 (ml-tuner)

SSH is disabled. Use AWS SSM:

```bash
# 1. Login to AWS SSO
aws sso login --profile zh-ml-mlengineer

# 2. Connect via SSM
aws ssm start-session --target i-0b43e0b5f1ee7c5e9 --region us-east-1 --profile zh-ml-mlengineer
```

**Instance Details:**
- Instance ID: `i-0b43e0b5f1ee7c5e9`
- Account: zh-ml (766806801073)
- IAM Role: `MLTuner` (includes `BedrockFullAccess`)
- OS: Ubuntu with Docker and Python 3.12 (uv)

### Coolify API (clifford management)

**See:** `~/git/me/mytech/operations/coolify-api.md` for full documentation.

**Quick commands:**

```bash
# Get token
TOKEN=$(ssh clifford "sudo cat /var/lib/docker/data/coolify-api/token.env | cut -d= -f2")

# Find application UUID
ssh clifford "curl -s -H 'Authorization: Bearer $TOKEN' 'http://localhost:8000/api/v1/applications'" | jq '.[] | select(.name=="llm-bench-service") | {uuid, name}'

# List env vars
APP_UUID="hg04gw08gwcc4c0400k848wc"
ssh clifford "curl -s -H 'Authorization: Bearer $TOKEN' 'http://localhost:8000/api/v1/applications/$APP_UUID/envs'" | jq '.[] | {key, value}'

# Update env var
ssh clifford "curl -X PATCH -H 'Authorization: Bearer $TOKEN' -H 'Content-Type: application/json' \
  'http://localhost:8000/api/v1/applications/$APP_UUID/envs' \
  -d '{\"key\": \"FRESH_MINUTES\", \"value\": \"180\"}'"

# Deploy (triggers rebuild)
ssh clifford "curl -X POST -H 'Authorization: Bearer $TOKEN' -H 'Content-Type: application/json' \
  -d '{\"uuid\": \"$APP_UUID\"}' 'http://localhost:8000/api/v1/deploy'"
```

**Key env vars:**

| Variable | Purpose | Example |
|----------|---------|---------|
| `BENCHMARK_PROVIDERS` | Which providers to run | `anthropic,groq,openai,vertex` |
| `FRESH_MINUTES` | Benchmark interval | `180` (3 hours) |
| `MONGODB_URI` | Database connection | `mongodb://writer:...@host.docker.internal:27017/llm-bench` |
| `OPENAI_API_KEY` | OpenAI access | `sk-...` |

### Checking Logs

```bash
# clifford runner
ssh clifford 'docker logs $(docker ps -qf "name=llm-bench-service") --tail 100'

# Follow live
ssh clifford 'docker logs -f $(docker ps -qf "name=llm-bench-service")'

# Check for specific errors
ssh clifford 'docker logs $(docker ps -qf "name=llm-bench-service") 2>&1 | grep "Error\|Failed\|Exception"'
```

### Verifying Deployment

```bash
# Check container is running
ssh clifford 'docker ps | grep llm-bench'

# Check env vars in running container
ssh clifford 'docker exec $(docker ps -qf "name=llm-bench-service") env | grep BENCHMARK'

# Test MongoDB connection
ssh clifford 'docker exec $(docker ps -qf "name=llm-bench-service") python -c "from pymongo import MongoClient; import os; c = MongoClient(os.getenv(\"MONGODB_URI\")); print(c.list_database_names())"'
```

---

## Health Monitoring

**Daily email:** Sent by Sauron at 08:00 UTC

**Subject:** `[CRITICAL/WARNING/INFO] LLM Benchmarks Daily Health`

**Contents:**
- Error rate by provider
- Models with only errors
- Likely deprecated models
- Code fixes needed

**Manual health check:**

```bash
# Quick health overview (last 24h)
var yesterday = new Date(Date.now() - 24*60*60*1000);
var successes = db.metrics_cloud_v2.countDocuments({run_ts: {$gte: yesterday}});
var errors = db.errors_cloud.countDocuments({ts: {$gte: yesterday}});
print("Successes: " + successes + ", Errors: " + errors + " (" + (errors/(successes+errors)*100).toFixed(1) + "% error rate)");

# Provider breakdown
db.errors_cloud.aggregate([
  {$match: {ts: {$gte: yesterday}}},
  {$group: {_id: "$provider", errors: {$sum: 1}}},
  {$sort: {errors: -1}}
]).forEach(e => {
  var provider = e._id;
  var s = db.metrics_cloud_v2.countDocuments({provider: provider, run_ts: {$gte: yesterday}});
  print(provider + ": " + s + " ok / " + e.errors + " err (" + (e.errors/(s+e.errors)*100).toFixed(1) + "%)");
});
```

---

## Troubleshooting Guide

### Issue: Bedrock models all failing with ProfileNotFound

**Symptom:** `ProfileNotFound: The config profile (zh-ml-mlengineer) could not be found`

**Cause:** `AWS_PROFILE` env var set in clifford's llm-bench-service

**Solution:** Remove `AWS_PROFILE` from Coolify env vars. Bedrock should only run on EC2 with IAM role.

### Issue: Groq models failing with "output_tokens not within 10%"

**Symptom:** Guard models output 1-2 tokens, compound models output 1000+ tokens

**Cause:** These models aren't suitable for fixed-token benchmarking

**Solution:** Disable unsuitable models:
```bash
db.models.updateOne(
  {provider: "groq", model_id: "llama-guard-4-12b"},
  {$set: {enabled: false, disabled_reason: "Guard model outputs only 2 tokens", disabled_at: new Date()}}
)
```

### Issue: OpenAI o3-mini failing with "empty chat content"

**Symptom:** `Unsupported parameter: 'max_tokens' is not supported`

**Cause:** Reasoning models require Responses API

**Solution:** Update to latest code (as of 2026-01-18, this is implemented). Model will auto-detect and use `/v1/responses`.

### Issue: Vertex Claude models hitting rate limits

**Symptom:** `429 - Resource has been exhausted (e.g. check quota)`

**Cause:** Vertex AI quota exhausted

**Solution:**
1. Increase quota in GCP console
2. Reduce benchmark frequency
3. Temporarily disable some Claude models

### Issue: No benchmarks running at all

**Debugging steps:**

```bash
# 1. Check container is running
ssh clifford 'docker ps | grep llm-bench-service'

# 2. Check recent logs for errors
ssh clifford 'docker logs --tail 50 $(docker ps -qf "name=llm-bench-service") 2>&1'

# 3. Check BENCHMARK_PROVIDERS setting
ssh clifford 'docker exec $(docker ps -qf "name=llm-bench-service") env | grep BENCHMARK_PROVIDERS'

# 4. Check if models are enabled
mongosh "$MONGODB_URI" --quiet --eval 'db.models.countDocuments({enabled: true})'

# 5. Check last successful run
mongosh "$MONGODB_URI" --quiet --eval 'db.metrics_cloud_v2.find().sort({run_ts: -1}).limit(1).forEach(printjson)'
```

---

## Development Workflow

### Local Testing

```bash
# Install dependencies
uv sync

# Run single model
uv run python api/bench_headless.py --providers openai --limit 1

# Test specific provider implementation
uv run python test_openai_provider.py

# Test reasoning models (o1/o3/o4)
uv run python test_reasoning_models_e2e.py
```

### Adding a New Provider

1. Create provider file: `api/llm_bench/cloud/providers/{provider_name}.py`
2. Implement `generate(config: CloudConfig, run_config: dict) -> dict`
3. Add to `PROVIDER_MODULES` in `bench_headless.py`
4. Add API keys to Coolify env vars
5. Add to `BENCHMARK_PROVIDERS` env var
6. Test locally first

**Required metrics format:**
```python
{
    "gen_ts": datetime,           # Timestamp
    "requested_tokens": int,      # What we asked for
    "output_tokens": int,         # What we got
    "generate_time": float,       # Total time (seconds)
    "tokens_per_second": float,   # Speed
    "time_to_first_token": float, # Latency (optional)
    "times_between_tokens": list, # Inter-token times (optional)
}
```

---

## Related Documentation

- **Dashboard:** `~/git/llmbench/llm-benchmarks-dashboard/backend/CLAUDE.md`
- **Sauron (scheduler):** `~/git/sauron/AGENTS.md`
- **Coolify API:** `~/git/me/mytech/operations/coolify-api.md`
- **OpenAI Reasoning Models:** `REASONING_MODELS.md` (this repo)
- **Provider Discovery:** Implemented in Sauron, stores to `provider_catalog` collection
