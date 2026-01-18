# LLM Benchmarks Troubleshooting Guide

Detailed error investigation patterns and solutions.

---

## Error Investigation Workflows

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

### Error Analysis by Model

```bash
# Errors by model (last 24h)
var yesterday = new Date(Date.now() - 24*60*60*1000);
db.errors_cloud.aggregate([
  {$match: {provider: "openai", ts: {$gte: yesterday}}},
  {$group: {_id: "$model_name", count: {$sum: 1}, sample_msg: {$first: "$message"}}},
  {$sort: {count: -1}}
]).forEach(e => print(e.count + "x " + e._id + ": " + e.sample_msg.substring(0, 100)))
```

### Success vs Error Ratio

```bash
var weekAgo = new Date(Date.now() - 7*24*60*60*1000);
var model = "gpt-4o";
var successes = db.metrics_cloud_v2.countDocuments({provider: "openai", model_name: model, run_ts: {$gte: weekAgo}});
var errors = db.errors_cloud.countDocuments({provider: "openai", model_name: model, ts: {$gte: weekAgo}});
print(model + ": " + successes + " successes, " + errors + " errors = " + (successes/(successes+errors)*100).toFixed(1) + "% success rate");
```

### Categorize Errors by Type

```bash
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
            else: {$cond: {
              if: {$regexMatch: {input: "$message", regex: /ProfileNotFound/}},
              then: "AWS ProfileNotFound",
              else: "Other"
            }}
          }}
        }}
      }
    }
  }},
  {$group: {_id: "$error_type", count: {$sum: 1}}},
  {$sort: {count: -1}}
])
```

---

## Common Error Patterns

### 1. Bedrock ProfileNotFound

**Symptom:**
```
ProfileNotFound: The config profile (zh-ml-mlengineer) could not be found
```

**Cause:** `AWS_PROFILE` env var set in clifford's llm-bench-service container

**Solution:**
```bash
# Remove AWS_PROFILE from Coolify env vars
TOKEN=$(ssh clifford "sudo cat /var/lib/docker/data/coolify-api/token.env | cut -d= -f2")
APP_UUID="hg04gw08gwcc4c0400k848wc"

# Find the env var UUID
ssh clifford "curl -s -H 'Authorization: Bearer $TOKEN' \
  'http://localhost:8000/api/v1/applications/$APP_UUID/envs'" | jq '.[] | select(.key=="AWS_PROFILE")'

# Delete it
ssh clifford "curl -X DELETE -H 'Authorization: Bearer $TOKEN' \
  'http://localhost:8000/api/v1/applications/$APP_UUID/envs/ENV_UUID_HERE'"

# Redeploy
ssh clifford "curl -X POST -H 'Authorization: Bearer $TOKEN' -H 'Content-Type: application/json' \
  -d '{\"uuid\": \"$APP_UUID\"}' 'http://localhost:8000/api/v1/deploy'"
```

**Why this matters:** Bedrock should ONLY run on EC2 with IAM role, not on clifford.

---

### 2. Token Count Mismatches

**Symptom:**
```
output_tokens 1729 not within 10% of requested 64
```

**Causes:**

| Model Type | Output | Why Unsuitable |
|------------|--------|----------------|
| Guard models | 1-2 tokens | Just output "safe"/"unsafe" |
| Compound/agentic | 1000+ tokens | Elaborate beyond request |
| Thinking models | 3 tokens | Use tokens for internal reasoning |

**Solution:** Disable unsuitable models:
```bash
db.models.updateOne(
  {provider: "groq", model_id: "llama-guard-4-12b"},
  {$set: {enabled: false, disabled_reason: "Guard model outputs only 2 tokens", disabled_at: new Date()}}
)
```

**Models to avoid:**
- Guard: `*-guard-*`
- Compound: `*compound*`
- Examples: `llama-guard-4-12b`, `compound-beta`, `groq/compound-mini`

---

### 3. OpenAI Reasoning Models (o1/o3/o4)

**Symptom:**
```
Unsupported parameter: 'max_tokens' is not supported with this model.
Use 'max_completion_tokens' instead.
```

Or:
```
This model is only supported in v1/responses and not in v1/chat/completions
```

**Cause:** Reasoning models (o1, o3, o4 series) require different API endpoint

**Solution:** Code implementation (as of 2026-01-18):
- Auto-detects reasoning models by prefix
- Routes to `/v1/responses` endpoint
- Uses `max_output_tokens` parameter
- See `REASONING_MODELS.md` for implementation details

**Affected models:** o1, o1-pro, o3, o3-mini, o3-pro, o4, o4-mini

---

### 4. Vertex Rate Limits

**Symptom:**
```
429 - Resource has been exhausted (e.g. check quota)
Quota exceeded for aiplatform.googleapis.com/online_prediction_requests_per_base_model
```

**Cause:** Vertex AI quota exhausted (common with Claude models)

**Solutions:**

**Option 1:** Increase quota in GCP console
```bash
# Check current quota usage
gcloud compute project-info describe --project=YOUR_PROJECT
```

**Option 2:** Reduce benchmark frequency
```bash
# Update FRESH_MINUTES (default: 180 = 3 hours)
# Via Coolify API - increase to 360 (6 hours) or 720 (12 hours)
```

**Option 3:** Temporarily disable high-traffic models
```bash
# Disable some Claude models on Vertex
db.models.updateMany(
  {provider: "vertex", model_id: /claude/},
  {$set: {enabled: false, disabled_reason: "Temporarily disabled due to quota limits"}}
)
```

---

### 5. No Benchmarks Running

**Debugging checklist:**

```bash
# 1. Check container is running
ssh clifford 'docker ps | grep llm-bench-service'

# 2. Check recent logs for errors
ssh clifford 'docker logs --tail 50 $(docker ps -qf "name=llm-bench-service") 2>&1'

# 3. Check BENCHMARK_PROVIDERS setting
ssh clifford 'docker exec $(docker ps -qf "name=llm-bench-service") env | grep BENCHMARK_PROVIDERS'

# 4. Check if models are enabled in database
mongosh "$MONGODB_URI" --quiet --eval 'db.models.countDocuments({enabled: true})'

# 5. Check last successful run
mongosh "$MONGODB_URI" --quiet --eval 'db.metrics_cloud_v2.find().sort({run_ts: -1}).limit(1).forEach(printjson)'

# 6. Check for blocking errors
ssh clifford 'docker logs $(docker ps -qf "name=llm-bench-service") 2>&1 | tail -100 | grep -i "error\|exception\|failed"'
```

**Common causes:**
- Container not running (check Coolify)
- `BENCHMARK_PROVIDERS` empty or misconfigured
- All models disabled in database
- MongoDB connection failed
- API keys missing/expired

---

## Health Report Analysis

Daily email format:
```
Status: CRITICAL/WARNING/INFO
- Error rate by provider
- Models with only errors
- Likely deprecated models
- Code fixes needed
```

### Manual Health Check

```bash
# Quick overview (last 24h)
var yesterday = new Date(Date.now() - 24*60*60*1000);
var successes = db.metrics_cloud_v2.countDocuments({run_ts: {$gte: yesterday}});
var errors = db.errors_cloud.countDocuments({ts: {$gte: yesterday}});
print("Successes: " + successes + ", Errors: " + errors + " (" + (errors/(successes+errors)*100).toFixed(1) + "% error rate)");

# Provider breakdown with success rates
db.errors_cloud.aggregate([
  {$match: {ts: {$gte: yesterday}}},
  {$group: {_id: "$provider", errors: {$sum: 1}}},
  {$sort: {errors: -1}}
]).forEach(e => {
  var provider = e._id;
  var s = db.metrics_cloud_v2.countDocuments({provider: provider, run_ts: {$gte: yesterday}});
  print(provider + ": " + s + " ok / " + e.errors + " err (" + (e.errors/(s+e.errors)*100).toFixed(1) + "%)");
});

# Models with ONLY errors (never succeed)
var failingModels = db.errors_cloud.distinct("model_name", {ts: {$gte: yesterday}});
var successfulModels = new Set(db.metrics_cloud_v2.distinct("model_name", {run_ts: {$gte: yesterday}}));
print("\nModels with ONLY errors:");
failingModels.filter(m => !successfulModels.has(m)).forEach(m => print("  " + m));
```

---

## Provider-Specific Notes

### Bedrock

**Critical:** Model IDs must use correct prefixes
- New Anthropic/Meta: `us.anthropic.*`, `us.meta.*`
- Old Meta: `meta.*`
- Others: `mistral.*`, `cohere.*`, `amazon.*`

**Why:** `us.` prefix enables cross-region inference profiles. Without it, models fail with "Invocation with on-demand throughput isn't supported".

**DO NOT benchmark:**
- Image models (nova-canvas)
- Non-prefixed Anthropic models

### OpenAI

**Reasoning models:** o1, o3, o4 series auto-detected, use Responses API

**DO NOT benchmark:**
- Embeddings (text-embedding-*)
- TTS/Audio (tts-*, whisper-*)
- Image (dall-e, sora)
- Moderation models

### Groq

**DO NOT benchmark:**
- Guard models (llama-guard-*, output 1-2 tokens)
- Compound models (compound-*, output 1000+ tokens)
- GPT-OSS models (broken, output 1 token)

### Vertex

**Common issues:**
- Rate limits on Claude models (quota)
- Gemini thinking models output only 3 tokens

**Check quota:**
```bash
gcloud alpha billing accounts get-iam-policy ACCOUNT_ID
```

---

## Deployment Verification

### After Deploy

```bash
# Wait for deployment
sleep 60

# Check new container started
ssh clifford 'docker ps --format "{{.Names}}\t{{.Status}}\t{{.CreatedAt}}" | grep llm-bench-service'

# Verify env vars loaded
ssh clifford 'docker exec $(docker ps -qf "name=llm-bench-service") env | grep BENCHMARK_PROVIDERS'

# Check first benchmark run
ssh clifford 'docker logs -f $(docker ps -qf "name=llm-bench-service")' # Watch for ✅ Success lines

# Verify MongoDB writes
mongosh "$MONGODB_URI" --eval "db.metrics_cloud_v2.find().sort({run_ts: -1}).limit(5)"
```

---

## Model Management Best Practices

### When to Disable (Not Delete)

**Disable when:**
- Model deprecated by provider
- API incompatibility discovered
- Consistently high error rate (>50%)
- Unsuitable for benchmarking (guard, compound, etc.)

**Delete when:**
- Never (use disabled flag instead)

**Why:** Preserves history and allows re-enabling if issues are fixed.

### Checking Model History

```bash
# Recently disabled models
var weekAgo = new Date(Date.now() - 7*24*60*60*1000);
db.models.find({enabled: false, disabled_at: {$gte: weekAgo}}, {provider: 1, model_id: 1, disabled_reason: 1}).forEach(d => {
  print(d.provider + ":" + d.model_id + " → " + d.disabled_reason)
})

# Models with recent errors but still enabled
var yesterday = new Date(Date.now() - 24*60*60*1000);
var errorModels = db.errors_cloud.distinct("model_name", {ts: {$gte: yesterday}});
errorModels.forEach(m => {
  var doc = db.models.findOne({model_id: m, enabled: true});
  if (doc) {
    var errorCount = db.errors_cloud.countDocuments({model_name: m, ts: {$gte: yesterday}});
    var successCount = db.metrics_cloud_v2.countDocuments({model_name: m, run_ts: {$gte: yesterday}});
    print(m + ": " + errorCount + " errors, " + successCount + " successes");
  }
})
```

---

## Coolify Operations

**Full docs:** `~/git/me/mytech/operations/coolify-api.md`

### Key Commands

```bash
# Get API token
TOKEN=$(ssh clifford "sudo cat /var/lib/docker/data/coolify-api/token.env | cut -d= -f2")

# llm-bench-service UUID
APP_UUID="hg04gw08gwcc4c0400k848wc"

# List env vars
ssh clifford "curl -s -H 'Authorization: Bearer $TOKEN' \
  'http://localhost:8000/api/v1/applications/$APP_UUID/envs'" | jq '.[] | {key, value}'

# Update env var
ssh clifford "curl -X PATCH -H 'Authorization: Bearer $TOKEN' -H 'Content-Type: application/json' \
  'http://localhost:8000/api/v1/applications/$APP_UUID/envs' \
  -d '{\"key\": \"FRESH_MINUTES\", \"value\": \"180\"}'"

# Deploy
ssh clifford "curl -X POST -H 'Authorization: Bearer $TOKEN' -H 'Content-Type: application/json' \
  -d '{\"uuid\": \"$APP_UUID\"}' 'http://localhost:8000/api/v1/deploy'"

# Check deployment status
ssh clifford "curl -s -H 'Authorization: Bearer $TOKEN' \
  'http://localhost:8000/api/v1/deployments/DEPLOY_UUID'" | jq '{status, created_at}'
```

### Important Env Vars

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `BENCHMARK_PROVIDERS` | Which providers to run | `all` | Comma-separated, exclude bedrock on clifford |
| `FRESH_MINUTES` | Run interval | 30 | 180 = 3 hours is common |
| `MONGODB_URI` | Database connection | Required | Use `host.docker.internal` from container |

---

## EC2 (Bedrock) Operations

**Access:**
```bash
aws sso login --profile zh-ml-mlengineer
aws ssm start-session --target i-0b43e0b5f1ee7c5e9 --region us-east-1 --profile zh-ml-mlengineer
```

**Instance:** i-0b43e0b5f1ee7c5e9 (zh-ml account 766806801073)

**Runner:** `bench_simple_runner.py` with HTTP ingest

**Required env vars:**
- `INGEST_API_URL=https://bench-ingest.drose.io/ingest`
- `INGEST_API_KEY=xxx`
- `BENCHMARK_MODELS=us.anthropic...` (comma-separated)

---

## Development

### Testing Locally

```bash
cd ~/git/llmbench/llm-benchmarks

# Install dependencies
uv sync

# Run single provider
uv run python api/bench_headless.py --providers openai --limit 1

# Test reasoning models
uv run python test_reasoning_models_e2e.py

# Test specific provider
uv run python test_openai_provider.py
```

### Adding New Providers

1. Create `api/llm_bench/cloud/providers/{name}.py`
2. Implement `generate(config: CloudConfig, run_config: dict) -> dict`
3. Add to `PROVIDER_MODULES` dict in `bench_headless.py`
4. Add API key to Coolify env vars
5. Add to `BENCHMARK_PROVIDERS`
6. Test locally before production

**Required return format:**
```python
{
    "gen_ts": datetime,           # Timestamp
    "requested_tokens": int,      # Requested
    "output_tokens": int,         # Actual
    "generate_time": float,       # Total seconds
    "tokens_per_second": float,   # Speed
    "time_to_first_token": float, # Latency (optional)
    "times_between_tokens": list, # Inter-token times (optional)
}
```

---

## Model Addition Guidelines

### Check Before Adding

```bash
# 1. Is it enabled in another provider?
db.models.findOne({model_id: "model-name"})

# 2. Is it in provider_catalog?
db.provider_catalog.findOne({model_id: "model-name"})

# 3. Similar models exist?
db.models.find({model_id: /similar-pattern/})
```

### Model ID Patterns to Avoid

| Pattern | Reason |
|---------|--------|
| `*embedding*` | Not chat |
| `*tts*`, `*whisper*` | Audio |
| `*dall-e*`, `*sora*` | Image |
| `*moderation*` | Not generation |
| `*-guard-*` | Output 1-2 tokens |
| `*compound*` | Output 1000+ tokens |

### Bedrock Model ID Examples

✅ **Correct:**
```
us.anthropic.claude-opus-4-5-20251101-v1:0
us.meta.llama4-maverick-17b-instruct-v1:0
```

❌ **Wrong (will fail):**
```
anthropic.claude-opus-4-5-20251101-v1:0  # Missing us. prefix
meta.llama4-maverick-17b-instruct-v1:0   # New Llama needs us. prefix
```

---

## Monitoring

### Check Recent Activity

```bash
# Last 10 successful runs
db.metrics_cloud_v2.find({}, {provider: 1, model_name: 1, run_ts: 1, tokens_per_second: 1}).sort({run_ts: -1}).limit(10)

# Last 10 errors
db.errors_cloud.find({}, {provider: 1, model_name: 1, ts: 1, message: 1}).sort({ts: -1}).limit(10)

# Check specific model
var model = "gpt-4o";
var latest = db.metrics_cloud_v2.findOne({model_name: model}, {run_ts: 1, tokens_per_second: 1}, {sort: {run_ts: -1}});
print("Latest run: " + latest.run_ts + " at " + latest.tokens_per_second.toFixed(2) + " tokens/sec");
```

### Provider Catalog (Discovery)

```bash
# Models discovered by provider
db.provider_catalog.aggregate([
  {$group: {_id: "$provider", total: {$sum: 1}, not_in_db: {$sum: {$cond: ["$in_our_db", 0, 1]}}}},
  {$sort: {total: -1}}
])

# New models to add (not in our database)
db.provider_catalog.find({in_our_db: false, provider: "openai"}, {model_id: 1, name: 1})
```
