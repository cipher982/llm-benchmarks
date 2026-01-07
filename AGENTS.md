# LLM Benchmarks API Service

This is the backend code that has a Docker container with a scheduler and makes calls to benchmark providers across cloud and local services. Cloud providers represent ~90% of the effort and value; local model benchmarking has been deprioritized.

## Repository Structure

### Nested Repository Structure

**Note:** `~/git/llmbench/` is NOT a git repository itself. It contains two separate repos:
- `~/git/llmbench/llm-benchmarks/` - Benchmark runner
- `~/git/llmbench/llm-benchmarks-dashboard/` - Dashboard UI (Next.js)

Always `cd` into the specific subdirectory before git operations.

## Deployment

**Two deployment instances:**
- **clifford** (all providers except Bedrock) - `ssh clifford`
- **ml-tuner-demo-server** (Bedrock only) - Access via AWS SSM (see below)

### Architecture: HTTP Ingest

Due to MongoDB being restricted to Tailscale/local access, Bedrock runners on AWS use an **HTTP Ingest Bridge**:

1. **bench-ingest API**: Deployed on `clifford` at `https://bench-ingest.drose.io`. It receives benchmark results via HTTPS and writes them to MongoDB.
2. **bench_simple_runner.py**: A lightweight runner deployed on EC2 that reuses the library's provider logic but POSTs results to the ingest API instead of connecting to MongoDB directly.

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

### Running the Bedrock Benchmarks

The runner is typically deployed as a Docker container or managed via `uv`:

```bash
# Start the daemon (every 30 minutes)
uv run python api/bench_simple_runner.py --provider bedrock --daemon --interval 30
```

Required environment variables:
- `INGEST_API_URL=https://bench-ingest.drose.io/ingest`
- `INGEST_API_KEY=xxx`
- `BENCHMARK_MODELS=us.anthropic.claude-3-5-sonnet-20241022-v2:0,...`

---

## Adding Models to the Database

Models are stored in the `models` collection in MongoDB. The scheduler reads enabled models from this collection.

### Model Document Schema

```javascript
{
  provider: "bedrock",           // Provider name
  model_id: "us.anthropic...",   // The actual API model ID
  enabled: true,                 // Set false to disable without deleting
  deprecated: false,             // For lifecycle tracking
  created_at: ISODate(),         // When added
  // Optional fields for disabled models:
  disabled_reason: "...",        // Why it was disabled
  disabled_at: ISODate()         // When disabled
}
```

### ⚠️ Bedrock Model ID Rules (CRITICAL)

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

### Adding a New Model

```bash
mongosh "$MONGODB_URI" --eval '
db.models.insertOne({
  provider: "bedrock",
  model_id: "us.anthropic.claude-NEW-MODEL-v1:0",  // Use correct prefix!
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
  { provider: "bedrock", model_id: "the-model-id" },
  { $set: { enabled: false, disabled_reason: "Reason here", disabled_at: new Date() } }
)
'
```

### Checking Current Models

```bash
# List enabled models for a provider
mongosh "$MONGODB_URI" --quiet --eval '
db.models.find({provider: "bedrock", enabled: true}).forEach(d => print(d.model_id))
'

# Check disabled models with reasons
mongosh "$MONGODB_URI" --quiet --eval '
db.models.find({provider: "bedrock", enabled: false, disabled_reason: {$exists: true}}).forEach(d => {
  print(d.model_id + " → " + d.disabled_reason)
})
'
```
