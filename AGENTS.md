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
- **clifford** (all providers) - `ssh clifford`
- **aws-poc** (Bedrock only) - SSH blocked, use AWS SSM (see below)

Both write to the same MongoDB on clifford. For specs and details, see `~/git/mytech/infrastructure/vps.md`.

### Accessing aws-poc (Bedrock EC2)

SSH port 22 is blocked by security policy. Use AWS SSM instead:

```bash
# 1. Login to AWS SSO
aws sso login --profile zh-marketing-preprod-engineer

# 2. Connect via SSM
aws ssm start-session --target i-056bc81c58a387657 --region us-east-1 --profile zh-marketing-preprod-engineer

# 3. Run commands (need sudo for docker)
aws ssm start-session --target i-056bc81c58a387657 --region us-east-1 --profile zh-marketing-preprod-engineer \
  --document-name AWS-StartInteractiveCommand --parameters command='sudo docker ps'
```

Instance ID: `i-056bc81c58a387657`

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
mongosh "mongodb://writer:***REMOVED***@5.161.97.53/llm-bench?authSource=llm-bench" --eval '
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
mongosh "mongodb://writer:***REMOVED***@5.161.97.53/llm-bench?authSource=llm-bench" --eval '
db.models.updateOne(
  { provider: "bedrock", model_id: "the-model-id" },
  { $set: { enabled: false, disabled_reason: "Reason here", disabled_at: new Date() } }
)
'
```

### Checking Current Models

```bash
# List enabled models for a provider
mongosh "mongodb://writer:***REMOVED***@5.161.97.53/llm-bench?authSource=llm-bench" --quiet --eval '
db.models.find({provider: "bedrock", enabled: true}).forEach(d => print(d.model_id))
'

# Check disabled models with reasons
mongosh "mongodb://writer:***REMOVED***@5.161.97.53/llm-bench?authSource=llm-bench" --quiet --eval '
db.models.find({provider: "bedrock", enabled: false, disabled_reason: {$exists: true}}).forEach(d => {
  print(d.model_id + " → " + d.disabled_reason)
})
'
```
