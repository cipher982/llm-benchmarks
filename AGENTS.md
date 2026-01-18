# LLM Benchmarks API Service

Benchmark runner that calls LLM APIs and measures performance. Deployed as two instances:
- **clifford** (VPS): All providers except Bedrock
- **EC2**: Bedrock only (uses IAM role)

---

## Repository Structure

**Parent directory structure:**
```
~/git/llmbench/
├── llm-benchmarks/           # THIS REPO - Benchmark runner
└── llm-benchmarks-dashboard/ # Separate repo - Next.js dashboard
```

Always `cd` into the specific subdirectory before git operations.

**Key files:**
- `api/bench_headless.py` - Main runner (daemon mode)
- `api/llm_bench/cloud/providers/` - Provider implementations
- `api/llm_bench/models_db.py` - Loads models from MongoDB
- `REASONING_MODELS.md` - OpenAI o1/o3/o4 documentation
- `TROUBLESHOOTING.md` - Error patterns and solutions

---

## Architecture

### Two Runners

| Instance | Providers | Why Separate |
|----------|-----------|--------------|
| **clifford** | anthropic, cerebras, deepinfra, fireworks, groq, openai, together, vertex | Direct MongoDB access |
| **EC2** | bedrock | Needs AWS IAM role + MongoDB via HTTP bridge |

**HTTP Ingest Bridge:** EC2 POSTs results to `https://bench-ingest.drose.io` (on clifford), which writes to MongoDB.

**Configuration:** Set via Coolify env vars (see `~/git/me/mytech/operations/coolify-api.md`)

---

## MongoDB

**Connection:** Via Coolify env vars or ask user for `MONGODB_URI`

**Key collections:**
- `models` - Enabled models (provider, model_id, enabled, deprecated)
- `metrics_cloud_v2` - Successful runs (run_ts, tokens_per_second)
- `errors_cloud` - Failed runs (ts, message, stage)
- `provider_catalog` - Models discovered from provider APIs (managed by Sauron)

**Quick commands:**
```bash
# Check errors (last 24h)
mongosh "$MONGODB_URI" --eval "db.errors_cloud.aggregate([{\$match: {ts: {\$gte: new Date(Date.now()-86400000)}}}, {\$group: {_id: '\$provider', count: {\$sum: 1}}}])"

# List enabled models
mongosh "$MONGODB_URI" --eval "db.models.find({enabled: true}, {provider: 1, model_id: 1})"

# Disable a model
mongosh "$MONGODB_URI" --eval "db.models.updateOne({provider: 'X', model_id: 'Y'}, {\$set: {enabled: false, disabled_reason: 'Reason', disabled_at: new Date()}})"
```

---

## Common Issues

| Symptom | Quick Fix |
|---------|-----------|
| Bedrock `ProfileNotFound: zh-ml-mlengineer` | Remove `AWS_PROFILE` from clifford env vars (Bedrock runs on EC2 only) |
| Groq `output_tokens not within 10%` | Disable guard/compound models (unsuitable for benchmarking) |
| OpenAI o3-mini `unsupported parameter: max_tokens` | Update code - needs Responses API (see REASONING_MODELS.md) |
| Vertex `429 quota exceeded` | Increase GCP quota or reduce frequency |

**Full troubleshooting:** See `TROUBLESHOOTING.md`

---

## Deployment

**clifford (via Coolify API):**
```bash
# See ~/git/me/mytech/operations/coolify-api.md for token/UUID
TOKEN=$(ssh clifford "sudo cat /var/lib/docker/data/coolify-api/token.env | cut -d= -f2")
ssh clifford "curl -X POST -H 'Authorization: Bearer $TOKEN' -H 'Content-Type: application/json' \
  -d '{\"uuid\": \"hg04gw08gwcc4c0400k848wc\"}' 'http://localhost:8000/api/v1/deploy'"
```

**EC2 (via AWS SSM):**
```bash
aws sso login --profile zh-ml-mlengineer
aws ssm start-session --target i-0b43e0b5f1ee7c5e9 --region us-east-1 --profile zh-ml-mlengineer
```

**Logs:**
```bash
ssh clifford 'docker logs -f $(docker ps -qf "name=llm-bench-service")'
```

---

## Adding/Disabling Models

**Critical rules:**
- **Bedrock:** Use `us.anthropic.*` / `us.meta.*` prefixes (not `anthropic.*` / `meta.*`)
- **OpenAI:** o1/o3/o4 models auto-detected as reasoning models
- **Avoid:** Guard models, compound models, embeddings, TTS, image models

**Commands:**
```bash
# Add
mongosh "$MONGODB_URI" --eval "db.models.insertOne({provider: 'openai', model_id: 'gpt-4o', enabled: true, created_at: new Date()})"

# Disable
mongosh "$MONGODB_URI" --eval "db.models.updateOne({provider: 'groq', model_id: 'llama-guard-4-12b'}, {\$set: {enabled: false, disabled_reason: 'Guard model', disabled_at: new Date()}})"
```

---

## Provider Discovery

**Sauron job:** `llm-bench-provider-discovery` (07:00 UTC daily)

Fetches models from provider APIs → stores in `provider_catalog` → emails new models.

**Providers:** Groq, Together, Cerebras, OpenAI, Anthropic, Fireworks, DeepInfra

**Check catalog:**
```bash
ssh clifford 'mongosh "$MONGODB_URI" --quiet --eval "db.provider_catalog.aggregate([{\$group: {_id: \"\$provider\", count: {\$sum: 1}}}])"'
```

---

## Related Docs

- **Detailed troubleshooting:** `TROUBLESHOOTING.md` (create if complex patterns emerge)
- **Dashboard:** `~/git/llmbench/llm-benchmarks-dashboard/backend/CLAUDE.md`
- **Sauron:** `~/git/sauron/AGENTS.md`
- **Coolify API:** `~/git/me/mytech/operations/coolify-api.md`
- **OpenAI Reasoning:** `REASONING_MODELS.md` (this repo)
