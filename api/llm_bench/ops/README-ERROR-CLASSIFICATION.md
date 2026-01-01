# LLM-based Error Classification System

This system replaces brittle keyword-matching with intelligent LLM classification at the error rollup level, reducing API calls by 100-1000x.

## How It Works

### 1. Initial Classification (Hot Path)

When an error occurs, `error_taxonomy.py::classify_error()` does minimal classification:

- Extracts HTTP status codes
- Classifies based ONLY on status codes:
  - 401/403 → `auth`
  - 402 → `billing`
  - 429 → `rate_limit`
  - 5xx → `transient_provider`
  - 404 → `hard_model`
  - Everything else → `unknown`

This keeps the hot path fast and avoids brittle string matching.

### 2. LLM Classification (Async, Batched)

The `llm_error_classifier.py` module runs periodically (e.g., before daily health check) to classify unknown errors:

- Queries `error_rollups` collection for unclassified fingerprints
- Batches them (default: 50 at a time)
- Sends to Claude Haiku or GPT-4o-mini for classification
- Updates rollups with classification results

**Key principle:** Classify once per unique error fingerprint, not per occurrence.

## Error Categories

| Category | Description | Example |
|----------|-------------|---------|
| `auth` | Authentication/authorization issues | Invalid API key, expired token, AWS profile not found |
| `billing` | Payment/quota issues | Insufficient credits, payment required |
| `rate_limit` | Throttling | 429 errors, quota exceeded |
| `hard_model` | Model doesn't exist | Model not found, deprecated, removed |
| `hard_capability` | Wrong API/parameters | Wrong endpoint, unsupported feature, API mismatch |
| `transient_provider` | Server errors | 5xx errors, service unavailable |
| `network` | Connection issues | Timeout, DNS failure, connection reset |
| `unknown` | Cannot determine | Ambiguous errors |

## Usage

### Classify Unclassified Errors

```bash
# From repo root
cd /Users/davidrose/git/llmbench/llm-benchmarks

# Classify up to 200 errors (default)
./ops/classify-errors.sh

# Classify up to 500 errors
./ops/classify-errors.sh --max 500

# Classify ALL unclassified errors
./ops/classify-errors.sh --all

# Use OpenAI instead of Anthropic
./ops/classify-errors.sh --use-openai

# Custom batch size
./ops/classify-errors.sh --batch-size 100
```

### Run from Daily Health Check

The daily health check automatically runs classification before analysis:

```bash
# Normal run (includes classification)
python ops/daily-health-check.py

# Skip classification (faster, but may have stale classifications)
python ops/daily-health-check.py --skip-classification
```

### Run Programmatically

```python
from llm_bench.ops.llm_error_classifier import classify_unclassified_rollups

# Classify up to 200 rollups
results = await classify_unclassified_rollups(batch_size=50, max_rollups=200)

print(f"Updated: {results['updated']}")
print(f"Skipped: {results['skipped']}")
print(f"Errors: {results['errors']}")
```

## Environment Setup

Requires one of these API keys:

```bash
# Anthropic (preferred - cheaper and faster)
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI (fallback)
export OPENAI_API_KEY="sk-..."

# MongoDB (required)
export MONGODB_URI="mongodb://..."
export MONGODB_DB="llm-bench"
export MONGODB_COLLECTION_ERROR_ROLLUPS="error_rollups"
```

## Database Schema

The classifier updates `error_rollups` collection:

```javascript
{
  fingerprint: "abc123...",          // Unique error fingerprint
  provider: "openai",
  model_name: "gpt-4",
  stage: "generate",
  error_kind: "hard_capability",     // Set by LLM
  classification_confidence: 0.95,   // 0.0-1.0
  classification_reasoning: "...",   // Why this classification
  classified_at: ISODate(),          // When classified
  classified_by: "llm",              // Classification source
  count: 42,                         // Number of occurrences
  first_seen: ISODate(),
  last_seen: ISODate(),
  sample_messages: ["...", "..."]    // Up to 3 samples
}
```

## Performance

- **Before:** Every error classified via keyword matching (fast but inaccurate)
- **After:**
  - Hot path: HTTP status only (microseconds)
  - LLM classification: Once per unique fingerprint (seconds, but batched)
  - Typical reduction: 100-1000x fewer LLM calls

Example: 10,000 errors with 50 unique fingerprints = 50 LLM calls instead of 10,000.

## Cost Estimates

- Claude Haiku: ~$0.25 per 1M input tokens, ~$1.25 per 1M output tokens
- GPT-4o-mini: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens

Typical batch of 50 errors:
- Input: ~10K tokens ($0.003 Haiku, $0.0015 GPT-4o-mini)
- Output: ~2K tokens ($0.003 Haiku, $0.0012 GPT-4o-mini)
- Total: ~$0.006 Haiku, ~$0.003 GPT-4o-mini per batch

Daily run with 200 errors = 4 batches = ~$0.024 Haiku, ~$0.012 GPT-4o-mini.

## Testing

```bash
# Test import
uv run env PYTHONPATH=api python -c "from llm_bench.ops.llm_error_classifier import classify_unclassified_rollups; print('OK')"

# Test classification (requires API key)
uv run env PYTHONPATH=api python -m llm_bench.ops.llm_error_classifier --max-rollups 10
```

## Troubleshooting

**Import errors:**
- Use `uv run env PYTHONPATH=api python ...` to run scripts
- The `api/` directory must be in PYTHONPATH

**No API key:**
- Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
- Anthropic is preferred (faster and cheaper)

**No unclassified errors:**
- This is good! Means the system is working
- Check `db.error_rollups.countDocuments({error_kind: "unknown"})` to verify

**Classification seems wrong:**
- Check `classification_confidence` field - low confidence may indicate ambiguity
- Review `classification_reasoning` for explanation
- File an issue if systemic misclassification

## Migration from Old System

The old system (`error_taxonomy.py` with keyword matching) has been simplified to only use HTTP status codes. Existing classifications remain in the database and will be gradually updated as the LLM classifier runs.

To force reclassification of all errors:

```bash
# Mark all rollups as unclassified
mongosh "$MONGODB_URI" --eval 'db.error_rollups.updateMany({}, {$set: {error_kind: "unknown"}})'

# Run classifier
./ops/classify-errors.sh --all
```

**Warning:** This is rarely needed and will incur LLM API costs.
