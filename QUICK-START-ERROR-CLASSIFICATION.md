# Quick Start: LLM Error Classification

## TL;DR

We replaced brittle keyword matching with intelligent LLM classification. Errors are now classified once per unique fingerprint (100-1000x fewer API calls).

## Quick Commands

```bash
# Classify errors (run this after deployment)
cd /Users/davidrose/git/llmbench/llm-benchmarks
./ops/classify-errors.sh

# Run daily health check (includes classification)
python ops/daily-health-check.py

# Test the system
uv run env PYTHONPATH=api python -c "from llm_bench.ops.llm_error_classifier import classify_unclassified_rollups; print('OK')"
```

## What You Need

```bash
# API Key (one of these)
export ANTHROPIC_API_KEY="sk-ant-..."  # Preferred (cheaper)
export OPENAI_API_KEY="sk-..."         # Fallback

# MongoDB (already configured)
export MONGODB_URI="mongodb://..."
```

## How It Works

### Before (Old System)
```
Error occurs → classify_error() with keyword matching → log to DB
Problem: Brittle, inaccurate, hard to maintain
```

### After (New System)
```
Error occurs → classify_error() with HTTP status only → log as "unknown" → rollup created
              ↓
Daily (or on-demand): LLM batch classifies unique fingerprints → updates rollups
Result: Accurate, maintainable, 100-1000x fewer LLM calls
```

## Error Categories

| Category | What It Means | Action Needed |
|----------|---------------|---------------|
| `auth` | Bad API key/credentials | Check env vars |
| `billing` | Payment/quota issue | Check account |
| `rate_limit` | Too many requests | Temporary, will resolve |
| `hard_model` | Model doesn't exist | Mark deprecated |
| `hard_capability` | **Code needs updating** | Fix API usage |
| `transient_provider` | Server error | Temporary, monitor |
| `network` | Connection problem | Check infrastructure |

**Key distinction:** `hard_capability` means OUR code is wrong, `hard_model` means the provider removed the model.

## Files Changed

1. **api/llm_bench/ops/error_taxonomy.py** - Simplified to HTTP status only
2. **ops/daily-health-check.py** - Added classification step
3. **api/llm_bench/ops/llm_error_classifier.py** - NEW: LLM classifier

## Cost

- Claude Haiku (preferred): ~$0.02 per day
- GPT-4o-mini (fallback): ~$0.01 per day
- Only charged for NEW error patterns

## Testing

```bash
# Test imports
uv run env PYTHONPATH=api python -c "from llm_bench.ops.llm_error_classifier import classify_unclassified_rollups; print('OK')"

# Test classification (without HTTP status = UNKNOWN)
uv run env PYTHONPATH=api python -c "from llm_bench.ops.error_taxonomy import classify_error; print(classify_error(message='Model not found').kind)"
# Output: ErrorKind.UNKNOWN ✓

# Test classification (with HTTP 404 = HARD_MODEL)
uv run env PYTHONPATH=api python -c "from llm_bench.ops.error_taxonomy import classify_error; print(classify_error(message='Error code: 404').kind)"
# Output: ErrorKind.HARD_MODEL ✓
```

## Deployment Checklist

- [ ] Code deployed to servers (clifford, aws-poc)
- [ ] API keys set in environment (ANTHROPIC_API_KEY or OPENAI_API_KEY)
- [ ] Run initial classification: `./ops/classify-errors.sh --all`
- [ ] Verify daily health check runs: `python ops/daily-health-check.py --dry-run`
- [ ] Monitor email reports for quality

## Troubleshooting

**Q: "No module named 'httpx'"**
A: Use `uv run env PYTHONPATH=api python ...` to run scripts

**Q: "No API key available"**
A: Set ANTHROPIC_API_KEY or OPENAI_API_KEY in environment

**Q: "No unclassified errors found"**
A: Good! The system is working. Wait for new errors or check MongoDB:
```bash
mongosh "$MONGODB_URI" --eval 'db.error_rollups.countDocuments({error_kind: "unknown"})'
```

**Q: "Classification seems wrong"**
A: Check the `classification_reasoning` field in the rollup document. Low confidence (<0.7) indicates ambiguity.

## More Info

- Full docs: `api/llm_bench/ops/README-ERROR-CLASSIFICATION.md`
- Implementation summary: `IMPLEMENTATION-SUMMARY.md`
