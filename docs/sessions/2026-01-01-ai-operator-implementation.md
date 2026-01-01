# AI Operator Implementation Session - 2026-01-01

## Summary

Built an AI Operator system for the LLM benchmarks platform - LLM-powered model lifecycle management. Completed all 4 phases but discovered a critical inefficiency in the final testing phase that needs to be fixed.

---

## What Was Built

### Phase 1: OpenRouter Cleanup
- Disabled 175 OpenRouter models (duplicates of direct providers)
- Disabled 4 known broken models (groq, cerebras, vertex, openai)
- Total enabled models reduced from ~397 to 222

### Phase 2: Operator Module (`api/llm_bench/operator/`)
```
operator/
├── engine.py      # Core LLM decision logic
├── io.py          # MongoDB integration
├── actions.py     # Auto-execution logic (with 404/401 + 48h check)
└── cli.py         # CLI: analyze, pending, execute
```

- LLM-powered reasoning replaces hardcoded if/else rules
- Falls back to `classifier.py` if LLM fails
- Stores decisions in `model_status.operator_decision` field

### Phase 3: Health Check Integration
- Modified `ops/daily-health-check.py` to call operator
- Email now includes AUTO-EXECUTED ACTIONS and SUGGESTIONS FOR REVIEW sections
- Auto-executes high-confidence (≥0.95) disables for 404/401 errors >48h
- Copy-paste MongoDB commands for manual approval

### Phase 4: Discovery System (`api/llm_bench/discovery/`)
```
discovery/
├── openrouter.py  # Fetch OpenRouter catalog (free, no auth)
├── matcher.py     # LLM matching to direct providers
└── cli.py         # CLI: fetch, report, stats
```

- Fetches 353 models from OpenRouter API
- LLM matches to direct providers (anthropic, bedrock, vertex, etc.)
- Generates copy-paste commands to add new models

---

## Git Commits (25 total)

All commits follow SDP-1 format: `phase N: description\n\n[SDP-1] ai-operator`

Key commits:
- `b6a281f` - phase 0: create ai-operator spec
- `762d39f` - phase 1: update spec with execution results
- `e8f9907` - phase 2: update spec with completion status
- `a87b0ac` - phase 3: update spec with implementation results
- `efa5940` - phase 4: update spec with completion status
- `151bd39` - fix: use max_completion_tokens in operator engine
- `a71fc08` - fix: hardcode gpt-5.2 model constant

---

## Lessons Learned

### 1. Model API Parameter Changes
**Issue:** `gpt-5.2` requires `max_completion_tokens`, not `max_tokens`
**Fix:** Updated all operator/discovery code to use correct parameter
**Lesson:** Always test with actual API before assuming parameter compatibility

### 2. Don't Use Env Var Fallbacks for Model Names
**Bad:**
```python
OPENAI_MODEL = os.getenv("OPERATOR_LLM_MODEL", "gpt-5.2")  # Hidden default!
```
**Good:**
```python
OPENAI_MODEL = "gpt-5.2"  # Explicit, no surprises
```
**Lesson:** Hidden defaults create debugging nightmares. Be explicit.

### 3. Commit After Each Change
**Issue:** Batched multiple fixes into single commits
**Lesson:** SDP-1 is clear - commit after EACH meaningful change. File created? Commit. Bug fixed? Commit.

### 4. Credentials in Code
**Issue:** Hardcoded MongoDB credentials in spec and CLI files
**Fix:** Replaced with `$MONGODB_URI` placeholder
**Lesson:** Never commit credentials, even in "example" code

### 5. The Big One: Per-Model LLM Calls Are Wasteful
**Issue:** Current engine makes ONE LLM call per model (523 calls for full analysis)
**Result:** ~260K tokens, ~$1-2 per run, 2.5 minutes runtime
**Should be:** Aggregate by pattern, ONE call with summary, ~5K tokens, $0.01

---

## Current State

### What Works
- All 4 phases implemented and code-reviewed via Codex
- CLI commands functional:
  ```bash
  uv run python -m api.llm_bench.operator.cli analyze --provider groq
  uv run python -m api.llm_bench.discovery.cli fetch
  uv run python -m api.llm_bench.discovery.cli report
  ```
- Health check integration works (tested with `--dry-run`)

### What's Broken/Inefficient
- **Operator makes 523 individual LLM calls** - needs refactoring to batch/aggregate
- Not pushed to origin yet (25 commits ahead)

### Test Results
```
Full operator run:
- Models analyzed: 523
- Time: 2m 45s (~315ms/model)
- Disable recommendations: 79
- Monitor recommendations: 427
- Estimated cost: ~$1-2/run (TOO HIGH)
```

---

## Critical Next Step: Refactor Engine for Batching

The current `engine.py` flow:
```
523 models → 523 LLM calls → 260K tokens → $1-2
```

Should be:
```
523 models
    → Pre-aggregate by pattern
    → Group provider-wide issues (billing, auth)
    → Group common errors (same 404 message)
    → Filter healthy models (recent successes)
    → ~10-20 unique situations
    → ONE LLM call with summary
    → ~5K tokens → $0.01
```

### Aggregation Logic Needed

1. **Provider-level issues**: If 90%+ of a provider's models have same error → one decision for provider
2. **Error pattern grouping**: If 10 models have identical "404 not found" → one decision
3. **Healthy model filtering**: If model had success in last 24h → auto-ignore, no LLM needed
4. **Batch the rest**: Remaining edge cases go in one summary prompt

### New Engine Structure (Proposed)

```python
async def generate_decisions(snapshots):
    # 1. Filter healthy (recent success) → ignore
    healthy = [s for s in snapshots if s.successes.successes_7d > 0]
    needs_analysis = [s for s in snapshots if s not in healthy]

    # 2. Group by provider-wide issues
    provider_issues = detect_provider_wide_issues(needs_analysis)

    # 3. Group by error pattern
    error_groups = group_by_error_pattern(needs_analysis)

    # 4. Build summary for LLM
    summary = format_aggregated_summary(provider_issues, error_groups)

    # 5. ONE LLM call
    decisions = await call_llm_for_batch_decisions(summary)

    return expand_decisions(decisions, healthy, provider_issues, error_groups)
```

---

## Files Changed

### New Files
```
api/llm_bench/operator/__init__.py
api/llm_bench/operator/__main__.py
api/llm_bench/operator/engine.py
api/llm_bench/operator/io.py
api/llm_bench/operator/actions.py
api/llm_bench/operator/cli.py

api/llm_bench/discovery/__init__.py
api/llm_bench/discovery/openrouter.py
api/llm_bench/discovery/matcher.py
api/llm_bench/discovery/cli.py

docs/specs/ai-operator.md
```

### Modified Files
```
ops/daily-health-check.py  # Added operator integration
```

---

## Commands Reference

```bash
# Operator CLI
uv run python -m api.llm_bench.operator.cli analyze --dry-run
uv run python -m api.llm_bench.operator.cli analyze --provider groq
uv run python -m api.llm_bench.operator.cli pending
uv run python -m api.llm_bench.operator.cli execute --no-dry-run --yes

# Discovery CLI
uv run python -m api.llm_bench.discovery.cli fetch
uv run python -m api.llm_bench.discovery.cli stats
uv run python -m api.llm_bench.discovery.cli report --max-matches 20

# Health check
uv run python ops/daily-health-check.py --dry-run
uv run python ops/daily-health-check.py --skip-operator --dry-run
uv run python ops/daily-health-check.py --operator-provider groq --dry-run
```

---

## Next Session Checklist

1. [ ] **Refactor `engine.py`** - Implement batching/aggregation (see proposed structure above)
2. [ ] **Test refactored engine** - Should complete in <10 seconds, <10K tokens
3. [ ] **Push to origin** - 25+ commits waiting
4. [ ] **Deploy to clifford** - Test in production environment
5. [ ] **Schedule discovery fetch** - Add cron for daily OpenRouter catalog sync
6. [ ] **Monitor first health emails** - Verify operator decisions are sensible

---

## Session Stats

- Duration: ~4 hours
- Commits: 25
- Lines of code: ~2,500 (new), ~200 (modified)
- Codex reviews: 8 (all passed after fixes)
- Cost inefficiency discovered: 100x (should be $0.01, currently $1-2)
