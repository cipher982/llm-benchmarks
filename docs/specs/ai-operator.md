# AI Operator Spec

**Status:** Ready for Implementation
**Created:** 2026-01-01
**Last Updated:** 2026-01-01
**Author:** David Rose + Claude

---

## Executive Summary

**Goal:** Build an LLM-operated benchmarking service where AI agents manage model lifecycle decisions - discovery, health monitoring, deprecation, and cleanup - with humans setting policy and reviewing audit trails.

**Core Principle:** "If an intern could figure it out, an LLM can too."

**Key Innovation:** Actual LLM reasoning at decision points, not deterministic rules with AI branding. The existing `classifier.py` decision tree becomes a fallback only.

**Immediate Value (Phase 1):**
- Disable 171 duplicate OpenRouter models (saves cost, improves accuracy)
- Fix broken models causing dashboard noise
- Clean foundation for future LLM reasoning

---

## Decision Log

### Decision 1: LLM Reasoning Over Rules

**Choice:** Use LLM calls for decisions, demote `classifier.py` to fallback

**Rationale:**
- Current `classifier.py` uses hardcoded if/else trees (age thresholds, error rates)
- An LLM can reason about the same signals: "This model returns 404 for 48 hours with error message 'model does not exist' → it's gone"
- Better handling of edge cases (temporary outages vs permanent removal)
- Aligns with "AI Operator" vision

**Implementation:**
```python
# OLD (classifier.py - deterministic)
if error_rate > 0.9 and days_failing > 2:
    return LifecycleStatus.LIKELY_DEPRECATED

# NEW (operator - LLM reasoning)
prompt = f"""
Model: {provider}/{model_id}
Error rate: {error_rate:.2%} over {days} days
Sample errors: {error_samples}
Last success: {last_success or 'Never'}

Should this model be disabled? Consider:
- Is this a temporary outage or permanent removal?
- Do error messages indicate the model is gone?
- Has enough time passed to be confident?

Return JSON: {{"action": "disable|monitor|ignore", "confidence": 0.0-1.0, "reasoning": "..."}}
"""
```

### Decision 2: Simple Model ID Matching

**Choice:** Let LLM guess provider model IDs, rely on API feedback loop

**Rationale:**
- Original design suggested web search for model ID mapping (OpenRouter → direct provider)
- But the API returns clear errors when model ID is wrong
- The LLM can learn from these errors and try alternatives
- Much simpler than maintaining mapping tables or complex web search logic

**Implementation:**
```python
# LLM sees: OpenRouter has "meta-llama/llama-3.3-70b-instruct"
# LLM guesses: Bedrock uses "us.meta.llama3-3-70b-instruct-v1:0"
# API returns: 404 "ValidationException: Model not found"
# LLM adjusts: Try "us.meta.llama3-3-70b-instruct-turbo-v1:0"
# Eventually learns the correct pattern

# No web search needed - the feedback loop teaches it
```

### Decision 3: Extend Existing Infrastructure

**Choice:** Add `operator_decision` field to existing `model_status` collection, reuse `LifecycleStatus` enum

**Rationale:**
- Don't duplicate the lifecycle system with new collections (`operator_suggestions`)
- `model_status` already tracks per-model health
- `LifecycleStatus` enum already has the states we need
- Just add operator reasoning alongside deterministic classification

**Schema Addition:**
```javascript
// Existing fields in model_status collection:
{
  provider: "groq",
  model_id: "llama-3.1-70b-specdec",
  lifecycle_status: "likely_deprecated",  // From classifier.py (deterministic)
  // NEW: Operator reasoning (LLM-based)
  operator_decision: {
    action: "disable",
    confidence: 0.95,
    reasoning: "Model returns 404 'does not exist' for 48 hours. Groq likely removed it.",
    suggested_at: ISODate("2026-01-01T08:00:00Z"),
    suggested_by: "ai-operator-v1",
    status: "pending",  // pending|approved|rejected|auto_executed
    executed_at: null
  }
}
```

### Decision 4: Start Simple

**Choice:** Phase 1 is just cleanup - disable OpenRouter + broken models. LLM reasoning comes in Phase 2.

**Rationale:**
- Don't overcomplicate the first step
- Get immediate value from cleaning up obvious issues
- Build foundation for LLM reasoning without blocking on it

---

## Architecture

### Current System (Before AI Operator)

```
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT SYSTEM                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Scheduler (3hr) ──► Run benchmarks ──► MongoDB             │
│                                            │                 │
│                                            ▼                 │
│                                    collector.py              │
│                                    (gather signals)          │
│                                            │                 │
│                                            ▼                 │
│                                    classifier.py             │
│                                    (if/else rules)           │
│                                            │                 │
│                                            ▼                 │
│                                    LifecycleStatus           │
│                                    (active/failing/etc)      │
│                                            │                 │
│                                            ▼                 │
│                                    Daily health email        │
│                                    (human reviews)           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Phase 2+ System (With AI Operator)

```
┌─────────────────────────────────────────────────────────────┐
│                  AI OPERATOR SYSTEM                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Scheduler (3hr) ──► Run benchmarks ──► MongoDB             │
│                                            │                 │
│                                            ▼                 │
│                                    collector.py              │
│                                    (gather signals)          │
│                                            │                 │
│                         ┌──────────────────┴─────────────┐  │
│                         ▼                                ▼  │
│                    operator.py                    classifier.py │
│                    (LLM reasoning)               (fallback)   │
│                         │                                │  │
│                         └──────────────────┬─────────────┘  │
│                                            ▼                 │
│                                    model_status             │
│                                    (with operator_decision) │
│                                            │                 │
│                         ┌──────────────────┴─────────────┐  │
│                         ▼                                ▼  │
│                   Auto-execute                   Suggest    │
│                   (404/401 >48h)                (review)    │
│                         │                                │  │
│                         └──────────────────┬─────────────┘  │
│                                            ▼                 │
│                                    Daily email              │
│                                    (audit trail)            │
│                                            │                 │
│                                            ▼                 │
│                                    Human approval           │
│                                    (copy-paste commands)    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Module Structure

```
api/llm_bench/
├── model_lifecycle/              # EXISTING - keep as-is
│   ├── classifier.py             # Deterministic fallback (demoted)
│   ├── collector.py              # Signal gathering (unchanged)
│   └── cli.py                    # Existing CLI
│
├── operator/                     # NEW - AI reasoning
│   ├── engine.py                 # Core LLM decision logic
│   ├── io.py                     # Load signals from MongoDB
│   ├── actions.py                # Execute decisions (disable models, etc.)
│   └── cli.py                    # CLI: run operator, view suggestions
│
├── discovery/                    # NEW - model discovery (Phase 4)
│   ├── openrouter.py             # Fetch OpenRouter catalog
│   ├── matcher.py                # Match to direct providers
│   └── cli.py                    # CLI: fetch, report
│
├── ops/
│   └── llm_error_classifier.py   # EXISTING - error categorization
│
└── models_db.py                  # EXISTING - model registry interface
```

---

## Implementation Phases

### Phase 1: OpenRouter Cleanup + Broken Models (Immediate)

**Status:** ✅ COMPLETED (2026-01-01)

**Goal:** Remove noise and get clean baseline

**Acceptance Criteria:**
- [x] All OpenRouter models disabled in `models` collection (175 models disabled)
- [x] OpenRouter credentials kept (for future discovery API)
- [x] Known broken models disabled:
  - groq/llama-3.1-70b-specdec ✅
  - cerebras/llama-4-maverick-17b-128e-instruct ✅ (corrected model ID)
  - vertex/gemini-1.5-flash-002 ✅
  - openai/gpt-4.5-preview ✅
- [ ] Dashboard shows only direct provider data (pending next benchmark run, ~3hrs)
- [ ] Daily health email no longer shows OpenRouter errors (pending next email)

**Execution Results:**
- OpenRouter models disabled: 175 (spec estimated 171, actual was 175)
- Broken models disabled: 4 (cerebras model ID corrected to `-instruct` variant)
- Total enabled models: 222 (down from ~397)
- Total disabled models: 301
- Total models in collection: 523

**Manual Execution (MongoDB):**
```bash
# Disable all OpenRouter models
mongosh "mongodb://writer:...@5.161.97.53/llm-bench?authSource=llm-bench" --eval '
db.models.updateMany(
  { provider: "openrouter" },
  {
    $set: {
      enabled: false,
      disabled_reason: "Phase 1 cleanup: OpenRouter duplicates direct providers, adds latency. Use for discovery only.",
      disabled_at: new Date()
    }
  }
)'

# Disable known broken models
mongosh "mongodb://..." --eval '
[
  { provider: "groq", model_id: "llama-3.1-70b-specdec" },
  { provider: "cerebras", model_id: "llama-4-maverick-17b-128e" },
  { provider: "vertex", model_id: "gemini-1.5-flash-002" },
  { provider: "openai", model_id: "gpt-4.5-preview" }
].forEach(m => {
  db.models.updateOne(
    { provider: m.provider, model_id: m.model_id },
    {
      $set: {
        enabled: false,
        disabled_reason: "Phase 1 cleanup: Model returns 404, confirmed removed by provider",
        disabled_at: new Date()
      }
    }
  )
})
'
```

**Test Commands:**
```bash
# Verify OpenRouter models disabled
mongosh "mongodb://..." --quiet --eval '
db.models.countDocuments({provider: "openrouter", enabled: false})
'
# Expected: 171

# Check remaining enabled model count
mongosh "mongodb://..." --quiet --eval '
db.models.countDocuments({enabled: true})
'
# Expected: ~200 (down from ~370)

# Wait 3+ hours, check dashboard has no OpenRouter data
curl -s "https://llm-benchmarks.com/api/processed?days=1" | jq '.table[] | select(.providerCanonical=="openrouter")'
# Expected: empty (no results)
```

**Duration:** 30 minutes (mostly verification time)

---

### Phase 2: Core Operator Module with LLM Reasoning

**Goal:** Build operator engine that uses LLM to reason about model health

**Acceptance Criteria:**
- [ ] `api/llm_bench/operator/engine.py` exists
- [ ] `generate_decisions()` function calls LLM with signal context
- [ ] Decisions stored in `model_status.operator_decision` field
- [ ] CLI command: `uv run python -m api.llm_bench.operator.cli analyze` generates suggestions
- [ ] Output includes confidence scores and reasoning
- [ ] Falls back to `classifier.py` if LLM call fails

**Core Engine Function:**
```python
# api/llm_bench/operator/engine.py

from typing import List
import openai
from ..model_lifecycle.collector import LifecycleSnapshot

async def generate_decisions(
    snapshots: List[LifecycleSnapshot]
) -> List[OperatorDecision]:
    """
    Use LLM to reason about model health signals and suggest actions.

    Falls back to classifier.py if LLM unavailable.
    """
    decisions = []

    for snapshot in snapshots:
        # Prepare context for LLM
        context = format_snapshot_context(snapshot)

        # Call LLM
        try:
            decision = await call_llm_for_decision(context)
            decisions.append(decision)
        except Exception as e:
            # Fallback to deterministic classifier
            fallback_decision = classifier.classify_snapshot(snapshot)
            decisions.append(convert_to_operator_decision(fallback_decision))

    return decisions

async def call_llm_for_decision(context: str) -> OperatorDecision:
    """
    Call OpenAI to reason about model health.
    """
    prompt = f"""
You are analyzing LLM model health. Based on these signals:

{context}

Determine the appropriate action:
- **disable**: Model is permanently broken (404, deprecated)
- **monitor**: Concerning but not conclusive (high error rate, but might recover)
- **ignore**: Healthy or temporary issue

Provide:
1. Recommended action
2. Confidence (0.0-1.0)
3. Clear reasoning explaining your decision

Return JSON:
{{
  "action": "disable|monitor|ignore",
  "confidence": 0.95,
  "reasoning": "Model returns 404 'does not exist' for 48 hours straight. Provider likely removed it."
}}
"""

    response = await openai.ChatCompletion.create(
        model="gpt-4o-mini",  # Fast and cheap for this use case
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    return OperatorDecision(
        action=result["action"],
        confidence=result["confidence"],
        reasoning=result["reasoning"],
        suggested_at=datetime.utcnow(),
        suggested_by="ai-operator-v1"
    )
```

**Test Commands:**
```bash
# Run operator analysis (dry-run, no writes)
cd /Users/davidrose/git/llmbench/llm-benchmarks
uv run env PYTHONPATH=. python -m api.llm_bench.operator.cli analyze --dry-run

# Expected output:
# Analyzed 200 models
# Decisions:
#   - 3 disable (high confidence)
#   - 12 monitor (medium confidence)
#   - 185 ignore (healthy)
#
# High-confidence disable recommendations:
# 1. groq/llama-3.1-70b-specdec
#    Reasoning: Model returns 404 for 48h, error "does not exist"
#    Confidence: 0.95

# Run with writes to model_status
uv run env PYTHONPATH=. python -m api.llm_bench.operator.cli analyze --write

# Verify operator_decision field added
mongosh "mongodb://..." --quiet --eval '
db.model_status.findOne({
  provider: "groq",
  model_id: "llama-3.1-70b-specdec"
})
'
# Expected: Shows operator_decision with action="disable", reasoning, etc.
```

**Duration:** 2-3 days (building engine, testing LLM prompts, CLI)

---

### Phase 3: Integration with Daily Health Check

**Goal:** Operator runs automatically, sends suggestions in daily email

**Acceptance Criteria:**
- [ ] `ops/daily-health-check.py` calls operator engine
- [ ] Email includes "AI Operator Suggestions" section
- [ ] Auto-executes high-confidence disables (404/401 >48h)
- [ ] Provides copy-paste commands for human-approved actions
- [ ] Tracks executed actions in `model_status.operator_decision.executed_at`

**Email Format:**
```
Subject: [OPERATED] Daily Health Report - 2 auto-actions, 5 suggestions

=== AUTO-EXECUTED ACTIONS ===

Disabled 2 models (404 errors >48h):

1. groq/llama-3.1-70b-specdec
   Reason: Model returns 404 "does not exist" for 48 hours
   Confidence: 0.95
   Executed: 2026-01-01 08:00 UTC

2. cerebras/llama-4-maverick-17b-128e
   Reason: Model returns 404 for 72 hours, no successful runs
   Confidence: 0.98
   Executed: 2026-01-01 08:00 UTC

=== SUGGESTIONS FOR REVIEW ===

5 models flagged for monitoring:

1. fireworks/deepseek-v3
   Action: Update model ID to "deepseek-v3-0324"
   Reason: 100% error rate with "model not found", but other DeepSeek models work. Likely renamed.
   Confidence: 0.85

   Approve with:
   mongosh "mongodb://..." --eval 'db.models.updateOne(
     {provider: "fireworks", model_id: "deepseek-v3"},
     {$set: {model_id: "deepseek-v3-0324"}}
   )'

[...]
```

**Test Commands:**
```bash
# Run health check (triggers operator)
cd /Users/davidrose/git/llmbench/llm-benchmarks
uv run env PYTHONPATH=. python ops/daily-health-check.py

# Check for auto-executed actions
mongosh "mongodb://..." --quiet --eval '
db.model_status.find({
  "operator_decision.status": "auto_executed"
}).forEach(doc => {
  print(doc.provider + "/" + doc.model_id + " → " + doc.operator_decision.reasoning)
})
'

# Verify email sent (check logs or inbox)
```

**Duration:** 1-2 days (integration, email formatting, testing)

---

### Phase 4: Discovery System (OpenRouter as Catalog)

**Goal:** Use OpenRouter as discovery layer to find new models

**Acceptance Criteria:**
- [ ] Daily fetch of OpenRouter `/api/v1/models` (free, no auth)
- [ ] Store in `openrouter_catalog` collection
- [ ] LLM matches OpenRouter models to direct providers
- [ ] Suggests new models to add with copy-paste commands
- [ ] Tracks confidence of matches

**OpenRouter API:**
```bash
# Free API, no auth required
curl -s "https://openrouter.ai/api/v1/models" | jq '.data | length'
# Returns: 353 models with full metadata
```

**Discovery Flow:**
```python
# 1. Fetch OpenRouter catalog daily
openrouter_models = fetch_openrouter_models()

# 2. Compare to our models collection
our_models = db.models.find({enabled: True})

# 3. Find new models (in OpenRouter, not in our DB)
new_models = [m for m in openrouter_models if not in_our_db(m)]

# 4. LLM matches to direct providers
for openrouter_model in new_models:
    prompt = f"""
    OpenRouter has: {openrouter_model['id']}
    Name: {openrouter_model['name']}

    Which direct provider offers this model?
    Guess the model ID they use (you can be wrong, API will correct).

    Return JSON: {{"provider": "anthropic|openai|bedrock|...", "model_id": "...", "confidence": 0.8}}
    """

    match = await llm_call(prompt)

    # 5. Suggest to human via email
    suggestions.append({
        "openrouter_id": openrouter_model['id'],
        "provider": match['provider'],
        "model_id": match['model_id'],
        "confidence": match['confidence']
    })
```

**Test Commands:**
```bash
# Fetch OpenRouter catalog
cd /Users/davidrose/git/llmbench/llm-benchmarks
uv run env PYTHONPATH=. python -m api.llm_bench.discovery.cli fetch

# See new models discovered
uv run env PYTHONPATH=. python -m api.llm_bench.discovery.cli report

# Expected output:
# Found 12 new models on OpenRouter:
#
# 1. x-ai/grok-2-1212 → grok/grok-2-1212 (confidence: 0.90)
# 2. anthropic/claude-opus-4-20251212 → anthropic/claude-opus-4-20251212-v1:0 (confidence: 0.95)
# [...]

# Check openrouter_catalog collection
mongosh "mongodb://..." --quiet --eval '
db.openrouter_catalog.countDocuments()
'
# Expected: ~353 models
```

**Duration:** 2-3 days (API integration, matching logic, reporting)

---

## Test Commands

### Phase 1 Verification

```bash
# Count disabled OpenRouter models
mongosh "mongodb://writer:***REMOVED***@5.161.97.53/llm-bench?authSource=llm-bench" --quiet --eval '
db.models.countDocuments({provider: "openrouter", enabled: false})
'

# Count total enabled models (should drop ~170)
mongosh "mongodb://..." --quiet --eval '
db.models.countDocuments({enabled: true})
'

# Wait 3+ hours for next benchmark run, verify no OpenRouter in dashboard
curl -s "https://llm-benchmarks.com/api/processed?days=1" | \
  jq '.table[] | select(.providerCanonical=="openrouter") | .provider + "/" + .model_name'
# Expected: empty (no output)
```

### Phase 2 Verification

```bash
# Run operator in dry-run mode
cd /Users/davidrose/git/llmbench/llm-benchmarks
uv run env PYTHONPATH=. python -m api.llm_bench.operator.cli analyze --dry-run

# Check operator decisions written to DB
mongosh "mongodb://..." --quiet --eval '
db.model_status.countDocuments({"operator_decision": {$exists: true}})
'

# View a specific decision
mongosh "mongodb://..." --quiet --eval '
db.model_status.findOne(
  {provider: "groq", model_id: "llama-3.1-70b-specdec"},
  {operator_decision: 1, lifecycle_status: 1}
)
'
```

### Phase 3 Verification

```bash
# Run health check
uv run env PYTHONPATH=. python ops/daily-health-check.py

# Check for auto-executed actions
mongosh "mongodb://..." --quiet --eval '
db.model_status.find({"operator_decision.status": "auto_executed"}).count()
'

# View auto-executed actions
mongosh "mongodb://..." --quiet --eval '
db.model_status.find(
  {"operator_decision.status": "auto_executed"},
  {provider: 1, model_id: 1, "operator_decision.reasoning": 1}
)
'
```

### Phase 4 Verification

```bash
# Fetch OpenRouter catalog
uv run env PYTHONPATH=. python -m api.llm_bench.discovery.cli fetch

# Count OpenRouter models
mongosh "mongodb://..." --quiet --eval '
db.openrouter_catalog.countDocuments()
'

# View discovery report
uv run env PYTHONPATH=. python -m api.llm_bench.discovery.cli report

# Check for new model suggestions
mongosh "mongodb://..." --quiet --eval '
db.openrouter_catalog.find({matched_provider: {$exists: true}}).limit(5)
'
```

---

## Key Technical Details

### MongoDB Schema Changes

**Existing `models` collection:**
```javascript
{
  provider: "groq",
  model_id: "llama-3.1-70b-specdec",
  enabled: false,  // Phase 1 sets this
  disabled_reason: "Phase 1 cleanup: Model returns 404",
  disabled_at: ISODate("2026-01-01")
}
```

**New field in `model_status` collection (Phase 2):**
```javascript
{
  provider: "groq",
  model_id: "llama-3.1-70b-specdec",
  lifecycle_status: "likely_deprecated",  // From classifier.py (deterministic)

  // NEW: Operator decision (LLM reasoning)
  operator_decision: {
    action: "disable",
    confidence: 0.95,
    reasoning: "Model returns 404 'does not exist' for 48 hours. Groq likely removed it.",
    suggested_at: ISODate("2026-01-01T08:00:00Z"),
    suggested_by: "ai-operator-v1",
    status: "auto_executed",  // pending|approved|rejected|auto_executed
    executed_at: ISODate("2026-01-01T08:05:00Z")
  }
}
```

**New collection for OpenRouter discovery (Phase 4):**
```javascript
// openrouter_catalog
{
  openrouter_id: "anthropic/claude-opus-4",
  name: "Anthropic: Claude Opus 4",
  org: "anthropic",
  pricing: { prompt: 0.000015, completion: 0.000075 },
  context_length: 200000,
  created: ISODate("2025-06-15"),
  first_seen_at: ISODate("2026-01-01"),
  last_seen_at: ISODate("2026-01-01"),

  // LLM matching results
  matched_provider: "anthropic",
  matched_model_id: "claude-opus-4-20250514",
  match_confidence: 0.95,
  match_reasoning: "Direct match - Anthropic model available via their API"
}
```

### Fallback Strategy

If LLM calls fail (rate limits, API down, etc.), the system falls back to existing `classifier.py` logic:

```python
# In operator/engine.py

async def generate_decisions(snapshots):
    decisions = []
    llm_failures = 0

    for snapshot in snapshots:
        try:
            # Try LLM reasoning first
            decision = await call_llm_for_decision(snapshot)
            decisions.append(decision)
        except Exception as e:
            # Fallback to deterministic classifier
            logger.warning(f"LLM call failed for {snapshot.model_id}, using fallback: {e}")
            llm_failures += 1

            fallback_decision = classifier.classify_snapshot(snapshot)
            operator_decision = convert_to_operator_decision(
                fallback_decision,
                suggested_by="classifier-fallback"
            )
            decisions.append(operator_decision)

    if llm_failures > 0:
        logger.info(f"Used fallback classifier for {llm_failures}/{len(snapshots)} models")

    return decisions
```

### Auto-Execution Rules

Only these conditions trigger auto-execution (no human approval needed):

1. **404 errors for 48+ hours** with error message containing "not found" or "does not exist"
2. **401 errors for 48+ hours** with auth-related messages
3. **LLM confidence ≥ 0.95** for disable action

Everything else goes to human review via email.

---

## FAQ

### Why demote classifier.py instead of removing it?

- It's a working fallback if LLM calls fail
- Useful for comparison ("What would the rules say vs what does LLM say?")
- Already integrated into existing code

### Why not use web search for model ID matching?

- Adds complexity (web search API, parsing results)
- The API feedback loop is simpler: LLM guesses → API errors → LLM adjusts
- If wrong, the error messages are clear: "Model not found: xyz"

### Why OpenRouter for discovery if we disable it for benchmarks?

- OpenRouter aggregates 350+ models from all providers
- Their catalog is always up-to-date
- API is free and doesn't require auth
- It's a discovery layer, not a benchmark target

### What happens to existing deprecation snapshots?

- Keep them! They're historical baselines for the dashboard
- Operator decisions supplement (not replace) the existing system
- Dashboard continues using `deprecation_snapshots` for grey baseline charts

---

## Document History

| Date | Author | Changes |
|------|--------|---------|
| 2026-01-01 | David Rose + Claude | Initial spec from design doc, incorporating review feedback |

---

**Next Steps:**
1. Review and approve this spec
2. Execute Phase 1 (manual MongoDB commands)
3. Begin Phase 2 implementation (operator engine)
