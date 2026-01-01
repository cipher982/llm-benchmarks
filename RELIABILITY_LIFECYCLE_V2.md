# Reliability & Lifecycle v2 (Spec + Review-Integrated)

Status: implement-now (phase 1)

This document captures the agreed spec for improving the benchmarking system’s reliability, lifecycle decisioning, and operational signal quality. It incorporates the review feedback and clarifies edge-case semantics before implementation.

## Goals
- Reduce error spam and convert failures into actionable signals.
- Prevent bad API keys / billing issues from causing mass false deprecation/disable decisions.
- Provide conservative, evidence-based recommendations for disabling models (initially **recommendations-only**, no auto-mutation).
- Preserve benchmark integrity: benchmarking remains the primary purpose.

## Non-goals
- Perfect “deprecated” truth (confidence + evidence, not certainty).
- Web scraping as a primary catalog source.

---

## Decisioning: Logging Only (No Actions)

Per core principle: **do not take automated actions in the runner** (no pausing, no disabling, no mutating the catalog). We only log structured signals and produce recommendations that a human can apply separately.

### Model-level decisioning (conservative; recommendations-only)

**Key rule:** `hard_capability` should **not** trigger auto-disable (these are usually “needs code update” issues).

**Fields (in `models`)**
- existing: `provider`, `model_id`, `enabled`, `deprecated`, `created_at`
- add (as-needed): `auto_disabled`, `disabled_at`, `disabled_reason`, `re_enabled_at`, `re_enable_reason`

**Disable recommendations (initially)**
- Start recommendations-only for 2–4 weeks.
- Optional future auto-disable only for:
  - `error_kind = hard_model`
  - model had at least 1 success historically
  - provider NOT paused
  - 5+ consecutive failures across 3+ cycles
- Keep `hard_capability` and “never succeeded” cases recommendations-only permanently.

---

## Error Taxonomy (write-time classification)

Classify errors at write-time and store on each `errors_cloud` document.

**`error_kind`**
- `auth` (401/403; invalid key; unauthorized)
- `billing` (402; insufficient credits; overdue invoices)
- `rate_limit` (429; quota exceeded)
- `hard_model` (404 model not found / “no endpoints found” / “does not exist”)
- `hard_capability` (“not a chat model”, “responses-only”, wrong endpoint)
- `transient_provider` (5xx)
- `network` (timeouts, connection errors)
- `unknown`

**Stored fields**
- `http_status`, `provider_error_code`
- `normalized_message`
- `fingerprint` (sha256 of provider/model/stage/error_kind/normalized_message)

**Error sampling**
- Keep last 3 unique normalized messages per `(provider, model, error_kind)` in rollups.

---

## Rollups / Dedupe

Maintain rollups to avoid 200k identical error docs becoming the primary UI signal.

**Collection: `error_rollups`**
- key: `fingerprint` (unique)
- `provider`, `model_name`, `stage`, `error_kind`
- `first_seen`, `last_seen`, `count`
- `sample_messages` (up to 3 unique normalized messages)

---

## Streak semantics

Any success resets all failure streaks for that model (conceptual; used for recommendations).

**Defaults**
- “Had recent success” lookback: 90 days
- Hard-fail threshold for recommendation: 5 failures
- “3+ cycles”: failures span at least ~2 hours (implementation approximation)

---

## Catalog snapshots (future; non-optional for authoritative providers)

Authoritative providers: OpenAI, Anthropic, Vertex, Bedrock.

Unreliable catalogs: OpenRouter, Together, Fireworks, Groq (weak signal only).

**Collection: `provider_catalog_snapshots`**
- `provider`, `collected_at`, `model_ids`, `source`

---

## Rollout Plan

1) `error_kind` + normalization + rollups (no behavior change besides extra fields).
2) Recommendations-only CLI for disables/investigation queues.
3) Optional: authoritative catalog snapshots and feed lifecycle confidence.
