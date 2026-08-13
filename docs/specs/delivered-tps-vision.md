# Delivered TPS — The Single-Metric Product Vision

**Status:** approved direction (2026-08-12). Step 2 shipped: the runner records
`time_to_64_visible_tokens_seconds` on streaming lanes (reasoning deltas never
advance the clock), `/api/delivered-tps` derives the median per model, and the
leaderboard renders above the charts on `/cloud`. Reasoning-model publication
(2048-budget rows) is step 3, not yet live.
**Supersedes as product vision:** the two-metric (steady-state + floor latency)
leaderboard proposal and any visible-vs-total split on the leaderboard.
**Build order and owners-only decisions at the bottom.**

## Product thesis

The site answers one question honestly:

> **How quickly does this model deliver visible output, including the time it
> spends thinking?**

It measures live provider APIs on a schedule and publishes measured numbers,
not vendor claims. The audience is a developer choosing a model to route
through OpenRouter, or comparing the few direct providers they already consume.
Every row says exactly what was served, by whom, and how it was measured.

## The one number: Delivered TPS

```
delivered_tok/s = 64 visible answer tokens / (time from request start to the 64th visible token)
```

- Reasoning tokens count as **time**, never as **output**.
- The run uses a generous completion budget (see profiles) so reasoning models
  actually reach visible text; measurement ends at the 64th visible token, not
  at budget exhaustion.
- One scalar per model. Chat and reasoning models perform the same visible
  task, so Delivered TPS is comparable across model classes — no special
  casing, which matters in a world where reasoning is becoming the default.

Rationale (why not the alternatives):
- Total-token TPS (generated incl. reasoning / time) measures provider compute
  activity, not useful output — an invisible-token inflation problem.
- Visible-token TPS alone is unmeasurable at a 64-token budget for thinking
  models and collapses all models onto the same small range when reasoning is
  the norm.
- Time-to-answer is correct in spirit but requires judging answer completeness
  — subjective for a throughput benchmark. Delivered TPS keeps tok/s
  intuition while folding thinking delay into the time side, where it belongs.

## Measurement protocol

- Fixed prompt family (the site's long-form story prompt), fixed reasoning
  settings, generous completion budget:
  - chat models: 512 tokens
  - reasoning-class models: 2048 tokens
- The runner already records per-token timing (`times_between_tokens`, TTFT,
  generate_time) — Delivered TPS is derived from existing long-run
  instrumentation; no new architecture.
- The steady-state estimator (Theil-Sen slope + intercept, bootstrap ±15% CI)
  still powers the detail page; it does not drive the leaderboard headline.
- The legacy 64-token series remains, frozen as a secondary "burst / short
  answer" number on the detail page. History is never deleted.
- Per-row `profile` tag so the measurement budget is visible.

## Leaderboard (the page)

- One ranked list. Each row: rank, model + provider, **Delivered TPS**, a
  restrained freshness indicator.
- Sort descending. One rounded value (`18.4 tok/s`). No burst/steady columns,
  no visible/total split, no confidence interval, no reasoning badge, no
  latency column. "Reasoning model" is not a badge when reasoning is the
  default.
- Provenance is a muted second line, not a column:
  ```
  Claude 4 · Anthropic
  served via OpenRouter by Anthropic
  ```
- Visual grammar: quiet. One strong numeric column, one accent color for
  speed, thin proportional bars, muted metadata, generous spacing. Color only
  for freshness or measurement problems, never model categories. The page
  reads as a ranked list, not an observability dashboard.

## Model detail page

Holds everything the leaderboard refuses: Delivered TPS history, time to first
visible token, visible and reasoning token counts, total generation time,
statistical confidence + sample count, prompt / reasoning setting /
methodology, and route + serving-provider provenance (endpoint, route history,
upstream rotation).

## Routing and migration policy (2026-08-12)

Routing is provenance, not a leaderboard dimension.

- **Direct lanes** (real numbers for providers consumed directly): openai,
  vertex/GCP, bedrock/AWS. Enforced at resolution time
  (`DIRECT_PROVIDERS` in `scheduler/routing.py`) — the runner refuses to route
  these even with a stale route document.
- **Or-served lanes** (every other provider: deepinfra, together, fireworks,
  groq, cerebras, anthropic): routed through OpenRouter with **no provider
  pinning**; the observed upstream is read from OR response metadata and
  becomes part of the row (`route_policy: "or-served"`).
- Never combine measurements from different serving providers into one result.
  Switch the canonical route explicitly and preserve the transition in history.
- Direct vs or-served live side by side in the same leaderboard; the
  difference is the muted provenance line only.

## Catalog policy

A row deserves to exist iff it is popular or comparable, measurable through a
lane we actually call, and ideally an OpenRouter model. Target ~100 enabled
models (from 225):

- keep the OR-visible popular set (~88 routing-planned, minus overlap)
- keep direct-lane frontier models actually consumed (openai/vertex/bedrock)
- drop ~90 deepinfra long tail (noise + billing fragmentation)
- drop ~24 not-on-OpenRouter models (Llama-2/3 base, edge Qwen, OCR, old
  checkpoints) + 10 proxy resell duplicates

Removal = `disabled` flag with logged reason; rows are never hard-deleted.
Admission enforces the rule automatically (OR id or approved direct-lane
reason + valid recent measurement + not a proxy duplicate).

## Invariants / alerting

An invariant must check the correct profile for the model; known-accepted
states are classified, never paged.

- Every model gets an `expected_profile` (chat: 512 / reasoning: 2048). The
  measurement invariant evaluates against it — the 19 "unmeasurable" models
  become measurable, not exemptions.
- Three tiers:
  - **PAGE:** liveness watchdog; both queue invariants red; site-wide zero
    rows for >24h.
  - **WARN (log, no page):** single model stale; or-served lane regresses vs
    its own history; canary failures; sustained shared-pool 429s.
  - **SUPPRESSED:** removal-pending models; reasoning models on their own
    profile; transient throttling; mid-canary routes.

## Unavoidable tradeoffs

1. Delivered TPS intentionally penalizes reasoning time. Correct for
   experienced speed; it is not a pure decoder-performance number.
2. The result is workload- and reasoning-setting-specific. The site states its
   fixed profile and resists pretending one number predicts every task.

## Build order (dependencies, reversibility)

1. Estimator as published metric infrastructure (collecting already).
2. **Delivered TPS headline** + detail-page split (keystone — reversible flag).
3. Invariant tiers + per-model `expected_profile` (kills the paging).
4. Finish routing: promote canary-passed routes, finish the ~88 planned.
5. Catalog cleanup: drop long tail + not-on-OR + duplicates (reversible flag).
6. Retire non-consumed direct keys **only after** 14 days of green routed
   lanes (the single low-reversibility commit point).
7. UX rebuild: single leaderboard, provenance line, detail pages.

## Owner decisions (taken 2026-08-12)

1. Headline = Delivered TPS (not total tok/s). **Decided.**
2. Estimator long-run profile becomes the default; 64-token demoted to detail
   page. **Decided (pending build).**
3. Keep openai / vertex / bedrock direct. **Decided.**

## In-flight state (as of 2026-08-12)

- 24 routes live (anthropic 6, deepinfra 17, together 1); 33 more canaried
  (wave 2: 0/33 passed — taxonomy: 13 allowlist-gap lanes fixed in 41cf542,
  15 reasoning-at-64-token class now addressed by the 2048 profile + Delivered
  TPS, 4 timeouts need longer canary deadline, 1 near-miss at 0.92).
- Queue-invariant leaks fixed (admission probes + long-profile samples for
  bedrock; swept; green).
- Wave evidence + decisions: `/private/tmp/openrouter-v4/v5_*` (mirrored to
  cube-artifacts `openrouter-consolidation/v5/`).