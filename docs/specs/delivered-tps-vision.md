# Delivered TPS — The Single-Metric Product Vision

**Status:** approved direction (2026-08-12), **amended 2026-08-18** — see
"Amendment" below. The runner records `time_to_64_visible_tokens_seconds` on
streaming lanes (reasoning deltas never advance the clock), `/api/delivered-tps`
derives the estimate **per endpoint** and the leaderboard renders above the
charts on `/cloud`. Reasoning-model publication (2048-budget rows) is step 3,
not yet live.

> **Two claims below are superseded and left in place for the record:** "one
> scalar per model" (the unit is now the *endpoint*) and "no confidence
> interval" (official rows publish one). The Amendment section is authoritative
> where the two disagree.
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
- One scalar per **endpoint** (amended 2026-08-18; originally "per model").
  Chat and reasoning models perform the same visible task, so Delivered TPS is
  comparable across model classes — no special casing, which matters in a world
  where reasoning is becoming the default. It is not comparable across
  *deployments* of one model: `gpt-oss-120b` is served at fp4 and at bf16 by
  different providers on different hardware, and averaging those reports a
  speed nobody serves at.

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
- Ranked by **tier**, not by value (amended 2026-08-18). One rounded value
  (`18.4 tok/s`) plus the 95% interval on official rows. No burst/steady
  columns, no visible/total split, no reasoning badge, no latency column.
  "Reasoning model" is not a badge when reasoning is the default.
  - The original "sort descending, no confidence interval" is superseded. With
    ~91% of the fleet's legacy timings batched and a 3h sampling cadence, an
    ordering without an interval is a claim the measurement cannot support.
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
---

## Amendment (2026-08-18) — publication policy

Authoritative where it disagrees with anything above. Decided by
`hatch codex sol` at `xhigh` reasoning; implemented in
`utils/endpointPublication.ts` and pinned by `tests/endpointPublication.test.js`.

### The unit is the endpoint

An OpenRouter endpoint is a (model, provider-deployment) pair — `novita/fp8`,
`deepinfra/bf16`. 693 of them across 246 models and 61 providers. Deployments of
one model differ in quantization and hardware and are not interchangeable, so
quantization splits the ranking axis: fp4 never ranks against bf16.

### Three publication states

| State | Gate | Shown | Ranked |
|---|---|---|---|
| `insufficient` | < 8 usable samples | nothing | no |
| `preliminary` | 8 samples, 24h span, 2 UTC dates, 4 of 6 blocks (~1 day) | figure only | **no** |
| `official` | 30 samples, 96h span, 5 UTC dates, all 6 blocks (~4 days) | figure + 95% interval | yes |

Re-evaluated on every refresh over a rolling 7-day window. An endpoint that
stops qualifying loses its rank rather than keeping a stale one.

Why not a floor of 5: an endpoint is measured on a 3h cadence, so a small
sample is dominated by whichever few hours it happened to land in. Five samples
ranks scheduling phase.

### Two rules that carry most of it

**Deduplicate to one sample per 30-minute bucket.** A scheduler re-running one
endpoint in a tight loop must not be able to buy significance. Not
hypothetical: on the night endpoint scheduling shipped, a health-identity bug
put one endpoint at 348 samples inside a single 3-hour window.

**Bootstrap over whole UTC dates, not runs.** Runs within a day share load,
routing and time of day. Resampling them individually treats correlated
observations as independent and reports a precision the sampling design cannot
support. 10,000 replicates, seed derived from endpoint identity so a refresh
does not reshuffle the interval.

### The estimate is `64 / median(T64)`

Not the median of per-run rates. Per-run rates are a constant over a
right-skewed time, so their median exaggerates fast runs. Take the median on
the timing scale and convert once. The original implementation had the other
one.

### Ranking by non-overlap only

A is faster than B only when A's whole interval sits above B's. Anything
transitively connected by overlap forms one tier, shares a rank, is listed
alphabetically, and is labelled **order unresolved** — not *equal*. The claim
is that this measurement cannot tell them apart.

These are conservative tiers, not a proven global ordering: individual 95%
intervals give no family-wise guarantee across hundreds of comparisons.

`unknown` quantization is not ranked at all. It is missing metadata rather than
a coherent class, and it covers 310 of 693 endpoints including Groq — ranking
it would silently compare an fp4 deployment against a bf16 one. **This narrows
the earlier owner decision** (2026-08-17: "`unknown` never merges with a known
value"), which put unknown on its own axis rather than excluding it. Reversible
in one line in `rankEndpoints`.

### Availability is a separate property

Timeouts, 429s, 5xx and completions shorter than 64 visible tokens are counted
as outcomes and never enter the speed estimate. Folding them in lets an
endpoint that fails most requests look fast on the few it serves.
`/api/endpoint-availability`.

### OpenRouter's own telemetry

Their `/endpoints` API returns p50/p75/p90/p99 throughput over the trailing 30
minutes of their real traffic. It is **never** a prior, a fallback for
endpoints below threshold, a sample filter, or a ranking input: it aggregates
uncontrolled workloads with varying prompt and output lengths, and mixing it
with a controlled 64-token measurement produces a number that means neither.
It may appear on an endpoint detail page under an explicit external-telemetry
label. `uptime_1d` and status belong in the availability section.

### Legacy `tokens_per_second`

Frozen, retained, never blended, never deleted. Stamped `legacy_sse_window`
with `rank_eligible=false` (`utils/legacyMetric.ts`). It is timed from batched
SSE deltas — 4440 batched against 413 resolved on pinned rows — so it cannot
support a comparison between providers. The charts keep the history and now
disclose what it is; no surface orders anything by it.

### Naming

**"Delivered TPS (64-token, end-to-end)."** The site should not describe itself
as measuring decoder or generation throughput: at high speed this is
predominantly startup latency, and it measures client-observed delivery for one
specific workload. TTFT may be shown as diagnostic metadata and never combined
into another ranking.

### Where it renders (owner decision, 2026-08-18)

David removed the ranked leaderboard from `/cloud` (`deb3b83`). The page keeps
the throughput distribution as its primary visual comparison; Delivered TPS is
served from `/api/delivered-tps` for consumers that need an end-to-end ranking,
with publication state, interval and tier on every row.

This **narrows sol's "sole published headline" recommendation**: the ranking
policy above governs the metric wherever it is consumed, but the site does not
currently present a ranked list. The gate is what makes that safe either way —
an endpoint below threshold has no publishable number regardless of which
surface asks for it.

**Update 2026-09-02.** David handed product ownership to the agent, and the
ranked list is back on `/cloud`, above the charts, in three explicit states:
official rows ranked by tier with intervals; preliminary rows alphabetical with
a figure and no rank; and, when nothing qualifies, one line stating the count
under measurement and the gate. On that date 0 of 763 pinned endpoints had
reached even `preliminary`, because most endpoints receive about one sample a
week — the sampling policy, not the metric, is what decides whether this list
ever fills.
