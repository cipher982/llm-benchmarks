# LLM Bench platform plan

Status: draft for review · 2026-08-04

Consolidates the outstanding work across `llm-benchmarks` (runner),
`llm-benchmarks-dashboard` (site), `bench-ingest`, and the Sauron jobs repo.

## Where we are

On 2026-08-04 the collection pipeline was recovered from a long degradation.
The relevant facts, because they justify most of what follows:

- The runner container ran 2026-07-21 to 07-29 in state `Up` producing nothing.
  Worker threads had died on an unhandled Mongo error; the process never
  exited, so `restart: unless-stopped` never fired, and the container carried no
  healthcheck. Only a host reboot cleared it.
- `bench_jobs` had 774 permanently dead-lettered jobs and no resurrection path.
  `max_attempts` was 2, so every transient failure — rate limit, timeout,
  provider 500, overload — permanently removed a model from the site. The
  decline was a ratchet, not an incident.
- 530 models were enabled; 62 had run in 24h. DeepInfra had been dead on a
  billing 402 since 2026-06-10. Together had 269 enabled models of which 2 ran.
- `llm-bench-health` and `llm-bench-provider-discovery` were both registered
  `enabled=False`. Nothing alerted, and `provider_catalog` had not been written
  since 2026-04-29 while still appearing to be a live data source.

Fixed and deployed: worker fault isolation, a liveness watchdog plus container
healthcheck, retry classification with exponential backoff, a dead-letter sweep,
per-provider concurrency raised, catalogue quarantine applied, both Sauron jobs
re-enabled, coverage alerting live. Coverage went from 11.7% to 93%.

What remains is the reason it degraded in the first place: **nothing keeps our
picture of the world in sync with the world.** That was done by hand, by
opening agent sessions, and it silently stopped.

## The unifying idea

Three problems that look separate are the same problem:

1. Which models does a provider actually serve right now?
2. Which of those are benchmarkable text models?
3. Which provider-specific IDs refer to the same underlying model?

All three are "reconcile our catalogue against reality, continuously." They
should be one nightly system, not three manual processes. Workstream A below is
that system. B, C and D are independent.

---

## Workstream A — catalogue reconciliation

### A1. Why declaration fails

The existing `llm-bench-provider-discovery` already did "read `/models`, add
what's new." That is how Together reached 269 enabled models with 2 working.
Verified against live APIs on 2026-08-04:

| Provider | `/models` says | Reality |
|---|---|---|
| together | `type: "chat"`, normal pricing, `running: false` for **all 274** | dedicated-endpoint-only models are indistinguishable from serverless ones; they return `400 Unable to access non-serverless model` |
| fireworks | lists the model | `404 not found, inaccessible, and/or not deployed` |
| cerebras | lists the model | `404 does not exist or you do not have access` |
| groq | lists the model | `400 requires terms acceptance` (TTS models) |
| openai | lists the model | `dall-e`, `sora`, `realtime`, `transcribe`, `davinci-002` |

No field predicts serverless text availability. I looked specifically and did
not find one. Filtering by name pattern also failed: two passes of regex
cleanup still missed `veo`, `kling`, `vidu`, `ideogram`, `parakeet`,
`happyhorse`, `pixverse`.

### A2. Probe before promote

A candidate is admitted because **it passed a real benchmark call**, not
because an API listed it. This subsumes modality classification entirely: an
image or TTS model cannot return text tokens at a measurable rate, so it fails
admission on its own. There is no brand list to maintain.

```
00:00  refresh provider_catalog from every provider API          (exists today)
       ↓
       diff catalogue against models collection
       ↓
NEW    → insert enabled:false, status:"probing" → enqueue N probe jobs
         N successes  → assign identity (A3) → enabled:true
         terminal     → enabled:false, disabled_reason:<observed error>
         transient    → retry tomorrow
       ↓
GONE   → absent from provider API for 3 consecutive days → deprecated:true
       ↓
STALE  → enabled, zero successes in 7d, terminal errors → demote
       ↓
       weekly digest of promotions, demotions, deprecations
```

Hysteresis on deprecation matters: absent for one poll is a blip, absent for
three days is a retirement. Without it the catalogue flaps.

Most pieces exist: `provider_catalog`, the error taxonomy
(`hard_model`/`billing`/`auth`/`transient_provider`), `catalog_quarantine` for
demotion, `bench_model_health` for freshness. The missing half is promotion and
the reconciler that joins them.

Open question for review: probe cost. Roughly 1,000 candidates × N calls
one-off, then near-zero at steady state since only new models are probed. Worth
bounding before building.

### A3. Model identity — LLM normalizes, code groups

`llm-benchmarks-dashboard/backend/utils/modelMapping.ts` is a 377-line
hand-maintained table mapping provider IDs to display names. It is the same
recurring-work problem as the catalogue, and it is already wrong in at least one
place: `meta-llama/Meta-Llama-3-8B` (base) and `Meta-Llama-3-8B-Instruct` both
map to `llama-3-8b`. Those are different models.

The chart groups provider lines under one model name, so identity assignment
decides what the site claims. The governing asymmetry:

> **A false merge is far worse than a missed merge.** Grouping Together's FP8
> Turbo deployment with Bedrock's BF16 reports a provider speed difference that
> is actually a quantization difference, silently. A missed merge shows two
> lines instead of one: visible and self-correcting.

Bias toward not merging under uncertainty, and make uncertainty visible.

**Rejected shape:** "here are 300 model IDs, group them." Non-idempotent — one
new model reshuffles existing groups. Unreviewable. Unstable across runs.

**Proposed shape:** one LLM call per *unseen* model ID, returning structured
attributes; grouping is then deterministic code over those attributes.

```json
{
  "developer": "meta", "family": "llama", "version": "3.1", "params": "8B",
  "variant": "instruct",
  "quantization": "fp8",
  "context_variant": null,
  "serving_optimization": "turbo",
  "confidence": 0.95,
  "reasoning": "..."
}
```

```
canonical = f(developer, family, version, params, variant)
```

`quantization` and `serving_optimization` are deliberately excluded from the key
and carried as display annotations. Properties this buys:

| Property | Effect |
|---|---|
| Idempotent | each model normalized independently; new models cannot perturb existing groups |
| Cacheable | only unseen IDs need a call — roughly 20/day, not 300 |
| Reviewable | every decision is a row with confidence and reasoning |
| Policy in code | "should Turbo group with base?" is a change in `f()`, with no LLM re-run |

The separation is the point: the *judgment* (what is this string?) goes to the
model, where fuzzy world knowledge belongs; the *policy* (what counts as the
same model?) stays in code where it can be versioned and tested.

Guardrails:

- Confidence threshold routes to auto-apply or review. `pages/admin/model-review.tsx`
  and `utils/modelReview/` already exist.
- Never auto-merge into a group with an established time series; that rewrites
  published history.
- Flag groups whose members show persistently non-overlapping throughput
  distributions. Not proof of a bad merge, but a cheap detector.
- Store structured output plus prompt and model version so results can be
  re-derived when the prompt improves.

**Validation before trust:** the existing 377-line table is hand-built ground
truth. Run the normalizer across it and check it reproduces the mappings. Every
disagreement is an LLM error or a latent table bug. Do this before wiring
identity into promotion.

Provider routing: OpenRouter per the global agent config.

### A4. Chart display

Legend currently reads `bedrock / groq`. It should read `bedrock (bf16) /
groq (fp8)` where variants differ. On the llama-3.1-8b panel groq runs ~255
tok/s against bedrock ~100. Some of that is genuinely groq's hardware, which is
the interesting story; if part is quantization, flattening both to a bare
provider name makes the chart misleading.

---

## Workstream B — measurement semantics

`docs/reasoning-token-budget-spike.md`, branch `spike/reasoning-token-budget`,
is a completed proposal awaiting a decision. Summary: versioned benchmark
profiles, answer-yield separated from generated-work metrics, and reasoning
budget exhaustion recorded as a measurement outcome rather than an error.

It surfaced a problem beyond its brief: **the published time series is already
protocol-contaminated.** Of 1,182 Together rows in 30 days, 975 came from
multi-attempt retries, 860 had a final output cap different from the nominal 64
tokens, and 119 were silently labeled reasoning-disabled. The runner retries
with escalating budgets and publishes only the final successful attempt,
discarding earlier durations — biasing distributions toward requests that
eventually produced text.

Related and currently unfixed: 4 Qwen models are disabled because the runner
issues a non-streaming request and those models require `stream=true`. Adding a
streaming path touches the same accounting code, so it should be sequenced with
whatever B decides rather than patched separately.

This workstream needs a decision from David before implementation. It also
gates any schema change the reconciler would write.

---

## Workstream C — dashboard redesign

`llm-benchmarks-dashboard/backend/docs/redesign-epic.md`. Direction chosen
(Console), Phase 1 complete. Phases 2–5 not started: token and primitive layer,
chart rework, page-by-page port, verification. Independent of A and B except
that A3 changes what the chart legend can show and B may change which metrics
are published.

Phase 0 gap still open: `/providers/[provider]` and `/models/[provider]/[model]`
return 500 locally because they query MongoDB with no fixture fallback, so two
of five page templates cannot be iterated offline.

---

## Workstream D — cleanup

Verified during the 2026-08-04 recovery:

| Item | Evidence |
|---|---|
| Case-insensitive unique index on `models` | index is unique but case-sensitive; 28 duplicate groups, 56 docs. `together/qwen/Qwen2.5-7B-Instruct-Turbo` and `.../qwen2.5-7b-instruct-turbo` are **both enabled** — benchmarked twice, would appear twice on the site |
| Derive worker lanes from enabled models | `_worker_providers` returns all of `PROVIDER_MODULES` minus exclusions, so threads spawn for `openrouter`, `runpod`, `lambda`, `anyscale`, all at 0 enabled models |
| Delete dead `MuiDataGrid` theme block | ~70 lines in `components/theme/theme.ts`; `DataGrid` used by no page; drop `@mui/x-data-grid` |
| Fix hydration error | `components/tables/TanStackTable.tsx:293` renders `<th>` inside `<div>` |
| Prune `models` | 1,170 documents, 225 enabled |
| Archive stale root files | `add-latest-verified-models.js`, `add-missing-models.js`, `comprehensive-model-discovery-agent.js`, `get-all-current-models.js`, `research-provider-apis.js`, `test-models-api.js`, four competing `STATUS_DASHBOARD_DESIGN*.md`, `DATA_MODEL_ANALYSIS.md`, `automation-summary.md`, `VERTEX-SETUP-COMPLETE.md` |

The six one-off scripts all do some version of "figure out what models exist" —
the job Workstream A would own permanently. Their existence is the symptom.

Remaining coverage gaps (16 models): 3 stale Bedrock IDs on the EC2 runner,
2 stepfun models, `codex-mini-latest`, and 10 Together entries of which 4 are
case-duplicates that the A-workstream index fix resolves.

---

## Ordering

```
D (cleanup)          — independent, do anytime, small
A3 normalizer        — validate against the existing table first
A2 probe/promote     — depends on nothing; A3 assigns identity at promotion
A1 reconciler job    — wraps A2 + A3 into the nightly loop
B  measurement       — needs a decision; gates schema changes
C  redesign          — independent; A3 affects legend, B affects metrics shown
```

A3 validation is the first thing worth doing, because it has an objectively
checkable answer and the existing table is a free eval set.

## Open questions

1. Probe budget: how many calls per candidate, and what is the acceptable
   one-off cost for the initial ~1,000-model sweep?
2. Should reasoning-on and reasoning-off be separate published profiles, or
   should only one be published per model?
3. Does grouping policy merge serving optimizations (`Turbo`, `instant`) with
   the base model, or split them? This is a product call, not a technical one.
4. Is rewriting historical rows in scope when identity assignments change, or
   does identity only apply going forward?
