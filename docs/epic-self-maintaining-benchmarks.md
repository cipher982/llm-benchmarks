# Epic: a self-maintaining benchmark site

Status: draft for review · 2026-08-04
Supersedes `docs/platform-plan.md`, which is folded in below.

## Goal

llm-benchmarks.com should run without David. Not "with less effort" — without
him. He should not curate model lists, chase dead providers, approve routine
changes, or open an agent session to fix things that broke quietly.

He maintained all of this by hand for years. That is being retired because
current models can do it continuously and at a cost that does not matter.

**Done means:** a new model appears at a provider and is benchmarked, correctly
grouped and published within a day, with no human involvement. A model that
disappears is deprecated the same way. When something breaks, the system
notices, fixes what it can, and reports only what it genuinely could not.

## The governing evidence

On 2026-08-04 the pipeline was found badly degraded and recovered. What broke
determines what this epic builds:

| Failure | Kind | Duration undetected |
|---|---|---|
| Worker threads died on an unhandled Mongo error; process stayed up, container reported `Up` | runtime | 8 days |
| Dead-lettered jobs had no path back; every transient failure permanently removed a model | design | ~2 months |
| Queue never checked whether a model was still enabled | seam | until reviewed |
| `liveness_status` reports healthy if any one provider has recent data | design | shipped same day |
| `llm-bench-health` and `llm-bench-provider-discovery` registered `enabled=False` | config | 3 months |
| Provider `/models` does not indicate serverless availability | external | ~2 months |
| DeepInfra dead on a billing 402 | external | 7 weeks |

Coverage decayed from full to 11.7% of enabled models with no single incident.

**Every one of these is two components that individually work, disagreeing —
or a component reporting success while doing nothing.** None would have been
caught by unit tests; the runner had 29 passing tests throughout the outage.

That is why this epic starts with production invariants rather than a test
suite, and why the autonomy work is sequenced before any cosmetic refactor.

---

## Phase 0 — eyes

An agent needs eyes before hands. Continuous assertions over production state,
evaluated on a schedule, each one derived from a failure above.

| Invariant | Catches |
|---|---|
| No queued or running job targets a disabled or deprecated model | the 88% violation found on 2026-08-04 |
| Every enabled provider wrote a metric within its cadence | dead provider lanes masked by a healthy one |
| Every enabled model has data within N× cadence, or a recorded terminal reason | silent model loss |
| `provider_catalog.last_seen_at` under 48h | a discovery job disabled or failing |
| Dead-letter count not growing week over week | ratchets |
| No model enabled twice under different casing | double-counting on the site |
| Every Sauron job tagged `llm-benchmarks` ran within its cadence | jobs registered `enabled=False` |
| Row volume within a band of trailing median, per provider | partial collapse |

Rules:

- Each invariant returns pass/fail plus the offending records, not a boolean.
- Violations are **acted on where the action is reversible** — cancel ineligible
  jobs, requeue starved models, disable a model with a recorded reason. Report
  only what could not be fixed.
- A failed alert delivery fails the job. A page nobody received is not a page.
- Thresholds are absolute where the correct value is known (ineligible jobs must
  be zero) and relative to trailing history where it is not.

Also in Phase 0, from the adversarial review of the prior plan:

- `liveness_status` must require per-provider progress, not any-provider.
- Coverage alert floors (currently 50% model, 75% provider) must not let a large
  regression land as `[INFO]`.
- Observe one real scheduled discovery run and one health run, including email
  outcome, before treating either control as live.

Exit criteria: every invariant either green or carrying a dated,
machine-readable exception, for seven consecutive days — several cannot go
green until Phase 1 lands, so requiring green alone is circular. Plus at least
one violation detected and auto-remediated with no human involvement.

---

## Phase 1 — the reconciler

One nightly job that keeps the catalogue in sync with reality.

```
refresh provider_catalog from every provider API
  ↓
diff against models
  ↓
NEW   → insert enabled:false, status:"probing" → probe
        pass → assign identity → enabled:true
        terminal fail → enabled:false + observed reason
        transient → retry tomorrow
  ↓
GONE  → absent from 3 consecutive *completed* syncs → deprecated:true
        (a failed or skipped sync is not evidence of absence)
  ↓
STALE → enabled, no success in 7d, terminal errors → demote
        (exempt when the whole provider is failing — a 7-day billing lapse
         must not demote a provider's entire catalogue, which is exactly what
         DeepInfra's 402 would have done)
```

**Probe before promote.** A candidate is admitted because a real benchmark call
succeeded, not because an API listed it. This subsumes modality filtering
entirely: an image or TTS model cannot return text tokens at a measurable rate.
Two passes of name-pattern filtering still let through `veo`, `kling`, `vidu`,
`ideogram` and `parakeet`; the probe needs no brand list and never goes stale.

Verified 2026-08-04 — no provider field predicts serverless text availability:

| Provider | Claims | Reality |
|---|---|---|
| together | `type: "chat"`, normal pricing, `running: false` for all 274 | dedicated-only models are indistinguishable |
| fireworks | lists it | `404 not deployed` |
| cerebras | lists it | `404 no access` |
| groq | lists it | `400 requires terms acceptance` |
| openai | lists it | `dall-e`, `sora`, `realtime`, `davinci-002` |

**Stages stay separable.** The adversarial review was right that these three
concerns share a data model but must not share a failure boundary. Discovery is
free and read-only; probing has paid side effects and takes days to establish
stability; identity is a semantic publication decision. An LLM or OpenRouter
outage must not stop catalogue refresh, and must not stop collection under an
ungrouped name. Each stage commits its own results and degrades independently.

Exit criteria: a new model at any provider reaches the site within 24h with no
human action, and a retired model is deprecated within 4 days.

---

## Phase 2 — model identity

`llm-benchmarks-dashboard/backend/utils/modelMapping.ts` is 377 hand-maintained
lines, one of five files totalling 1,427 lines that all map model names. It is
already wrong: `Meta-Llama-3-8B` (base) and `Meta-Llama-3-8B-Instruct` both map
to `llama-3-8b`.

The governing asymmetry: **a false merge is far worse than a missed merge.**
Grouping Together's FP8 deployment with Bedrock's BF16 reports a provider speed
difference that is actually a quantization difference, silently. A missed merge
shows two lines instead of one — visible and self-correcting.

**LLM normalizes, code groups.** One call per *unseen* model ID returns
structured attributes; grouping is deterministic over those attributes.

```json
{ "developer": "meta", "family": "llama", "version": "3.1", "params": "8B",
  "variant": "instruct", "quantization": "fp8", "context_variant": null,
  "serving_optimization": "turbo", "confidence": 0.95, "reasoning": "..." }
```

```
canonical = f(developer, family, version, params, variant)
```

`quantization` and `serving_optimization` are excluded from the key and carried
as display annotations, so the chart legend reads `groq (fp8)` rather than
`groq`. This is idempotent (new models cannot perturb existing groups),
cacheable (~20 calls/day), reviewable per decision, and keeps *policy* in code
where it can be versioned while *judgment* goes to the model.

Uncertainty escalates to more inference — a second opinion, a stronger model, an
evidence pass against provider docs — never to a human queue. Merges into an
established time series must be non-destructive: write the new identity forward
and keep the old series addressable rather than rewriting history.

**Validate against the right target.** Production runs
`USE_DATABASE_MODELS=true`, so `modelMapping.ts` is the *fallback* path and
`models.display_name` is what actually ships. Validate against live display
names first; the 377-line table is a useful second corpus, not ground truth.

**Split by default on ambiguity, and prefer agreement over confidence.** The
self-reported confidence field is uncalibrated. Two independent derivations
agreeing is evidence; a model asserting 0.95 is not. Ambiguous cases stay split
into separate series, which is the safe direction given a false merge is worse
than a missed one — and it is what makes the no-human-queue rule sound rather
than dogmatic.

Caching per unseen ID never revisits mutable aliases such as `-latest`; those
need periodic re-derivation.

Exit criteria: normalizer reproduces live display names within a stated error
rate, with every disagreement explained.

---

## Phase 3 — measurement semantics

`docs/reasoning-token-budget-spike.md` is a completed proposal awaiting a
decision. It found the published series is already protocol-contaminated: of
1,182 Together rows in 30 days, 975 came from multi-attempt retries, 860 had a
final cap different from the nominal 64 tokens, and 119 were silently labeled
reasoning-disabled. The runner publishes only the final successful attempt,
biasing distributions toward requests that eventually produced text.

Needs a decision on versioned benchmark profiles, separating answer yield from
generated work, and recording budget exhaustion as an outcome rather than an
error. Gates any schema change the reconciler writes. Streaming-only model
support (4 disabled Qwen models) sequences here, same accounting code.

---

## Phase 4 — delete, then refactor what survives

In this order, because Phases 1–2 obsolete the messiest code:

- delete `modelMapping.ts` and most of `modelMappingDB`/`Merge` (~800 lines)
- delete the six root-level one-off discovery scripts
- archive four competing `STATUS_DASHBOARD_DESIGN*.md` and other stale docs
- case-insensitive unique index on `models`; prune 1,170 docs to what is real
- derive worker lanes from enabled models, not `PROVIDER_MODULES`
- delete the dead `MuiDataGrid` theme block and drop `@mui/x-data-grid`
- fix the `<th>` in `<div>` hydration error at `TanStackTable.tsx:293`
- fix `docs/` being gitignored while five docs are tracked

Only then refactor what remains and is genuinely load-bearing: `dataProcessing.ts`
(510) and `pages/api/processed.ts` (473). Capture current output as golden
fixtures before cutting — a safety net exactly where the knife goes, not blanket
coverage. `cloud.tsx` (764) is rewritten by Phase 5 and needs no separate work.

---

## Phase 5 — dashboard

`llm-benchmarks-dashboard/backend/docs/redesign-epic.md`. Console direction
chosen, Phase 1 complete, Phases 2–5 outstanding. Independent of the rest except
that Phase 2 changes what the legend shows and Phase 3 changes which metrics
exist. Open blocker: provider and model pages return 500 locally with no fixture
fallback, so two of five templates cannot be iterated offline.

---

## Sequencing

```
Phase 0  eyes              ← start here; closes open review findings
         case-insensitive unique index on models moves HERE from Phase 4:
         production has 5 case-duplicate enabled pairs today, and the
         reconciler must not insert against an index that cannot see them
         sample_role provenance moves HERE from Phase 3: probe rows must be
         separable from published measurements before any probing starts
Phase 1  reconciler        ← needs 0 to verify itself
Phase 2  identity          ← validate against the table first; independent of 1
Phase 3  measurement       ← needs a decision from David
Phase 4  delete + refactor ← after 1 and 2 remove the code
Phase 5  dashboard         ← independent
```

## Open questions

1. Probe budget: calls per candidate, and acceptable one-off cost for the
   initial ~1,000-model sweep.
2. Reasoning-on and reasoning-off as separate published profiles, or one?
3. Does grouping merge serving optimizations (`Turbo`, `instant`) with the base
   model, or split them? Product call, not technical.
4. Is rewriting historical rows ever in scope when identity changes, or is
   identity forward-only?
5. What is the escalation channel when the system genuinely cannot proceed, and
   what is the maximum acceptable rate of such escalations?

## Non-goals

- Blanket unit test coverage. The failures were seam failures; isolated tests
  do not see them.
- Refactoring for tidiness ahead of the autonomy work.
- Any human review queue. Routing low confidence to a person is a design
  failure in this system.
