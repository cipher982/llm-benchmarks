# Epic: a self-maintaining benchmark site

Status: revised after two independent reviews · 2026-08-04
Supersedes `docs/platform-plan.md`, which is folded in below.
Reviews: `docs/epic-review-claude-fable-5.md`, `docs/epic-review-gpt-5.6-sol.md`.

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

**The authority boundary.** A small set of events cannot be resolved by any
amount of inference, because they need authority the automation does not hold:
adding funds or accepting a new spending level, accepting provider terms,
recovering an account behind human-bound 2FA or identity proofing, responding to
a compromised credential when the automation's own authority must be revoked,
renewing the domain, and destroying published history. The system stays
autonomous *around* these: mark the provider `blocked_external_authority`, stop
its spend, keep every other provider running, preserve evidence, and send a
non-blocking notification. Stating this envelope is not a retreat from the goal
— an epic that claims to eliminate it would just be wrong about where the
remaining human touchpoints are.

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

An agent needs eyes before hands. Both reviews rejected the first version of
this phase for the same reason, and the correction is the spine of everything
below.

### The defect: a detector that can green itself

The original design evaluated every invariant against live catalogue state and
then auto-remediated violations by mutating that same state. So the cheapest way
to satisfy "every enabled model is fresh" was to disable the stale model.
Disabling a whole provider improved coverage ratios. Reclassifying dead letters
until growth stopped satisfied the dead-letter check. Each resulting state is
internally consistent and says nothing about whether the site is correct.

Seven consecutive green days under that design proves only that the checks
agreed with the state after their own actions.

### The correction: observation and action are separate

**The denominator is an immutable record, not live state.** A `desired_set`
snapshot is captured on a schedule and never modified. Coverage checks read the
newest snapshot that is *already older* than the window they judge, so an action
taken now cannot change the verdict on the window that produced it — only a
later one, where the difference between two snapshots is itself the audit trail.
`desired_set_is_not_silently_shrinking` watches that difference directly, which
is the check that would have caught the decay to 11.7%: no single incident, just
individually reasonable demotions adding up.

**Every run is recorded.** An immutable check-run row per evaluation, with
inputs, threshold version and per-check outcome. Without it, a quiet period is
indistinguishable from one where nothing ran — which is exactly how a disabled
Sauron job passed for three months as a live data source.

**Missing inputs are not a pass.** A check that cannot reach what it needs
returns `CannotEvaluate`, which reads as neither green nor a violation. "I could
not look" and "I looked and it was fine" need different fixes.

**The watcher lives outside what it watches.** A Sauron job that checks Sauron
jobs cannot report when Sauron, clifford, networking, or its own schedule is
down. An external dead-man heartbeat and a black-box check against the public
site — both off clifford — are the only things that cover that domain.

### The invariants

Built and passing at `api/llm_bench/ops/invariants.py`:

| Invariant | Denominator | Catches |
|---|---|---|
| `no_work_for_disabled_models` | live, both sides | the 88% violation found on 2026-08-04 |
| `no_case_duplicate_models` | live | double-counting on the site |
| `no_job_is_stuck_in_queue` | live | a lane that accepts work and never drains |
| `every_provider_is_progressing` | snapshot | dead lanes masked by a healthy one |
| `desired_models_are_being_measured` | snapshot | silent model loss |
| `desired_set_is_not_silently_shrinking` | snapshot | the detector shrinking its own scope |
| `discovery_completed_recently` | run ledger | a discovery job disabled, failing or truncated |
| `terminal_reasons_are_current` | live | a `disabled_reason` that suppresses forever |

Agreement checks read live state deliberately: when queue and catalogue
disagree, the disagreement *is* the fault, and cancelling the job moves no
denominator.

Three checks from the first version were removed as unsound:

- `catalogue_is_fresh` read the newest `provider_catalog` row. Requiring every
  row to be fresh fires forever on genuinely retired models; requiring the
  newest to be fresh passes on a partial one-row response. The observable event
  is a **completed provider sync**, so this now reads a discovery run ledger and
  refuses to evaluate without one.
- `provider_volume_within_band` compared against a trailing median, which
  *learns a degraded state*. The July failure was a slow ratchet; a moving
  baseline eventually calls it normal.
- `dead_letters_are_not_accumulating` counted a pile rather than an outcome. A
  stable 1,409 dead letters is equally compatible with full coverage and with
  none.

`terminal_reasons_are_current` is new and load-bearing: a terminal reason needs
an expiry and a recovery probe, or it becomes the ratchet in a different
costume. DeepInfra's 402 cleared when the balance returned; a `billing` label
written once and trusted forever would have kept it dead.

### Action classes

Not "reversible, therefore automatic". Enabling a model spends money and
publishes a semantic claim, and setting `enabled:false` later reverses neither.

| Class | Actions | Gate |
|---|---|---|
| Safe | cancel ineligible work, requeue an exact transient failure under a bound | the violation itself |
| Evidence-gated | disable, deprecate, change cadence, stamp a terminal reason | separate evidence + blast-radius rules |
| Authority-gated | anything in the authority boundary above | notify and continue elsewhere |

Every mutation batch carries a batch ID, a before-image, and an inverse
operation. `disabled_reason` plus a timestamp is an audit hint, not a rollback
mechanism — without prior values a confidently wrong agent can publish and then
erase the evidence needed to recover.

Built at `api/llm_bench/ops/mutations.py` and used by admission. An over-limit
batch applies *nothing* rather than its first N changes, because half a
migration is the state nobody designed for. There is a per-provider cap under
the global one, since one provider going dark is the shape the July decay
actually had, and a kill switch stops every mutation while read-only monitoring
keeps running.

Limits are admission requirements, not implementation details: per-run and
per-provider call caps, per-run and daily USD ceilings from conservative output
caps, a maximum catalogue delta per run, provider circuit breakers for auth /
billing / rate-limit / broad 5xx, and a kill switch that stops paid probes and
mutations while read-only monitoring continues. Billing recovery uses one
low-frequency sentinel, not a retry of every model.

None of these gates is a human approval queue.

### Exit criteria: fault injection, not green time

Passive green time is close to a tautology. The phase is certified by injecting
each failure and observing detection and recovery:

1. Kill one provider lane while another continues → provider-specific failure
   and recovery, no process restart.
2. Feed discovery a partial or empty page → the run fails and performs zero
   deprecations.
3. Fail alert delivery and stop Sauron → the external dead man fires.
4. Seed an ineligible job → cancelled without a provider call.
5. Seed a stale model under provider-wide auth failure → provider pause, not
   model demotion.
6. Attempt an over-limit mutation batch → contained, with a reversible record.

### Done in this phase

- [x] Queue/catalogue eligibility gate (`cf1295f`, deployed; 223 jobs cancelled,
      zero ineligible remaining).
- [x] Invariant engine reworked against both reviews.
- [x] Worker lanes derived from the catalogue rather than `PROVIDER_MODULES`, so
      "no metric from this provider" is unambiguous.
- [x] `liveness_status` reports per-provider progress; the aggregate boolean
      stays aggregate on purpose, because it drives process exit and restarting
      the container fixes neither auth nor billing.
- [x] Failed alert delivery fails the health job.
- [x] Discovery and health jobs verified by triggering them and checking the
      outcome, not the status field.
- [x] Discovery run ledger. Each provider read now records start, end, status,
      raw vs accepted counts, pagination completion and source version, and any
      provider error fails the run instead of passing when another provider
      happened to find something new.
- [x] Case duplicates resolved and a case-insensitive unique index installed,
      scoped to enabled rows so the 28 disabled duplicate groups keep their
      display history. Verified by attempting the duplicate insert.
- [x] Minimal metric contract: sample role, benchmark profile, protocol
      version, attempt group. Probe samples route to a separate collection and
      do not record freshness.
- [x] Black-box publication check, running off the app: fetches the endpoint a
      visitor gets and asserts freshness, absolute model and provider floors,
      and no model published under two spellings.
- [ ] External dead man off clifford.
- [ ] Fault-injection certification (the six cases above).

### What the coverage checks found once the snapshot settled

37 of 220 enabled models had no measurement in over four hours, while 182 sat on
a healthy ~90-minute cycle — so the threshold is separating genuine laggards
from normal cadence rather than firing on everything. The 37 break down as:

| Cause | Count | Note |
|---|---|---|
| `visible output empty after token budget exhausted` | ~19 | the reasoning-token defect, Phase 3 |
| timeout | 8 | mostly 405B/70B models at 120s |
| stale terminal reasons | 8 | no billing errors in the last 6h |
| rate limit / transient | 2 | self-recovering |

**The single largest blocker is not breakage.** A 64-token budget is consumed
entirely by hidden reasoning tokens on these models, leaving no visible text, so
the validator rejects a response the provider considers successful. This is the
defect `docs/reasoning-token-budget-spike.md` identified. It is not fixable
without deciding the profile question, because raising the cap changes what the
number means and makes new rows incomparable with old ones. Left alone
deliberately rather than patched silently.

`discovery_completed_recently` also correctly flagged Bedrock and Vertex, which
have no discovery authority at all — the gap Sol named.

**What running it against production immediately found**, which is the argument
for this phase existing:

- Five models enabled under two spellings, benchmarked and drawn twice.
- 466 disabled models carrying a terminal reason with no expiry — though nearly
  all correctly dead, which is what recalibrated the check.
- Six jobs the first version of the queue check misread as stalled, when they
  were in normal backoff.

Two of those three were the checks being wrong rather than production. Both
were found by running against live state, not by reasoning about the code.

---

## Cross-cutting: delete silent fallbacks

A silent fallback converts a loud failure into a quiet wrongness. That is the
exact failure class this epic exists to remove, and the codebase is full of it.

`mapModelNames(data, useDatabase)` catches any error from database mapping and
silently substitutes the 377-line hardcoded table, logging only to console.
Production runs `USE_DATABASE_MODELS=true`, so if DB mapping breaks, the site
keeps serving stale hardcoded names and nothing says so. This is not
hypothetical: it caused a factual error in this very epic, which cited the
hardcoded table as ground truth when it is the fallback path.

Others of the same shape:

| Fallback | What it hides |
|---|---|
| `mapModelNames` → `mapModelNamesHardcoded` on any exception | DB mapping broken; site serves stale names |
| static JSON → live Mongo query | the regeneration cron being dead |
| `server.js` catching per-file static generation failures | a partial regeneration serving yesterday's file |
| `resolveDisplayFromHardcoded` in `naming.ts` | missing catalogue metadata |
| `dotenv.load_dotenv()` at import scope filling unset env | which collection the process actually reads |
| worker lanes started for providers with nothing enabled | a dead lane looking idle *(fixed)* |

The rule for this epic: **a fallback must either be loud or not exist.** If a
degraded path is genuinely wanted, taking it must set an explicit state that an
invariant can see and an operator can query — not a console log. Where the
fallback exists only because the primary path was once unreliable, delete it
and let the primary path fail visibly.

This is not deferred to Phase 4 cleanup. Every phase that touches one of these
removes it as part of the work, because leaving them in place undermines the
invariants the same phase is adding.

---

## Cross-cutting: is the published product actually right

Every invariant above can be green while the site publishes false merges, mixed
benchmark protocols, stale static files, or implausible values. They test
operational consistency, not publication correctness — and an autonomous system
that is confidently wrong is a failure mode the original epic did not cover at
all.

A black-box check, running outside the app's failure domain, fetches the same
static endpoint users receive and asserts: freshness, expected provider and
model coverage, no duplicate active identities, one benchmark protocol per
series, and traceability from every published series back to endpoint, profile,
identity decision and source evidence.

An independent verifier runs on new identity relations and on suspicious public
deltas. When it disagrees, the batch is quarantined and rolled back via its
inverse — not queued for David.

Provider model IDs, descriptions and documentation are **untrusted input** to
the identity model. Anything that feeds an LLM whose output has mutation
authority needs source allowlists and least-privilege database access.

---

## Phase 1 — the reconciler

One nightly job that keeps the catalogue in sync with reality.

```
refresh provider_catalog from every provider API   ← must be raw, paginated,
  ↓                                                   complete, and recorded
diff against models
  ↓
NEW   → insert enabled:false, status:"probing" → probe
        pass → assign identity → enabled:true (probationary)
        terminal fail → enabled:false + observed reason + recheck_after
        transient → retry tomorrow
  ↓
GONE  → absent from 3 consecutive *complete* syncs → deprecated:true
        (a failed, skipped or truncated sync is not evidence of absence)
  ↓
STALE → enabled, no success in 7d, terminal errors → demote
        (exempt when the whole provider is failing — a 7-day billing lapse
         must not demote a provider's entire catalogue, which is exactly what
         DeepInfra's 402 would have done)
```

### Discovery must be observable before it can drive deprecation

The current job is none of the things the diff above assumes. It covers seven
providers and omits Vertex and Bedrock. It performs one GET per provider and
ignores pagination — Anthropic's models API defaults to 20 rows and exposes
`has_more`/`last_id`, which the job does not follow. It filters names, modes,
Together types and prices *before* writing `provider_catalog` and drops pricing
and raw capabilities, so a filter change looks exactly like model removal. And
it raises on provider errors only when `total_new == 0`, so a partial run
returns success and becomes deprecation evidence.

Required before any automatic `GONE`:

- One immutable sync-run record per provider: start, end, cursor completion,
  raw count, accepted count, source version, terminal status.
- Store the raw provider row; filter at read time, not write time.
- Count absence only across complete successful runs.
- Distinguish `provider_absent`, `unavailable_to_account`, `unsupported_profile`
  and `provider_deprecated`. Public `deprecated` must not conflate provider
  retirement with "not benchmarkable by this account".
- Explicit discovery authorities for Vertex and Bedrock, or drop the claim to
  cover "any provider".

### Probe before promote — necessary, but one call is not admission

A real call is the final authority on whether *this account and runner* can
execute a benchmark profile. It is not authority on whether the endpoint belongs
in a comparable text-generation benchmark, and the current runner makes that
concrete: it accepts any positive output from variable-output providers, so
guard, moderation, router, compound and multimodal chat models pass while
measuring a different product. It also writes every accepted result to
`metrics_cloud_v2` before reporting success, so reusing that path would publish
probe samples and contaminate health. A response can consume billable
generation and then fail local validation — a failed probe is not a free probe.
And the reasoning spike shows a success can be selected after protocol-changing
retries or a reasoning-disable fallback, which is not admission evidence for the
nominal benchmark.

Admission is therefore a bounded evidence policy:

1. Cheap exclusions from provider-declared product, modality and capability
   metadata. Not authoritative — but enough to avoid obviously irrelevant paid
   calls. Names remain hints, never authority.
2. One non-public contract probe validating adapter, response shape, usage
   accounting and requested controls.
3. The exact frozen benchmark profile, run in shadow at least twice across
   separate collection windows, with a stated success ratio and no
   protocol-changing fallback.
4. A deterministic suitability policy for guard, moderation, routing, compound,
   alias and opaque endpoint classes. An evidence-gathering model adjudicates
   unknowns; unresolved cases stay disabled.
5. Promotion into a **probation** state, with the admission ratio still being
   measured. A provisional label is more honest than claiming one call
   established stability.

Verified 2026-08-04 — no provider field predicts serverless text availability:

| Provider | Claims | Reality |
|---|---|---|
| together | `type: "chat"`, normal pricing, `running: false` for all 274 | dedicated-only models are indistinguishable |
| fireworks | lists it | `404 not deployed` |
| cerebras | lists it | `404 no access` |
| groq | lists it | `400 requires terms acceptance` |
| openai | lists it | `dall-e`, `sora`, `realtime`, `davinci-002` |

Together's current documentation does separate serverless and dedicated
catalogues, and Anthropic's models endpoint exposes capabilities. That metadata
is not authoritative enough to promote, but it is good enough to skip paying for
calls that cannot possibly qualify.

**Stages stay separable.** Discovery is free and read-only; probing has paid
side effects and takes days to establish stability; identity is a semantic
publication decision. An LLM or OpenRouter outage must not stop catalogue
refresh, and must not stop collection under an ungrouped name. Each stage
commits its own results and degrades independently.

Exit criteria: a new model at any provider reaches the site within 24h with no
human action, labelled provisional until multi-window stability is established,
and a retired model is deprecated after three complete syncs confirm absence.

---

## Phase 2 — model identity

`llm-benchmarks-dashboard/backend/utils/modelMapping.ts` is 377 hand-maintained
lines, one of five files totalling 1,427 lines that all map model names. It is
already wrong: `Meta-Llama-3-8B` (base) and `Meta-Llama-3-8B-Instruct` both map
to `llama-3-8b`.

The governing asymmetry: **a false merge is worse than a missed merge.** A wrong
merge reports one provider as faster than another when the two rows are not
comparable, and it does so silently. A missed merge shows two lines instead of
one — visible and self-correcting.

### Quantization is out of scope — the data settled it

An earlier draft split identity into base model plus a quantization-aware
variant key. Measured against production on 2026-08-04, that was not worth
building:

- 3 of 220 enabled models declare quantization anywhere (1%). No provider
  exposes it as a field; it survives only as a suffix on some IDs.
- Splitting on declared markers would affect exactly **one** chart line,
  `llama-3.3-70b` — which is also the site's only four-provider line.
- On that line the quantized deployment is not the outlier. Groq runs 153 tok/s,
  Bedrock 83, Together's FP8 Turbo 48, DeepInfra 13. The spread is 12× and
  quantization does not explain it; provider infrastructure does.
- Those series are already thin — 8 and 13 samples over 30 days. Splitting makes
  sparse data sparser to encode a distinction that does not drive the number.

So grouping is by base model. Quantization and serving optimization are carried
as annotations for display. The schema keeps room to split later if a provider
starts publishing real quantization metadata *and* both sides have enough
samples to compare, but nothing is built for it now.

The same measurement corrected the assumption behind this phase. Charts show few
providers per model because coverage is thin, not because matching is broken:
158 of 185 display groups are genuinely single-provider, and 22 base models are
served by three or more providers while we benchmark most at zero or one. More
providers per line comes from Phase 1 admission, not from a better key.

### What it found on real data

Run against 80 production endpoints on 2026-08-04. It agrees with the live
mapping on 52 groups and finds cross-provider merges the hand-built table
misses entirely:

| Derived group | Providers the table keeps separate |
|---|---|
| `claude-haiku-4.5` | anthropic, bedrock, deepinfra |
| `claude-opus-4.7` | anthropic, bedrock, deepinfra |
| `claude-sonnet-4.6` | anthropic, bedrock, deepinfra |
| `claude-opus-4.1` | anthropic, bedrock |

That is the answer to "why do I only see two providers per line" — several
three-provider lines already exist in the data and the mapping splits them.

**It also caught a false merge in its own first version.** `claude-haiku-4.5`
and `claude-sonnet-4.5` produced the same key, because the attribute schema had
no place for a vendor tier: Llama is distinguished by parameter count, Claude by
name alone. Fixed by making `family` carry the tier, and by making an unknown
role its own token rather than defaulting to `base` — a default that would have
merged unlabelled endpoints into Meta's declared base weights.

Both were caught because derived keys are compared against the live mapping
before being allowed to drive it, not switched over on faith.

### One question, no scaffolding

An endpoint is placed by showing a model every group that exists and asking
which one it belongs to, or whether it is new. That is the whole mechanism.

Two earlier versions of this failed the same way, and both failures are worth
recording because the pull toward them is strong:

1. **An attribute schema.** IDs were decomposed into
   `developer/family/version/params/role` and a key assembled from them. That
   only works for names that decompose that way. Anthropic's do not — Claude is
   distinguished by tier, not parameter count — so Haiku merged with Sonnet, and
   the fix was to list the tiers in the prompt: `claude-haiku`, `claude-sonnet`,
   `gemini-flash`, `nova-pro`. That is the 377-line table in a different file,
   needing an entry for every vendor that names something a new way.
2. **Candidate filtering.** The question was right but the options were
   pre-selected by shared tokens, against a hand-maintained list of words to
   ignore (`instruct`, `turbo`, `versatile`). Choosing which groups an endpoint
   may be compared against is itself a claim about which models resemble each
   other — the judgment being delegated.

The whole list goes in the prompt. A few hundred short strings costs nothing
next to a wrong merge, and there is no rule left to get wrong. A vendor with a
convention nobody anticipated forms its own group without anyone editing
anything.

**Self-reported confidence is not a control-plane field.** `confidence: 0.95` is
not calibrated probability and must not drive publication or merge decisions.
Store evidence class and verifier outcome; keep confidence as a diagnostic if it
proves useful. Saving a `reasoning` string does not make a decision
reproducible — source evidence, prompt and model version, and the relation do.

**Validate against the right target.** Production runs
`USE_DATABASE_MODELS=true`, so `modelMapping.ts` is the *fallback* path and
`models.display_name` is what ships. The 377-line table is known to contain
false merges, so reproducing it measures imitation of old policy, not identity
accuracy. Evaluate against source-backed positive and negative cases with a
held-out set and an independent label — "reproduces the table within a stated
error rate" has no number and is not an exit criterion.

The identity schema and effective dates must exist before any mapping changes.
The dashboard currently applies today's mapping while reading old metric rows,
so forward-only identity cannot be enforced by the current read-time mapping.

---

## Phase 3 — measurement semantics

`docs/reasoning-token-budget-spike.md` is a completed proposal awaiting a
decision. It found the published series is already protocol-contaminated: of
1,182 Together rows in 30 days, 975 came from multi-attempt retries, 860 had a
final cap different from the nominal 64 tokens, and 119 were silently labeled
reasoning-disabled. The runner publishes only the final successful attempt,
biasing distributions toward requests that eventually produced text.

**The minimal part of this phase is a hard dependency on all probing, not a
later decision.** Before any probe runs, the metric contract needs: profile ID,
protocol version, `sample_role`, attempt group, cost/usage, and publication
filtering. Today a successful probe *is* a public metric and a model-health
success, so probing without this contaminates the series it is meant to protect.

The rest — versioned benchmark profiles, separating answer yield from generated
work, recording budget exhaustion as an outcome rather than an error — is a
one-time product decision that must be made before the autonomy work starts, not
a recurring operational dependency. Streaming-only model support (4 disabled
Qwen models) sequences here, same accounting code.

Per-profile cadence is defined here. Until it exists, a global `N × cadence`
staleness rule is not valid: production runs `FRESH_MINUTES=30`, but only 153 of
225 enabled models succeeded within 30 minutes and 201 within 90, and expensive
reasoning profiles will deliberately need different cadences.

---

## Phase 4 — delete, then refactor what survives

In this order, because Phases 1–2 obsolete the messiest code:

- delete `modelMapping.ts` and most of `modelMappingDB`/`Merge` (~800 lines)
- delete the six root-level one-off discovery scripts
- archive four competing `STATUS_DASHBOARD_DESIGN*.md` and other stale docs
- retire the operator package and the `/admin/model-review` workflow, which
  still encode confidence thresholds and pending human approval; leaving them
  means the repo carries two contradictory lifecycle authorities
- fix the `<th>` in `<div>` hydration error at `TanStackTable.tsx:293`
- delete the dead `MuiDataGrid` theme block and drop `@mui/x-data-grid`
- fix `docs/` being gitignored while five docs are tracked

Moved out of this phase: the case-insensitive unique index (Phase 0 — the
reconciler must not insert against an index that cannot see duplicates) and
worker-lane derivation (done). **Not** doing: pruning the 1,170 disabled model
documents. They are small, they retain audit and display data for historical
metrics, and old URLs may still resolve through them. Merge proven duplicates,
keep aliases and history.

Only then refactor what remains and is genuinely load-bearing: `dataProcessing.ts`
(510) and `pages/api/processed.ts` (473). Capture current output as golden
fixtures before cutting — a safety net exactly where the knife goes, not blanket
coverage.

---

## Phase 5 — dashboard: not in this epic

The redesign is valid work with its own epic
(`llm-benchmarks-dashboard/backend/docs/redesign-epic.md`, Console direction
chosen, Phase 1 complete) but it does not make the site self-maintaining and
should not sit on the autonomy critical path. Layout work can proceed in
parallel; chart and legend work depends on settled profile semantics from
Phase 3.

---

## Sequencing

Revised after review — several items had hidden dependency inversions.

```
1  lane selection (done), alert delivery (done), credential hygiene (done)
2  immutable desired-set + check-run records; fault-inject Phase 0 read-only,
   with no catalogue mutation at all
3  external dead man + black-box public contract check, both off clifford
4  minimal Phase 3 contract: profile ID, protocol version, sample_role,
   attempt group, cost — required before ANY probe runs
5  resolve case duplicates; install the intended unique constraint
6  discovery: raw, paginated, complete, with a run ledger
7  bounded shadow probes + action-specific remediation + circuit breakers
8  identity entities and effective dates; then evaluate assisted matching
   against source-backed cases
9  automatic promotion/demotion under batch and spend limits; end-to-end
   publication fault tests
```

Cleanup and refactoring follow the autonomy path rather than sitting inside it.

## Decisions

**Settled 2026-08-04:**

- **The benchmark surface is any endpoint that takes a query and returns text.**
  Guard, moderation, router and compound models are in. "How fast is Llama
  Guard" is a real question and latency on a classifier is useful. Only
  genuinely non-text endpoints are out — embeddings, TTS, transcription, image,
  video — and a live call is what establishes that, not a name pattern.
- **Grouping is by base model.** Quantization is annotation, not identity. See
  Phase 2 for the measurement that settled it.
- **Publication timing:** publish at 24h labelled provisional, promote after
  multi-window stability. Reversible, so it did not need to be escalated.

**Still open:**

1. **What daily and per-run spend can the automation exercise without David?**
   The 24h recovery snapshot held 3,554 successful metric rows and 2,099 error
   rows, so "hundreds of calls a day" already understates steady state and a
   1,000-candidate sweep needs a real number. This one genuinely needs David —
   it is money, which is inside the authority boundary.
2. **What are the authoritative discovery sources for Vertex and Bedrock?**
3. **Which external service owns the dead-man heartbeat**, so clifford and
   Sauron do not watch themselves?
4. Reasoning-on and reasoning-off as separate published profiles, or one?
5. Is rewriting historical rows ever in scope when identity changes, or is
   identity forward-only?

## Non-goals

- Blanket unit test coverage. The failures were seam failures; isolated tests
  do not see them.
- Refactoring for tidiness ahead of the autonomy work.
- Any human review queue. Routing low confidence to a person is a design
  failure in this system — ambiguity resolves to more evidence, an independent
  verifier, or a conservative no-action state.
- A general experiment framework for probes. The spike's explicit
  benchmark-profile object plus `sample_role` and attempt provenance is enough.
