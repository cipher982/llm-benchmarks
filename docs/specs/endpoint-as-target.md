# The endpoint is the benchmark target

Status: proposed (rev 2)
Date: 2026-08-17

Rev 2 incorporates an adversarial review and direct measurement against the
live OpenRouter API and the production runner. Claims below marked **measured**
were executed, not reasoned about.

## Summary

Make an OpenRouter *endpoint* — identified by its **exact endpoint tag** — the
unit of scheduling, measurement, storage and display. Today the scheduled unit
is a model and the serving provider is an outcome observed after the fact. That
mismatch is the origin of every open defect on the site.

The endpoint list is published, not something we build:

```
GET /api/v1/models/{author}/{slug}/endpoints
  → data.endpoints[] = { tag, provider_name, quantization, pricing,
                         context_length, max_completion_tokens,
                         supported_parameters, status,
                         uptime_last_30m, uptime_last_1d }
```

## Why now

### The site publishes the price floor and calls it throughput

`cloud/providers/openrouter.py:83` returns `{}` for every or-served lane:

```python
# or-served (marketplace) lanes measure OpenRouter's default routing: the
# user does not pin, so neither do we.
if config.misc.get("route_policy") == OR_SERVED_POLICY:
    return {}
```

OpenRouter's default routing is **price-prioritized weighted load balancing**
— outage filtering, then inverse-square price weighting among stable providers,
then fallbacks. Premium providers are not excluded; they are structurally
under-sampled. Over seven days of our rows the practical result was
indistinguishable from exclusion: Groq, Cerebras, Fireworks and SambaNova
appear zero times across 5,052 samples.

**Measured** — pinned vs unpinned from the production runner, 2026-08-17:

| Model | Unpinned (served) | Pinned Groq |
|---|---|---|
| `openai/gpt-oss-120b` | CoreWeave **65.7** | **719.6** |
| `meta-llama/llama-3.1-8b-instruct` | DeepInfra **24.6** | **640.1** |
| `meta-llama/llama-3.3-70b-instruct` | DeepInfra **13.7** | **259.2** |

Llama-3.1-8B published at 20 tok/s is not a measurement of Llama-3.1-8B. This
is one request parameter, not a lost provider or a dead key.

### Non-determinism is structural

38 of 205 models were served by more than one upstream in seven days.
`z-ai/glm-5.2` hit eleven, from Baidu at 38 tps to CoreWeave at 7.8. Crediting
each upstream separately (shipped 2026-08-17) is correct but converts one
scheduled target into eleven published rows whose sample counts nobody chose.
47 of 272 upstream-lanes carry fewer than five samples and render as ordinary
leaderboard entries.

### The complexity is migration scaffolding

One client module per provider (`groq.py`, `cerebras.py`, `fireworks.py`, …)
because the original unit of work was "a key we hold," plus route decisions,
promotion/demotion, a renewal daemon, paired canaries, consolidation batches
and unrouted-retirement. That layer manages the *transition* between direct
keys and OpenRouter. 325 of 385 enabled models now route through OpenRouter;
the transition is over, and none of that layer measures anything.

`transport_provider` is already inert — `'direct'` on all 5,052 OpenRouter rows
in 24h, because a model whose catalogue provider is `openrouter` routes "direct
to OpenRouter." The dashboard's lane-splitting guards a distinction absent from
the data.

## The matrix, measured

All 325 enabled OpenRouter models queried 2026-08-17; 258 resolved
(67 rate-limited unauthenticated — re-run with `OPENROUTER_API_KEY`).

```
total endpoints          946
per model                mean 3.7   median 2   max 31
single-endpoint models   124  (48%)
distinct providers       67
```

Fast providers, and the honest scope of the recovery:

| Provider | Enabled models reachable |
|---|---|
| Together | 22 |
| Fireworks | 9 |
| Groq | 8 |
| SambaNova | 6 |
| Cerebras | 2 |

Groq's catalogue is small. But those eight are `llama-3.1-8b`, `llama-3.3-70b`,
`llama-4-scout`, `gpt-oss-120b`, `gpt-oss-20b`, `minimax-m2.7` — the
high-throughput models the site is known for. Eight targets restore the top of
the leaderboard.

946 targets is ~3× today's 325. 48% of models are single-endpoint, where
pinning is a free no-op that removes a bug class; the cost lands entirely on the
multi-endpoint half.

## The measurement problem this exposes

**This is the highest-stakes finding and it is not about routing.** Pinning
moves the site into a throughput regime its instrument was never tested in.

**Measured** — `openai/gpt-oss-120b` and `llama-3.1-8b`, production runner:

| Pinned | Served | Chunks | TTFT | t→64th | Legacy TPS | Delivered TPS |
|---|---|---|---|---|---|---|
| cerebras | Cerebras | **13** | 0.419 | 0.448 | **3730.4** | 142.8 |
| groq | Groq | 152 | 0.520 | 0.650 | 581.5 | 98.5 |
| deepinfra | DeepInfra | 48 | 3.198 | — | 397.3 | — |
| *unpinned* | CoreWeave | 68 | 5.890 | 7.683 | 135.4 | 8.3 |
| groq | Groq | 256 | 0.116 | 0.882 | 243.7 | 72.6 |
| deepinfra/fp8 | DeepInfra | 247 | 0.378 | 2.180 | 36.6 | 29.4 |

Three conclusions:

1. **Legacy `tokens_per_second` is invalid in this regime.** It divides
   completion tokens by `elapsed - ttft`. Cerebras returned 256 tokens in
   **13 SSE chunks** over a 0.069s post-TTFT window, yielding 3730 tok/s. That
   measures the socket, not the model. Chunk granularity is provider-specific —
   13 chunks vs Groq's 256 for identical work — so legacy TPS is not comparable
   across providers at high speed *at all*.

2. **Delivered TPS is structurally immune.** `VisibleTokenClock`
   (`cloud/visible_tokens.py:48-56`) records wall-clock from `time_0` to the
   64th visible token. It never divides by a collapsing window, so no chunking
   artifact can inflate it.

3. **But at high speed Delivered TPS asymptotes to a latency measure.** For
   Cerebras, TTFT was 0.419 of a 0.448s denominator — **93% of the metric is
   time-to-first-token.** That is arguably correct for a metric named
   *Delivered*, but it must be stated plainly: above roughly 500 tok/s,
   Delivered TPS ranks providers on TTFT, not on generation speed.

**Therefore Delivered TPS must be the sole published headline before pinning
ships.** Publishing legacy TPS alongside pinned fast endpoints would put
demonstrably fabricated numbers (3730 tok/s) on the front page. If generation
speed is to be published independently of latency, it needs a separate
instrument — a second visible-token mark, with the rate taken between mark 64
and mark N so TTFT cancels — and that is new work, not a rename.

The `deepinfra` row that never reached 64 visible tokens inside 256 max_tokens
is the existing `budget_exhausted` case; it is an endpoint-profile
admissibility question, not a failure.

## Comparability: what pinning does not fix

Endpoints of one model are **not interchangeable artifacts**. Measured, for
`meta-llama/llama-3.1-8b-instruct`:

```
tag                provider_name   quantization   context   max_completion
deepinfra/fp8      DeepInfra       fp8            131072    16384
novita/fp8         Novita          fp8             16384    16384
groq               Groq            unknown        131072   131072
cloudflare/fp8     Cloudflare      fp8             32000    32000
coreweave/bf16     CoreWeave       bf16           128000   128000
```

Weight quantization differs (`fp8` vs `bf16` vs self-reported `unknown`),
context differs by 8×, and generation ceilings differ by 8×. A bf16 endpoint
and an fp8 endpoint of "the same model" are different artifacts, and fp8 is
typically faster. Ranking them against each other as *the same model served at
different speeds* is not sound without disclosure.

The product claim must narrow accordingly. The honest claim is:

> How fast a given model is served by a given endpoint, where an endpoint has a
> specific quantization and configuration.

not

> How fast provider A serves model X versus provider B.

Quantization must therefore be a **published, first-class field**, not
metadata. Where it is `unknown` it must render as unknown rather than being
silently grouped with a known value. This is a UI honesty requirement, and it
is the strongest argument for the endpoint model: the current design cannot
express it at all.

## Design

### Target identity

```python
@dataclass(frozen=True)
class EndpointTarget:
    model_id: str                       # "openai/gpt-oss-120b"
    endpoint_tag: str                   # "deepinfra/fp8" — the pin key AND identity
    provider_name: str                  # "DeepInfra" — display only
    quantization: str | None            # "fp8" | "bf16" | "unknown"
    context_length: int
    max_completion_tokens: int | None
    supported_parameters: frozenset[str]
```

Identity is `(model_id, endpoint_tag)`. **Not** `(model, provider)` — tags
carry region and variant suffixes (`deepinfra/fp8`, `google-vertex/us-east5`),
and a bare provider-family slug matches every variant, which reintroduces the
non-determinism the spec exists to remove. Discovery must reject or quarantine
any target whose tag is a family base while multiple variants exist.

### Request

```python
extra_body = {
    "provider": {
        "only": [target.endpoint_tag],
        "allow_fallbacks": False,
        "require_parameters": True,
    },
}
```

`allow_fallbacks: False` is mandatory — a fallback silently measures a
different target, which is the present failure mode. `require_parameters` is
**retained**: without it OpenRouter may silently ignore unsupported parameters,
so an endpoint could accept the request while dropping part of the measurement
protocol. Retaining it is safe here because `only` already constrains selection
to one candidate; it degrades a wrong-protocol run into a clean refusal.

Fallback must be disabled at *every* level for endpoint jobs, not just at
OpenRouter:

- the routed→direct source-provider fallback (`scheduler/runner.py:637-674`)
- the direct fallback when the OpenRouter quota gate is unavailable
  (`scheduler/worker.py:175-205`)

Quota exhaustion must leave the sample pending or failed. It must never
substitute another lane, because for an endpoint target a substituted lane is a
fabricated measurement.

### Verifying what actually served

The spec previously proposed asserting `observed_provider == pinned tag`. That
is type-invalid: `observed_provider` is a display string (`"Groq"`,
`"Google AI Studio"`) and `tag` is a routing slug (`"groq"`,
`"google-vertex/us-east5"`). `scheduler/runner.py:489-500` already does the
correct thing, comparing `observed_provider_slug` to `route_provider_slug`.

Rule: compare slug to slug where a slug is available. Where OpenRouter returns
only a display name, record it and mark the row's endpoint identity
*unverified* rather than asserting it. Reject rows whose identity is
contradicted; do not reject rows whose identity is merely unproven, but do not
let them into rankings either.

### Catalogue

An `endpoints` collection replaces the `models` collection's role for
OpenRouter-served targets:

```
{ model_id, endpoint_tag, provider_name, quantization, enabled,
  first_seen, last_seen, missing_passes,
  context_length, max_completion_tokens, supported_parameters,
  or_status, or_uptime_1d, disabled_reason, disabled_at, disabled_by }
```

Discovery is one loop: for each model, fetch `/endpoints`, upsert one document
per endpoint. Absence is handled with **hysteresis** — an endpoint is disabled
only after N consecutive *complete* discovery passes fail to list it, and a
rate-limited or partial pass is not a pass. A single missing pass disabling an
endpoint would repeat the 2026-08-04 `max_attempts` ratchet in a new place.

`or_status` and `uptime_last_1d` arrive free per endpoint and are a better
health signal than anything currently derived.

### Scheduling

`scheduler/queue.py:41-43,107-136` keys jobs on `(provider, model_id)` and
`scheduler/cli.py:120-172` selects stale *models*. Both must key on
`(model_id, endpoint_tag)`.

"Round-robin" is not a design. What is required is a quota allocator:

- a defined publication window and a minimum number of **spaced** successes per
  endpoint within it — spacing matters because throughput varies with load and
  time of day, and five bursts in one minute measure one moment
- randomized timing rather than burst catch-up
- higher quota for headline endpoints
- surplus budget spent on narrowing confidence intervals for headline
  endpoints, not on extending the tail
- an explicit daily spend ceiling
- oldest-eligible-first selection, cap the work per pass, never slice the
  population — the invariant from the 2026-08-05 learnings, now applied to a
  pool 3× larger

Publication requires a minimum sample count *and* a confidence interval narrow
enough to rank. A bare count of five is a gate, not evidence; the published
statistic and its dispersion must be specified before cutover (see open
questions).

Budget arithmetic at a 5-sample floor: 946 × 5 = 4,730 successes per window —
676/day at a 7-day window, 158/day at 30 days. Today's ~45-minute cadence
applied to 946 targets would generate ~30,000 attempts/day, roughly 6× the
7-day floor before retries. The current OpenRouter concurrency gate is 4
(`scheduler/policies.py:31`).

### Storage and dashboard

Rows carry `endpoint_tag` and `quantization`. `endpoint_tag` becomes
first-class through ingestion, aggregation, identity lookup, lifecycle status,
URL generation and chart grouping; `provider_name` stays display-only.

Current pipeline gaps that must move together:

- `RawData` has no endpoint field (`utils/processCloud.ts:13-32`)
- aggregation keys on model + resolved provider + transport
  (`utils/processCloud.ts:150-151`)
- metadata is keyed provider/model/transport, not endpoint
  (`utils/modelMappingDB.ts:97-98`)
- `OBSERVED_PROVIDER_ALIASES` deliberately collapses Google variants
  (`utils/providerMetadata.ts:48-55`) — correct today, wrong once regional tags
  are distinct targets

`resolveServingProvider` and its alias table are retired once rows carry tags,
retained only as a fallback for pre-migration rows.

### What is deleted

`ops/route_renewal.py`, `ops/openrouter_consolidation.py`, the paired canary,
unrouted-retirement, route promotion/demotion and evidence expiry;
`RouteDecision`, `transport_provider`, `DIRECT_TRANSPORT`/`OPENROUTER_TRANSPORT`
and the dashboard lane split; per-provider clients for anything OpenRouter
serves.

### What is kept

`DIRECT_PROVIDERS = {openai, vertex, bedrock}` is unchanged — David's real
consumption, deliberately unrouted, Bedrock on separate infrastructure with its
own ingest path. The endpoint model applies only to the OpenRouter-served
catalogue.

## Migration

1. **Fix the instrument.** Make Delivered TPS the sole published headline and
   stop publishing legacy `tokens_per_second`, or gate it below the speed where
   chunking invalidates it. This must land *before* step 2, or pinning puts
   3730 tok/s on the front page.
2. **Recover the fast targets.** Pin the Groq, Cerebras, Fireworks and
   SambaNova endpoints of the models that have them. Small, reversible,
   restores the leaderboard, validates pinning end to end.
3. **Build `endpoints` alongside `models`.** Discovery writes both; nothing
   reads `endpoints` yet.
4. **Move the scheduler to endpoint targets** behind a flag, dual-writing
   `endpoint_tag`, with all fallback paths disabled for endpoint jobs.
5. **Move the dashboard to `endpoint_tag`**, publishing quantization.
6. **Delete the scaffolding** once no row in the retention window predates
   step 4.

Steps 1 and 2 are independently valuable and commit to nothing further.

## Decisions taken

**Pre-cutover history is excluded from pinned rankings.** Those rows were
produced by unpinned, price-biased routing against an unknown exact endpoint,
with potentially different quantization, region and reasoning behaviour.
`observed_provider` cannot reconstruct endpoint identity. They are retained,
labelled "unpinned OpenRouter routing," excluded from endpoint rankings and
provider comparisons, and not rewritten or retroactively attributed. Pinned
series start at cutover.

This destroys no history but does change what the site ranks, so it is flagged
for David rather than assumed.

## Open questions

1. **Published statistic and dispersion.** Mean, median or a percentile; how
   outliers are handled; what interval is published. Required before cutover —
   ranking without dispersion is what makes a 1-sample lane look like a
   measurement.
2. **Should unpinned routing be published as a second surface?** Pinned routing
   measures a counterfactual: what you get *if you pin*. Unpinned measures what
   an ordinary OpenRouter user actually experiences. Both are real products.
   The current site accidentally publishes the second while claiming the first.
3. **Sample budget in dollars** against the endpoint price distribution.
4. **Do the 67 unresolved models resolve authenticated**, or are some genuinely
   endpoint-less and unbenchmarkable?
5. **Per-endpoint profile admissibility.** `supported_parameters` varies by
   endpoint; where a profile needs a parameter the endpoint is inadmissible for
   that profile, not globally.

---

# Rev 3 — review findings and amendments

Two independent adversarial reviews (`cursor grok`, `codex luna` at xhigh) plus
direct measurement. **Both reviews concluded the spec should not be implemented
as written.** I verified their load-bearing claims rather than accepting them;
the material ones held, two did not, and the verification surfaced a defect
neither review found.

## Measured resolutions

| Question | Result |
|---|---|
| Does a base slug pin one endpoint? | **Unverifiable.** `only:["deepinfra"]` and `only:["deepinfra/bf16"]` and `only:["deepinfra/turbo"]` all return `provider: "DeepInfra"`. The response never echoes the tag. |
| Does `require_parameters` break a valid pin? | **No.** Groq pins identically with and without it. Luna correct, Grok incorrect — **keep it**. |
| Does `sort: "throughput"` recover fast providers? | **Yes — Cerebras 4/4 runs**, no pinning, one parameter. |
| Is `transport_provider` inert? | **No — my rev 1/2 claim was wrong.** It reads `direct` because `RouteDecision.direct()` sets `DIRECT_TRANSPORT` (`routing.py:67,95`). The field is load-bearing. |

## A1 — Endpoint identity cannot be verified, only asserted

The single most important correction. OpenRouter returns a provider **display
name** in generation metadata, never the endpoint tag. `deepinfra/bf16` and
`deepinfra/turbo` are indistinguishable in the response.

Consequences:

- `runner.py:489-500` can verify only at base-slug granularity. Variant-level
  identity is an assertion about the request, not a fact about the response.
- Any published claim at variant granularity carries unverified provenance and
  must be labelled as such.
- The spec's original `observed == pinned` assertion was type-invalid; the
  amended rule is base-slug prefix matching, with variant identity recorded as
  asserted-not-verified.

## A2 — Quantization spans fp4 to bf16 on one model

`openai/gpt-oss-120b`, 20 endpoints: `coreweave/fp4`, `novita/fp4`,
`baseten/fp4`, `parasail/fp4`, `nebius/fp4`, `deepinfra/bf16`, `akashml/bf16`,
`deepinfra/turbo` (bf16), `cerebras/fp16`, `groq` (unknown), `sambanova`
(unknown).

fp4 is far faster than bf16 and materially worse. Ranking them on one axis as
"gpt-oss-120b" would be the most misleading number the site has published.

This directly contradicts existing repo policy: `identity.py:25-27` and
platform-plan A4 (2026-08-04) deliberately **dropped** quantization because
provider hardware explained a 12× spread better. Endpoint-as-target reopens
that decision. It must be reopened explicitly, with a chart rule, not as a side
effect.

**Amendment:** quantization is part of published identity. Endpoints of
differing quantization do not share a ranking axis. `unknown` never groups with
a known value.

## A3 — Step 1 is not a policy flip

Both reviews described pinning as reverting `_route_options` for eight models.
Verified false. The pinned path is gated on paired direct-vs-routed canary
evidence — `canary_state: passed`, S3 evidence URI + sha256, verified cost,
TPS/TTFT CI bounds (`routing.py:164-200`), `require_promotion_evidence=True` by
default.

That gate answers "is routing equivalent to going direct?" — the migration
question. There is no direct Groq lane to be equivalent to. Pinning for
*comparison* needs a new admission path that keeps `allow_fallbacks: False` and
slug verification but drops direct-equivalence evidence, because the evidence it
demands cannot exist.

## A4 — Endpoint `status` is already a published health signal

Four `gpt-oss-120b` endpoints carry `status: -2` (`siliconflow/fp8`,
`sambanova`, `nebius/fp4`, `mara`). Admitting by presence, as rev 2 proposed,
schedules known-bad targets. Admission must gate on `status >= 0`.

## A5 — Identity must not become the provider slug

Tags contain `/`. The naming contract generates URL slugs from canonical
provider fields, so `providerCanonical = "deepinfra/fp8"` produces slashes in
paths. Three fields stay: `providerCanonical=deepinfra`,
`endpointTag=deepinfra/fp8`, display `DeepInfra`. Tag is a distinct path
segment.

## A6 — Identity must flow through health, not just the job id

`scheduled_job_id` is `f"{provider}:{model_id}"` (`queue.py:41-42`); eligibility
(`queue.py:238-280`), cancellation (`:301-317`) and health updates
(`worker.py:237-249`) use the same pair. Changing only the job id lets one
endpoint's success mark its siblings fresh. Job id becomes
`openrouter:{model_id}:{tag}`; workers stay on the `openrouter` lane so no
worker is named after a tag.

## A7 — Endpoint-specific pricing

Budget clamping reads model-level pricing (`runner.py:288-325`) while pricing is
per endpoint. A premium endpoint can run on a cheap endpoint's budget, times
946. Snapshot endpoint completion price; unknown price fails closed to the
minimal profile.

## A8 — Capacity

946 targets against `OPENROUTER_CONCURRENCY=4` (`policies.py:31`), a 30s tick,
and a 100-job-per-pass enqueue cap. A per-pass cap is not fleet capacity
control. Needs a global budget, jittered `next_due`, and backpressure on queued
plus running work.

## A9 — Sampling

`N=5` is a gate, not evidence. Recommended: ≥30 valid samples per endpoint over
≥7 days, spanning ≥5 distinct days and ≥6 UTC time blocks, ≥30 min apart with
jitter; 50 before a headline provider comparison. Publish median with a 95%
block-bootstrap interval resampled by day; p10/p90 on detail. Do not winsorize.
Count timeouts and 429s separately as availability, or the speed number is
silently conditional on success.

## A10 — `sort: "throughput"` is a third product, and it is cheap

Not considered in rev 1/2. Verified: returns Cerebras 4/4 for `gpt-oss-120b`.

Three distinct products now exist, and the site should stop conflating them:

1. **Unpinned default** — what an ordinary OpenRouter user experiences. What
   the site publishes today, accidentally.
2. **`sort: "throughput"`** — the best speed obtainable for a model. One number
   per model, one parameter, no new machinery.
3. **Pinned per endpoint** — provider comparison. 946 targets, and per A1 only
   verifiable at base-slug granularity.

David's stated product is (3). (2) is a one-line fix that stops publishing the
price floor immediately and is worth shipping regardless.

## A11 — Direct-provider double publication

OpenRouter lists `amazon-bedrock`, `google-vertex/*`, `google-ai-studio/*`.
With `DIRECT_PROVIDERS = {openai, vertex, bedrock}` retained, the same model
would publish twice. OR tags in that set must be excluded from the endpoint
catalogue.

## A12 — Instrument, restated

Rev 2 called Delivered TPS "structurally immune." Too strong. It is immune to
the divide-by-collapsing-window artifact that produced 3730 tok/s, but it
timestamps the SSE delta that crosses token 64, so resolution is lost whenever
one delta carries multiple tokens. It is a valid **client-delivery** measure and
not a decoder-speed measure at any speed. Intervals should use
`time.monotonic_ns()`, not `time.time()` (`visible_tokens.py:56`).

**No client-side code can reconstruct decoder timing after SSE batching.**
Publishing decoder throughput requires per-token timestamps or server-side
generation duration, which OpenRouter does not provide.

Note: OpenRouter publishes `throughput_last_30m` p50/p75/p90/p99 and
`latency_last_30m` per endpoint on authenticated `/endpoints` calls, already
stored in `bench_route_decisions.endpoint_evidence`. That is an aggregate over
real traffic rather than a controlled workload, so it does not replace the
benchmark, but it is a free prior for scheduling and an independent cross-check.

## Revised plan

Both reviews independently recommend the same shape, and the verification
supports it:

1. **Instrument the instrument.** Record per-row stream chunk count and maximum
   tokens-per-chunk, so unresolvable rows are identifiable. Prerequisite for
   everything else — without it, pinning fast endpoints puts 3730 tok/s on the
   front page, and `tokens_per_second` drives the distribution, table and
   time-series charts, not just the leaderboard.
2. **Decide the product** (see below). Ship `sort: "throughput"` if the goal is
   to stop publishing the price floor now.
3. **Pin a listed set of full tags** for the fast endpoints via a new
   comparison-admission path, quantization-labelled, base-slug verified.
4. Only then consider whether the catalogue key becomes `(model, tag)` — after
   job identity, health, pricing, capacity and URLs are designed.

## Decisions required before step 2

1. **Which product?** (1), (2), (3), or more than one surface.
2. **Does quantization split the ranking axis?** This reverses platform-plan A4.
3. **What happens to pre-cutover history?** Recommended: retain, label
   "unpinned OpenRouter routing," exclude from pinned rankings, do not rewrite.
