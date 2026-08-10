# Epic: conservative OpenRouter consolidation

Status: guarded implementation, 30-pair costed canary passed, activation remains opt-in
Owner: LLM Bench
Started: 2026-08-09

## Outcome

Reduce the number of provider-specific serving integrations used by the bench
where OpenRouter can serve the same model through the intended provider route,
while keeping every currently enabled model benchmarkable. A model remains on
its direct lane when its exact OpenRouter model, provider endpoint, request
protocol, or measurement semantics cannot be proved.

This epic formalizes the audit and rollout. It does not authorize disabling
providers, deleting model rows, rewriting published history, or switching the
production scheduler until the audit and review gates pass.

## Current baseline

The enabled catalogue snapshot used by the failed planning session contained
241 models across nine providers:

| Provider | Enabled models |
| --- | ---: |
| deepinfra | 124 |
| openai | 36 |
| together | 22 |
| bedrock | 21 |
| fireworks | 19 |
| anthropic | 10 |
| groq | 4 |
| vertex | 4 |
| cerebras | 1 |
| **Total** | **241** |

The public OpenRouter `/api/v1/models` response observed on 2026-08-09 had
400 entries. That response is useful as a dated catalog snapshot, but catalog
membership alone is not coverage evidence. The existing bench documentation
already records cases where a provider's `/models` endpoint lists a model that
cannot answer a real request. The previous temporary classifier also used
substring matching and damaged dotted version names such as `Qwen2.5`; its
results are discarded.

OpenRouter exposes a model endpoint listing at
`GET /api/v1/models/:author/:slug/endpoints`. The listing records provider
names, supported parameters, pricing, and context length. Provider routing
accepts provider slugs, not a unique deployment identity. A route decision must
therefore record both the requested `route_provider_slug` and the provider
actually observed in the response. The observed display identity may differ
from the requested slug, so the audit must keep a reviewed identity mapping.
Provider routing can restrict a request with `provider.only` or
`provider.order`; `allow_fallbacks: false` and
`require_parameters: true` are required for a provider-specific coverage test.

References:

- [OpenRouter endpoint listing](https://openrouter.ai/docs/api/api-reference/endpoints/list-all-endpoints-for-a-model)
- [OpenRouter provider routing](https://openrouter.ai/docs/guides/routing/provider-selection)
- [`llm-benchmarks` platform plan](platform-plan.md)
- [`llm-benchmarks` operating guidance](../AGENTS.md)

## Definitions

Each enabled catalogue row is an endpoint, identified by its current
`(provider, model_id)` pair.

- **Source provider**: the provider represented by the existing bench row and
  its credentials, such as `deepinfra` or `anthropic`.
- **Transport**: the API path used to make the request, either the direct
  adapter or OpenRouter.
- **OpenRouter model ID**: the exact `author/slug` sent to OpenRouter. It must
  be stored as evidence; it cannot be reconstructed from a display name at run
  time.
- **Route provider slug**: the provider slug supplied to OpenRouter. A base slug
  may cover variants, so it is not the same thing as a unique deployment.
- **Observed provider**: the provider identity reported by OpenRouter for the
  completed request, normalized through a reviewed mapping to the requested
  route slug. A request restriction without observed identity is not route
  evidence.
- **Route replacement**: a production request for an existing source-provider
  row is sent through OpenRouter and constrained to the intended provider.
- **Direct fallback**: the existing source adapter remains available for that
  row and is used when the approved route is unavailable or fails its health
  gate.

## Policy

The default decision is `keep-direct`. A row may receive `route-or` only when
all of these statements are true:

1. An exact OpenRouter model ID is resolved from current OpenRouter data. A
   fuzzy stem, display-name substring, or model-family guess is not enough.
2. The model's endpoint list contains an explicitly reviewed provider slug for
   the intended source provider. A provider-family match alone is insufficient.
3. A request using the production benchmark protocol succeeds with the exact
   model ID, the intended provider restriction, fallbacks disabled, and
   `require_parameters: true`.
4. The request supports the fields used by the relevant benchmark profile:
   streaming, token cap, reasoning controls, and any required response fields.
5. The result has usable output and the observed provider identity normalizes to
   the requested route slug through the reviewed mapping. A success that
   silently fell back to another provider does not qualify.
6. The direct lane can be restored without changing the model's identity or
   deleting its historical rows.

If any condition is unknown, the row stays direct and the reason is recorded.
Bedrock is excluded from replacement in this epic because account-backed
OpenRouter/Bedrock routing is not configured or in scope, and because its AWS
account, region, provisioned capacity, and request semantics are not the same
measurement surface as a generic OpenRouter endpoint. Vertex is audited row by
row; it is not automatically removed merely because a similar Gemini model
appears on OpenRouter.

The audit must never disable a source provider or remove a model from the
catalogue. A routed row is not allowed to replace its direct fallback until
the route has passed its canary gate. A provider outage, OpenRouter outage,
billing failure, or route mismatch must leave the direct row eligible for
recovery.

The runtime resolver is currently behind `OPENROUTER_ROUTING_ENABLED`, which
defaults to off. A queued route snapshot is evidence and does not activate
OpenRouter by itself. Scheduled jobs read the reviewed route-decision
collection and freeze the snapshot onto the job. The resolver also requires a
passed canary record before it can select OpenRouter. Activation remains
blocked until the dashboard transport series and canary gates are in place.

The route-decision collection is `bench_route_decisions` by default (override
with `MONGODB_COLLECTION_ROUTE_DECISIONS`). Its minimum activation record is:

```text
source_provider, source_model_id
route_decision_version, state=active
transport_provider=openrouter, route_model_id, route_provider_slug
route_probe_id, provider_metadata_verified=true
canary_id, canary_state=passed
canary_successes >= canary_required_successes >= 1
canary_cost_status=verified, canary_promotion_gate=passed
canary_evidence_uri, canary_evidence_sha256
canary_tps_ci95_lower, canary_ttft_ci95_upper, canary_cost_ci95_upper
route_snapshot_at, expires_at/recheck_at
```

Availability probes populate `route_probe_id`; they do not populate the
passed canary fields. Missing, stale, or invalid records stay direct.

The report-only audit can be materialized with
`PYTHONPATH=api uv run python scripts/openrouter_route_decisions.py`. It writes
one decision per audited source row, uses `state=candidate` and
`canary_state=availability_passed` for availability-qualified routes, and keeps
every other row direct. The command only writes MongoDB when both `--apply` and
`--yes` are supplied. A materialized candidate remains direct until a paired
measurement canary promotes it. `--expected-source-count 241` makes the
denominator check explicit. `scripts/openrouter_promote_route.py` is the only
promotion bridge: it requires a passing costed canary, hashes the artifact,
copies its confidence-bound gates, and sets a finite expiry.

## Measurement contract

The current runner records source-provider rows in the published metrics
collection and already carries protocol/profile and attempt provenance. A
consolidation implementation must extend that provenance before switching any
production lane:

```text
source_provider       existing chart and catalogue provider
source_model_id       existing provider-specific model ID
transport_provider    direct | openrouter
route_model_id        exact OpenRouter model ID, when routed
route_provider_slug   requested OpenRouter provider slug, when routed
observed_provider     provider reported for the completed request
openrouter_response_id response or generation identifier
route_policy          direct | pinned-provider
route_snapshot_at     timestamp of the route evidence
route_probe_id        link to the coverage/canary evidence
route_decision_version version of the route decision
```

The existing `provider` field must not silently change from the source provider
to `openrouter`. Direct and routed measurements must not be averaged together.
The default publication decision is to expose routed transport as a separate
series. Keeping source identity while hiding transport would claim that a
direct-host measurement and an OpenRouter-served build are interchangeable.
Any change to that default is a blocking product decision before implementation.

Reasoning profiles, streaming behavior, hidden-token accounting, validation
policy, and `benchmark_profile_id` remain part of the route compatibility
decision. A route that answers text but changes the benchmark protocol is not a
drop-in replacement.

## Workstreams

### A. Freeze the audit input

- [ ] Export the enabled, non-deprecated `models` rows from production with a
      timestamp and the exact 241-row input.
- [ ] Record provider counts and the current direct adapter for each row.
- [ ] Do not use the stale `openrouter_catalog` collection as the sole source
      of truth. Record its freshness and field completeness in the report.
- [ ] Keep the input snapshot immutable for the duration of the audit so a
      changing catalogue cannot alter the denominator.

### B. Resolve exact OpenRouter coverage

- [ ] Fetch and persist an OpenRouter model snapshot with request timestamp,
      response metadata, and pagination/completeness fields.
- [ ] Resolve candidate model IDs using exact IDs first. Use display names or
      LLM assistance only to propose candidates for ambiguous rows; never use a
      fuzzy proposal as an automatic route.
- [ ] Query the endpoint listing for every candidate model ID and persist the
      provider name, route provider slug, supported parameters, context length,
      and pricing needed for the decision.
- [ ] Maintain an explicit mapping for provider names that differ between the
      bench and OpenRouter, such as Vertex endpoint slugs. Do not infer these
      mappings from a shared substring.
- [ ] Produce one evidence row per enabled model with:
      `provider`, `model_id`, `display_name`, `or_model_id`,
      `route_provider_slug`, `status`, `reason`, `snapshot_at`, and
      `evidence_refs`.
- [ ] Report `route-or`, `keep-direct`, `unsupported`, `protocol-incompatible`,
      `transient`, and `unknown` separately. `unknown` is a direct decision,
      not a reason to wait for David.

### C. Validate routes with bounded probes

- [ ] Use the same request shape as the production benchmark profile, including
      streaming, token limits, reasoning controls, and retries.
- [ ] Restrict the request to the intended OpenRouter provider slug and send:
      `provider.only`, `allow_fallbacks: false`, and
      `require_parameters: true`.
- [ ] Capture the effective request after adapter transformations, including
      API surface, token-cap field, stream mode, reasoning controls, retries,
      and parameter requirements.
- [ ] Request OpenRouter routing metadata with `X-OpenRouter-Metadata: enabled`
      or use the documented generation-metadata lookup. Capture the observed
      provider and response ID, then normalize the observed display identity
      through a reviewed mapping. If identity is absent or mismatched, fail
      closed to `keep-direct`.
- [ ] Use a bounded probe budget and concurrency limit. Charge unknown outcomes
      pessimistically and record every request outcome.
- [ ] Require successful output, valid usage/finish data, and the expected
      provider restriction and observed provider before marking a row covered.
- [ ] Separate route availability probes from measurement canaries. Availability
      probes answer whether the route works; they do not establish throughput
      equivalence.
- [ ] Repeat enough times to distinguish a route mismatch from a transient
      provider error. The initial recommendation is two successful availability
      probes on separate requests, with a third probe for any disagreement.
- [ ] Never enqueue a probe into the published metrics series. Probe results
      belong in a separate evidence collection or artifact.

### D. Define the route map

- [ ] Choose the durable route record shape and unique key. It must support
      multiple historical route decisions for one source row.
- [ ] Store the decision, evidence timestamp, exact model ID, provider slug,
      request compatibility, and expiry/recheck time.
- [ ] Keep direct credentials and adapters configured for every source provider
      while the route map is being tested.
- [ ] Define the OpenRouter outage, billing, rate-limit, and route-error
      behavior. The first failure must not delete the source row.
- [x] Add an explicit route health state so a bad route can fall back to direct
      without a manual catalogue edit.
- [x] Treat direct recovery as a separately logged application-level attempt.
      Never combine partial routed output and a direct retry into one sample.

### E. Canary and rollout

- [ ] Run the audit in report-only mode against the frozen snapshot containing
      241 rows on 2026-08-09, then produce a delta report for any newly enabled
      rows before rollout.
- [ ] Review the per-provider totals and every `route-or` decision, with special
      attention to dated models, aliases, reasoning models, and regional IDs.
- [x] Canary a small set of non-Bedrock rows with paired, randomized direct and
      pinned-OpenRouter transport evidence. Predeclare sample counts and
      thresholds for output validity, generated/visible throughput, TTFT,
      errors, cost, and observed provider.
- [ ] Keep direct-versus-routed canary comparison separate from route
      availability. A pinned route can be available while serving a materially
      different build from the direct API.
- [x] Promote only rows that pass the canary gate. Keep direct as an immediate
      rollback path.
- [ ] Expand in bounded batches and stop promotion when route health or model
      coverage falls below the pre-switch baseline.
- [ ] Verify the dashboard and health checks distinguish source provider from
      transport provider and do not merge incompatible protocol rows.

The guarded rollout procedure is:

1. Write a route decision with `canary_state=availability_passed` or another
   non-passed state. The scheduler will attach it to jobs, but the runner will
   keep the direct lane.
2. Run a paired, randomized direct-versus-pinned canary outside the published
   metrics collection. Record the direct and routed attempts under one
   `canary_id`, with separate transport fields and the same benchmark profile.
3. Promote only after the predeclared success count and thresholds pass by
   running `scripts/openrouter_promote_route.py`. It writes
   `canary_state=passed`, verified cost, the evidence hash, confidence bounds,
   `canary_successes`, `canary_required_successes`, and a finite expiry. The
   next scheduled jobs freeze that evidence; they do not reinterpret already
   queued snapshots.
4. Roll back by changing the route decision to `canary_state=rollback` or
   `state=direct`. New jobs then resolve direct. Existing routed jobs fail
   closed when their snapshot expires or is otherwise invalid, and every route
   failure has a separately logged `route_*` error before direct recovery.

An automated operator transition is available through
`scripts/openrouter_route_health.py`: a route failure can enter `state=cooldown`
with a bounded `cooldown_until`, and only an explicit recovery probe can return
it to `state=active`. Both transitions preserve the original evidence and can
be applied only with the same explicit `--apply --yes` write guard.

OpenRouter requests use a route-specific client timeout and all routed source
lanes share the scheduler process's `OPENROUTER_CONCURRENCY` gate. The gate is
not a replacement for a deployment-wide quota service when multiple scheduler
processes are run; that remains a deployment constraint before scaling out.

### F. Ongoing reconciliation

- [ ] Refresh OpenRouter model and endpoint evidence on a schedule with a
      complete-run ledger.
- [ ] Recheck routes when a model ID, endpoint list, supported parameter set,
      or benchmark profile changes.
- [x] Expire stale route evidence conservatively. Staleness keeps a row direct;
      it never silently broadens routing.
- [ ] Report route additions, removals, fallback events, and direct-provider
      coverage in the existing health/digest path.
- [ ] Preserve an audit trail for every route mutation and make the operation
      reversible.

### G. Required implementation prerequisites

- [x] Add source/transport dispatch to the scheduler and runner. A route map
      that the runner cannot interpret is not a usable fallback.
- [x] Repair the OpenRouter adapter's token accounting. Streaming chunk count
      is not token count; use provider-reported usage where available and
      persist generated, visible, and reasoning tokens separately.
- [x] Capture finish reason, response status, response ID, effective request,
      observed provider, and the reviewed provider-identity mapping for routed
      samples.
- [x] Add every route provenance field to the Mongo logging allowlist and add a
      persistence test through `log_mongo`.
- [x] Add route/profile-specific concurrency, spend limits, health hysteresis,
      cooldown, and recovery probes. Per-source lanes must not bypass a shared
      OpenRouter quota.
- [x] Version route decisions and attach the version to queued jobs so a route
      change cannot reinterpret already queued work.
- [x] Add an invariant that every enabled source row remains schedulable when a
      route record is missing, stale, or invalid. The result must be direct.
- [ ] Add an invariant that models enabled after the audit snapshot default to
      direct and enter the next reconciliation pass.
- [x] Add scheduler, metrics, error-schema, dashboard grouping, and no-model-loss
      tests before any production route is promoted.

## Acceptance criteria

The epic is ready for implementation when all of the following are true:

- [x] The 241-row audit input and provider totals are reproducible.
- [x] Every row has exactly one conservative route decision and evidence.
- [ ] No `route-or` decision relies only on name similarity or catalog presence.
- [ ] Bedrock is excluded from route-replacement candidates in this epic;
      account-backed OpenRouter/Bedrock routing is not configured or in scope.
- [ ] Vertex remains direct unless a separate transport/build comparison proves
      a routed measurement is suitable for a distinct published series.
- [ ] Every source provider and model remains represented and recoverable.
- [ ] The route data model preserves source identity, transport identity,
      observed provider identity, and historical metric compatibility.
- [ ] The OpenRouter adapter uses provider-reported usage or an equivalent
      authoritative token source; chunk counts are never published as tokens.
- [x] Missing, stale, invalid, or mismatched route evidence defaults to direct.
- [ ] Probe costs, concurrency, retries, and unknown outcomes are bounded.
- [x] A canary and rollback procedure is written and testable.
- [x] A no-model-loss test enumerates every enabled source row and proves that
      absent, stale, invalid, or mismatched route records select direct
      transport.
- [ ] The implementation plan names the runner, Mongo schema, scheduler,
      dashboard, health checks, and deployment surfaces that will change.
- [x] Hatch Sol has reviewed this epic and actionable findings are integrated.

## Explicit non-goals

- Replacing all direct providers with OpenRouter in one change.
- Disabling Bedrock, Vertex, or any other provider based on model-family
  similarity.
- Treating OpenRouter's 400-row public response as a complete proof of global
  service coverage.
- Rewriting published metrics or merging direct and routed samples silently.
- Adding an indefinite human review queue for ambiguous model matches.
- Implementing the route map or scheduler changes as part of this drafting
  pass.

## Decisions still required before implementation

1. The default is a separate published routed-transport series. If routed
   samples should retain the source-provider chart line, that decision must be
   made explicitly and must still expose transport and observed-provider
   provenance.
2. What minimum canary window and success count are sufficient for each route?
   The initial proposal is two successful probes plus a third on disagreement,
   followed by a bounded production canary.
3. What route evidence age requires revalidation? A stale route remains direct
   until refreshed.
4. Which reviewed provider-slug mappings should be accepted, especially for
   Vertex and provider variants? A mapping must not claim that the OpenRouter
   build is identical to the direct provider's build.

## Review gate

Hatch Sol's implementation review on 2026-08-10 found and required the
fail-closed promotion evidence, mandatory expiry, paired confidence bounds,
overall canary deadline, and 241-row preservation test. Those findings are
integrated in the route resolver, canary runner, promotion command, and test
suite. The final review is a second independent Hatch Sol pass focused on:

- false positives that could lose a provider or merge incompatible metrics;
- missing OpenRouter endpoint and fallback semantics;
- reasoning, streaming, and token-accounting incompatibilities;
- rollout, rollback, spend, and health-check gaps; and
- whether the task list is small enough to implement safely.

No production routing change is in scope until that final review is integrated.
