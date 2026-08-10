# OpenRouter full model equivalence and migration plan

Status: implementation and evidence pass complete; production promotion pending
Owner: LLM Bench
Started: 2026-08-10

## Goal

For every enabled LLM Bench source row, determine whether OpenRouter can serve
the same provider/model combination with the same benchmark protocol. Migrate
every row that passes the identity, compatibility, availability, performance,
cost, and health gates. Keep the existing direct adapter for every unresolved,
incompatible, failed, or policy-excluded row.

The unit of work is one source `(provider, model_id)` row. The target is not
"route all 241 rows". The target is "make a defensible decision for all 241
rows, and route every row that has enough evidence to be a valid replacement."

There are two separate equivalence questions:

1. **Model identity:** does the OpenRouter ID represent the same model build,
   version, family, modality, and parameter class as the source row?
2. **Transport compatibility:** can the OpenRouter request use the same
   benchmark profile with acceptable output, latency, error, cost, and provider
   identity?

A successful canary answers the second question. It does not prove the first.
Primary model metadata is required together with an exact ID or a versioned
reviewed alias table to establish model identity. Direct and routed
measurements remain separate transport/build series even after a route is
approved.

## Definition of done

This work is complete only when all of the following are true:

- The frozen enabled-model input is reconciled row-for-row with the final
  decision report. The counts sum exactly to the input count.
- Every row has one terminal decision and a reason. No row is silently dropped
  because a catalog endpoint was incomplete or a reviewer failed.
- Every `route-approved` row has an exact OpenRouter model ID, a reviewed
  provider mapping, primary model-identity evidence, endpoint evidence, a
  successful pinned-provider probe, and a passing paired canary.
- Every route decision records the source identity, transport identity,
  observed provider identity, evidence references, protocol compatibility, and
  expiry/recheck time.
- Every passing route is materialized in the route map and can be activated in
  bounded batches. Every non-passing row remains direct.
- Direct credentials and adapters remain available for every source row.
- A route failure, stale record, billing error, quota error, or provider
  mismatch returns the row to direct transport without deleting its source
  identity or historical metrics.
- The dashboard keeps direct and routed transport series separate and lifecycle
  status remains keyed by transport.
- The complete implementation, artifacts, and tests pass independent review by
  Hatch Sol and Cursor Grok.
- Every artifact used for a decision is hash-addressed and retrievable from
  the declared artifact store.

## Current baseline

The frozen planning input contains 241 enabled, non-deprecated rows across the
source providers. The previous pass found 56 possible matches in a limited
400-row OpenRouter discovery response. Those 56 are candidates, not validated
migrations. The other 185 are unresolved against that response, not proven
failures. This plan replaces that limited catalog-membership test with a
complete evidence pipeline.

The previous canary validated one route, `openai/gpt-4o-mini`, with 30 paired
requests. It showed approximate parity, not a general OpenRouter performance
claim. Production routing remains opt-in until this plan's gates pass.

The prior `/api/v1/models` response had `total_count=400`, 400 returned rows,
and no next link. That is complete according to the response's pagination
signals, but it may still be a limited public discovery view rather than a
global service catalog. The audit records both facts separately:
`pagination_complete` and `catalog_scope`. A complete discovery view is not
evidence that an absent model is globally unservable.

## Implementation result, 2026-08-10

The workflow is implemented and exercised against the frozen 241-row source
snapshot. The live run is deliberately conservative:

- The public OpenRouter discovery response contained 400 rows and did not
  establish global catalog completeness. Unseen IDs therefore remain
  `direct-unknown`; they are not treated as proven no-match.
- The reviewed alias table contains 56 candidate source rows. Cursor Grok and
  Hatch Sol independently reviewed all 56 and returned no rejected mappings.
  Each alias now resolves its evidence labels through a hash-addressed
  manifest and its reviewer names through durable Hatch receipts.
- The pinned availability probe scheduled 59 candidates and observed 55
  successful provider-identified responses. A successful probe alone did not
  activate a route.
- One paired 30-request canary passed for `openai/gpt-4o-mini` and was
  materialized as an active dry-run route. Its throughput ratio was 0.987,
  TTFT ratio 1.003, cost ratio 1.000, with zero errors.
- The final 241-row reconciliation has 1 `route-approved` row and 240 direct
  rows: 21 policy-excluded, 5 incompatible, 2 probe-failed, and 212 unknown.
  The unknown category includes rows hidden by the limited public catalog and
  candidates that have not yet received the required costed paired canary.
- No production Mongo reconciliation was applied because this run had no
  configured `MONGODB_URI`. The active route artifact is therefore a
  reviewable dry-run artifact, not a production mutation.
- Probe and canary commands now require a shared UTC-day reservation ledger,
  enforce a per-batch cap, and include input-token cost in the canary estimate.
  Reconciliation accepts a previous run and records new, changed, stale, and
  removed source keys.

The immutable v2 evidence bundle is described by
`artifact.manifest.v2.json` and stored under
`artifacts/llm-benchmarks/openrouter-consolidation/v2/`. It includes the frozen
inputs, alias reviews, audit, probes, canary, pricing, active route, final
decisions, and reconciliation report. The exact row-level states and hashes
are in `docs/audits/openrouter-migration-live-2026-08-10.md`.

## Row decision states

Every source row ends in exactly one of these states:

| State | Meaning | Runtime lane |
| --- | --- | --- |
| `route-approved` | Exact identity, compatibility, probe, canary, and health evidence all pass | OpenRouter with direct fallback |
| `direct-no-match` | No credible OpenRouter identity was found | Direct |
| `direct-ambiguous` | Candidates exist but evidence does not establish equivalence | Direct |
| `direct-incompatible` | The model is found, but protocol or benchmark semantics differ | Direct |
| `direct-probe-failed` | The pinned route was unavailable, mismatched, or unusable | Direct |
| `direct-canary-failed` | The route works but fails performance, error, cost, or output gates | Direct |
| `direct-policy-excluded` | Explicit policy excludes the transport, such as current Bedrock scope | Direct |
| `direct-unknown` | Evidence collection could not finish; the row is still safe and direct | Direct |

`route-candidate` is an intermediate pre-canary state only. It cannot activate
a route and must not appear in the final 241-row terminal decision report.

## Evidence contract

The final evidence row contains at least:

```text
source_provider
source_model_id
display_name
source_catalog_snapshot
or_model_id
or_catalog_snapshot
or_endpoint_evidence
model_identity_evidence
model_identity_rule_version
route_provider_slug
observed_provider_slug
provider_identity_mapping_version
protocol_compatibility
benchmark_profile_id
benchmark_profile_hash
effective_request_hash
direct_effective_request_hash
routed_effective_request_hash
probe_ids
canary_id
canary_state
canary_metrics_and_confidence_bounds
pricing_snapshot
pricing_status
decision_state
decision_reason
evidence_uri
evidence_sha256
created_at
expires_at
recheck_at
review_trace_ids
```

The evidence artifact is immutable. The route map references the artifact by
URI and SHA-256 rather than copying an unverified summary.

### Primary identity evidence

Primary identity evidence is one of:

- a source-provider model metadata response or checked-in provider model
  manifest containing canonical ID, family, version/build, modality, and
  parameter class;
- an official model card or provider documentation referenced by a hash-pinned
  URL; or
- an exact source-provider ID plus an OpenRouter model record whose canonical
  author/slug, version/build, family, modality, and parameter class all match.

The evidence row stores the source URL or artifact path, retrieval time,
selected fields, and SHA-256. An exact string match without this metadata is a
candidate only. A versioned alias must live in
`docs/specs/openrouter-model-aliases.v1.json`, with source ID, target ID,
provider mapping, supporting evidence references, reviewer IDs, and a rule
version. The table is the only approved way to turn a non-exact ID into a
candidate. A unique slug match is diagnostic output only and is explicitly
retired as an approval path.

## Workflow

### 1. Freeze the source input

- Export enabled, non-deprecated `models` rows with timestamp and provider
  counts.
- Assign a stable `source_row_id` to each `(provider, model_id)` pair.
- Store the exact input as an immutable artifact.
- Fail the run if the source count, uniqueness, or provider totals do not match
  the declared input.

### 2. Acquire the OpenRouter evidence set

- Fetch the OpenRouter model catalog and record request time, response headers,
  pagination, total count, and completeness signals.
- Fetch endpoint listings for candidate model IDs and persist provider names,
  provider slugs, pricing, context length, supported parameters, and endpoint
  status.
- Determine whether the public catalog is complete. If it is not complete,
  label it as a discovery view and do not call absence a global no-match.
- Use this executable completeness predicate: HTTP success, a valid `data`
  list, `total_count == len(data)`, no `links.next`, stable repeated count, and
  a recorded `catalog_scope` of `public-discovery` or `global`.
- If the response is incomplete, retry pagination before classification. If it
  remains incomplete, the row may be `direct-unknown`, never
  `direct-no-match`.
- An unknown OpenRouter ID cannot be probed. Probes only run for a concrete ID
  produced by an exact match or an allowlisted alias/transform.
- Preserve raw responses. Derived summaries are not the source of truth.

### 3. Deterministic normalization and candidate generation

Process all 241 rows in one deterministic batch. This stage may:

- normalize case and whitespace;
- split organization prefixes only on `/`;
- strip known Bedrock region and provider prefixes;
- apply only versioned, reviewed alias and transform rules. Generic removal of
  `-instruct`, dates, `:batch`, `:nitro`, or `-fast` is forbidden unless an
  explicit rule names the source family and target ID;
- preserve meaningful version dots such as `2.5`, `3.2`, and `0.2`;
- apply reviewed provider alias tables; and
- rank exact, canonical, alias, and token-similar candidates.

Fuzzy similarity may rank evidence for a reviewer, but it cannot generate an
approval candidate by itself. Candidate generation must retain the raw source
ID, the exact or allowlisted rule that produced each candidate, and every
candidate ID so a reviewer can see what was considered. No invented OpenRouter
IDs are allowed.

`unique_slug_candidates` in the pilot audit is retained only to explain prior
coverage results. The full pipeline must not pass those candidates to endpoint
fetching, probing, or materialization unless an allowlisted identity rule also
selects them.

### 4. Structured ambiguity review

Only rows without a deterministic exact identity or reviewed alias go to
model-assisted review. The reviewer receives raw evidence, not only a
similarity score. It must
return structured JSON with:

```text
match_status: exact | probable | ambiguous | no_match
selected_or_model_id
selected_provider_slug
confidence
supporting_facts[]
contradicting_facts[]
required_followup[]
```

Rules:

- Reviewers may select from supplied candidates or return `no_match`.
- Reviewers may not invent an OpenRouter ID or write a route decision.
- A `probable` result requires two independent reviewers to agree on the same
  ID and rule, plus primary model metadata supporting the identity.
- Any disagreement, hard constraint mismatch, missing primary evidence, or low
  confidence remains direct with `direct-ambiguous` or `direct-unknown` and a
  `recheck_at`. There is no indefinite human review queue.
- Reviewer prompts and outputs are stored with the row evidence.

### 5. Hard compatibility checks

Before any network probe, deterministic checks reject candidates with known
incompatibilities:

- wrong model family or organization;
- a version, date, parameter-size, or modality mismatch;
- incompatible context or output limits;
- missing streaming, reasoning, token usage, or response fields required by
  the benchmark profile;
- unsupported provider restriction; or
- a policy exclusion such as Bedrock's current direct-only scope.

These checks reduce probe spend. They do not turn a fuzzy score into proof.

### 6. Pinned availability probes

For every surviving candidate:

- send the production-shaped request through the OpenRouter adapter;
- use the exact OpenRouter model ID;
- restrict with `provider.only`, disable fallbacks, and require parameters;
- capture response ID, usage, finish state, effective request, and observed
  provider metadata;
- require usable output and the expected provider identity; and
- run at least two successful probes on separate requests, with a third when
  results disagree.

Probe results are stored outside published benchmark metrics. A successful
probe establishes availability and identity, not equivalence of performance.

### 7. Paired canaries

Every candidate that passes probing receives a paired direct/OpenRouter canary
using the same benchmark profile. The default gate is 30 pairs, balanced 15
direct-first and 15 OpenRouter-first, with deterministic order recording.

The profile, adapter version, direct effective request, routed effective
request, and pricing inputs are immutable and hashed separately. The canary is
a transport compatibility gate. It does
not authorize merging the routed series into direct history or claiming that
the two serving builds are identical.

Promotion requires:

- at least 29 successful pairs;
- route metadata verified for every routed attempt;
- route error delta no worse than 0.05;
- lower 95% confidence bound for throughput at least 0.80;
- upper 95% confidence bound for TTFT no greater than 1.50;
- upper 95% confidence bound for cost no greater than 1.10;
- valid provider-reported usage and output; and
- immutable canary evidence with pricing snapshot and SHA-256.
- direct and OpenRouter pricing are both present and independently sourced.

The thresholds are gates, not claims that OpenRouter is faster. A route that
fails any gate remains direct with the measured failure reason.

### Budgets and fallback reserve

No discovery or canary run may start without explicit budgets. The initial
defaults are:

- at most 200 availability probes per run;
- at most 10 routes in one canary batch;
- at most 30 pairs per route;
- a `$50` total audit/canary spend cap per batch;
- a `$5` per-route spend cap; and
- a daily cap enforced by the shared OpenRouter quota ledger.

The command stops before a request that would exceed a cap and records
`direct-unknown` with `budget-exhausted`. A routed attempt reserves at least
`max(10 seconds, 25% of the job deadline)` for direct recovery. If the route
cannot acquire quota within the remaining budget, the worker uses the direct
lane when that reserve is available; otherwise it records a bounded timeout
and keeps the next retry direct.

### 8. Promotion and bounded rollout

- Materialize one route decision per source row.
- Promote only from candidate to active through the guarded promotion command.
- Set a finite expiry and recheck time on every active route.
- Activate approved routes in bounded provider batches.
- Monitor routed error rate, fallback count, quota, cost, freshness, and
  provider identity.
- Pause the batch when health falls below the direct baseline.
- Roll back by changing state to direct or cooldown. New jobs resolve direct;
  queued jobs are checked against the route revocation generation before
  dispatch. Already running jobs have a maximum drain window of their frozen
  deadline plus lease grace; they cannot start a new routed attempt after
  revocation.

The runtime switch remains explicit and opt-in. No direct adapter or source
catalogue row is removed as part of migration.

This plan supersedes two pilot behaviors. The worker must reserve direct
fallback time instead of waiting the entire job deadline on the OpenRouter
semaphore, and the dispatch path must consult route revocation state before a
frozen queued snapshot is allowed to start. The normal route snapshot remains
immutable; revocation is an emergency safety override with its own evidence.

### Revocation schema

Each active route carries an integer `route_revocation_generation`, initially
zero. A `bench_route_revocations` record is keyed by
`(source_provider, source_model_id, generation)` and contains `revoked_at`,
`reason`, `operator`, and an evidence hash. The job snapshot copies the active
generation at enqueue time. Before dispatch, the worker reads the latest
revocation generation for that source row. If it is greater than the snapshot
generation, the job resolves direct and records `route-revoked`; it does not
reinterpret the rest of the frozen snapshot. A running child may drain only
until its original deadline plus lease grace. No new routed attempt starts
after revocation.

### 9. Reconciliation

Run the reconciler on a schedule and on catalog changes. It must:

- detect newly enabled rows and default them to direct;
- detect changed model IDs, endpoints, pricing, supported parameters, or
  provider mappings;
- expire routes whose evidence is stale;
- re-probe and re-canary changed routes; and
- produce a delta report for additions, removals, demotions, and recoveries.

The reconciler owns these durable records:

- `bench_route_reconciliation_runs`: run ID, source snapshot hash, OpenRouter
  catalog hash, alias-rule version, profile hash, start/end, status, budgets,
  and row counts;
- `bench_route_decisions`: one historical decision per source row and audit
  snapshot, including terminal state and evidence hash; and
- the artifact manifest: raw catalogs, endpoint responses, reviewer traces,
  probes, canaries, and the final report.

The scheduled job runs daily and on a detected catalog or source-model change.
Newly enabled rows default direct until a complete pass creates evidence.
Apply is idempotent on `(source_provider, source_model_id, audit_snapshot)`.

## State mapping

The report taxonomy and runtime state are deliberately mapped rather than
maintained as separate vocabularies:

| Evidence decision | Route document | Runtime result |
| --- | --- | --- |
| `route-candidate` | `state=candidate`, `canary_state=availability_passed` | Direct |
| `route-approved` before activation | `state=candidate`, `canary_state=passed` | Direct until explicit activation |
| `route-approved` active | `state=active`, `canary_state=passed` | OpenRouter with direct fallback |
| any `direct-*` state | `state=direct` with `reason_class` | Direct |
| route failure | `state=cooldown` with failure evidence | Direct |
| revoked route | `state=direct` or `state=revoked` | Direct, including queued-job revocation check |

`RouteDecision.from_snapshot` is the final enforcement point. Missing,
malformed, expired, revoked, or mismatched records resolve direct.

### Audit-to-runtime bridge

The existing audit vocabulary is retained as an input compatibility layer:

| Existing audit output | Full-plan decision | Materializer behavior |
| --- | --- | --- |
| `route-or` with verified probe | `route-candidate` | Candidate only; never active without canary promotion |
| `keep-direct` + `bedrock-out-of-scope` | `direct-policy-excluded` | Direct |
| `keep-direct` + `no-exact-or-ambiguous-model-id` and `catalog_scope=global` | `direct-no-match` | Direct |
| `keep-direct` + no-ID reason and `catalog_scope=public-discovery` or unknown | `direct-unknown` | Direct with recheck |
| `keep-direct` + ambiguous reason | `direct-ambiguous` | Direct |
| `keep-direct` + protocol reason | `direct-incompatible` | Direct |
| `keep-direct` + `source-provider-not-listed` | `direct-incompatible` | Direct |
| `keep-direct` + `endpoint-evidence-missing` | `direct-unknown` | Direct with recheck |
| `keep-direct` + probe, observed-provider, or output reason | `direct-probe-failed` | Direct |
| `keep-direct` + unknown or budget reason | `direct-unknown` | Direct with recheck |

The extended materializer accepts this bridge and emits the full-plan terminal
state alongside the legacy fields. The promotion command remains the only
place that can change a candidate into an active route.

### Candidate finalization

After probes, every `route-candidate` is finalized exactly once:

- a passing paired canary becomes `route-approved` and is eligible for guarded
  activation;
- a failed performance, error, output, or cost gate becomes
  `direct-canary-failed` with the raw canary evidence; and
- a missing canary, exhausted budget, or incomplete evidence becomes
  `direct-unknown` with a `recheck_at`.

The final report is generated only after this step, so its 241 rows contain no
intermediate `route-candidate` state.

## Delegation model

This is a small pipeline with a large deterministic surface and a narrow
judgment surface.

### Work kept deterministic

- catalog fetching and raw artifact storage;
- pagination and completeness checks;
- ID normalization;
- candidate scoring and hard compatibility checks;
- API probes and canaries;
- schemas, thresholds, hashing, expiry, retries, and route activation;
- row-count reconciliation; and
- production fallback behavior.

The existing `openrouter_coverage_audit.py` and `openrouter_route_probe.py`
remain the base surfaces for catalog joins and probes. New code extends them
only where a required artifact, completeness check, budget, or state is
missing. The plan does not create a second parallel matcher or route state
machine.

### Work delegated to bounded reviewers

- ambiguous identity adjudication;
- provider alias interpretation when raw metadata conflicts; and
- adversarial review of proposed matches.

The ambiguity set should be divided into provider or family batches, not one
agent per row. Each batch has a fixed input, output schema, cost cap, and trace.
The main process owns all writes and rejects malformed or contradictory review
outputs.

Recommended reviewer arrangement:

1. One deterministic pass over all 241 rows.
2. Parallel Cursor Grok reviews for independent ambiguous batches.
3. A second reviewer only for probable matches and disagreements.
4. Hatch Sol as the final adversarial reviewer of the route evidence and
   migration code, not as an unbounded row-by-row classifier.

If the ambiguous set is small, review it in one bounded run. If it is large,
improve the catalog and alias evidence before increasing agent count.

## Code and artifact surfaces

The implementation should keep matching mostly offline and keep production
runtime small.

- `scripts/openrouter_coverage_audit.py`: extend the existing raw catalog,
  completeness, allowlisted-alias, and row-decision pipeline.
- `scripts/openrouter_route_probe.py`: extend the existing pinned probe with
  budgets, profile hashes, pricing, and immutable evidence.
- `scripts/openrouter_review_ambiguities.py`: schema-checked reviewer packets
  and trace capture.
- `scripts/openrouter_route_decisions.py`: one decision per source row.
- `scripts/openrouter_paired_canary.py`: paired evidence and confidence bounds.
- `scripts/openrouter_promote_route.py`: the only candidate-to-active bridge.
- `scripts/openrouter_reconcile.py`: scheduled refresh and delta report, built
  on the existing coverage and decision artifacts.
- `api/llm_bench/scheduler/routing.py`: fail-closed runtime resolution.
- `api/llm_bench/scheduler/runner.py` and `worker.py`: transport execution,
  quota, timeout, fallback, and health behavior.
- `scripts/mongo_indexes.js`: route and lifecycle indexes with migration.
- Dashboard mapping and lifecycle utilities: transport-keyed publication.
- `artifact.manifest.json`: immutable evidence inventory.

## Acceptance checklist

### Coverage and identity

- [ ] Source input contains exactly 241 unique enabled rows, or the reconciler
      declares and records a changed count.
- [ ] Every source row has exactly one terminal decision.
- [ ] Every `route-approved` row has an exact `author/slug` and reviewed
      provider mapping.
- [ ] No route is approved from fuzzy similarity, display name, or catalog
      membership alone. Every non-exact match cites a versioned allowlisted
      alias/transform and primary identity evidence.
- [ ] Partial catalog absence is reported separately from proven no-match.
- [ ] Catalog completeness, discovery scope, source snapshot hash, alias-rule
      version, and profile hash are recorded in the reconciliation run.

### Runtime and evidence

- [ ] Every approved route passes pinned probes with observed provider identity.
- [ ] Every approved route passes the paired canary gate and has immutable
      pricing, metrics, confidence bounds, and expiry evidence.
- [ ] Both direct and OpenRouter pricing are present, source-attributed, and
      hash-pinned before a cost gate can pass.
- [ ] Missing, stale, malformed, mismatched, or expired evidence resolves
      direct.
- [ ] Route errors, quota failures, and timeouts enter cooldown before retry,
      with a reserved direct-fallback budget and explicit tests.
- [ ] Direct fallback remains available and separately logged.
- [ ] Direct and routed metrics and lifecycle records remain isolated.
- [ ] Revoked route generations prevent queued jobs from starting a routed
      attempt, and the maximum running-job drain window is tested.

### Migration and operations

- [ ] All passing routes are materialized and activated in bounded batches.
- [ ] Probe and canary commands enforce per-route, per-batch, and daily USD and
      request-count budgets before making requests.
- [ ] No passing route is omitted without a recorded operational reason.
- [ ] All non-passing rows remain benchmarkable through direct adapters.
- [ ] Reconciliation has an idempotent run record and detects new, changed,
      stale, and removed routes.
- [ ] Rollback is tested without deleting source rows or historical metrics.

### Review and release

- [ ] API tests, dashboard tests, type checks, lint, artifact verification,
      and route-map reconciliation pass.
- [ ] Hatch Sol returns `READY` on the implementation and evidence.
- [ ] Cursor Grok returns `READY` on the implementation and evidence.
- [ ] The final report lists every row's state and the exact count routed,
      direct, ambiguous, incompatible, failed, excluded, and unknown.
- [ ] The final report distinguishes model identity proof from transport
      compatibility and does not merge direct and routed builds.

## Release report

The final report must answer these questions plainly:

1. How many of the 241 rows had a valid, proven OpenRouter equivalent?
2. How many were migrated and are active?
3. How many remain direct, and why for each category?
4. Which routes failed probing or canary, and what evidence supports that?
5. Did the migration change benchmark coverage, cost, latency, or output
   semantics?
6. Can every active route be rolled back without losing a source provider?

The headline number is the count of `route-approved` rows, not the count of
name-similar candidates.

## Review findings incorporated

The initial independent reviews found the draft not ready. This plan
incorporates their blockers:

- identity proof is now separate from transport compatibility and requires
  primary evidence plus versioned aliases;
- catalog completeness and public discovery scope are recorded explicitly;
- unknown IDs are never probed and incomplete catalogs produce
  `direct-unknown`;
- direct and OpenRouter pricing, profile hashes, and effective-request hashes
  are required;
- probe, canary, per-route, batch, and daily budgets are bounded;
- the reconciler has named inputs, outputs, collections, and idempotency;
- evidence states map explicitly to existing route states;
- queued-job revocation and a direct-fallback reserve are part of rollout; and
- ambiguity is resolved by a second evidence pass or remains direct, without
  an indefinite human queue.

Final plan review receipts:

- Hatch Sol: `hatch_20260810T185345.257155000Z_8af2fbb944ad96a2`, `READY`.
- Cursor Grok: `hatch_20260810T185412.164616000Z_b1da3ccb6be50963`, `READY`.
