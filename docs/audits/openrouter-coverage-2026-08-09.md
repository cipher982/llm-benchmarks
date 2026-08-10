# OpenRouter coverage audit

Run date: 2026-08-09
Mode: report-only
Source snapshot: 241 enabled, non-deprecated model rows
OpenRouter catalog snapshot: 400 models
Probe spend: approximately $0.0561

The audit resolved 83 unique candidate model IDs from the snapshot and probed
77 source rows twice, covering 69 unique IDs. Fourteen additional IDs came from
ambiguous candidate sets; they were not independently eligible without an
explicit mapping.

This audit made no MongoDB, scheduler, catalogue, or production routing
changes. It only fetched OpenRouter model and endpoint metadata and sent bounded
streaming probes with provider restrictions, disabled fallbacks, required
parameters, and routing metadata enabled.

## Result

| Decision | Rows |
| --- | ---: |
| `route-or` | 56 |
| `keep-direct` | 185 |
| **Total** | **241** |

`route-or` means two successful probes for the exact OpenRouter model ID, the
reviewed provider slug, usable visible output, and matching observed-provider
metadata. It is availability evidence, not proof that OpenRouter serves the
same build or has the same throughput as the direct API.

## By source provider

| Source provider | Enabled | Route through OR | Keep direct |
| --- | ---: | ---: | ---: |
| anthropic | 10 | 3 | 7 |
| bedrock | 21 | 0 | 21 |
| cerebras | 1 | 0 | 1 |
| deepinfra | 124 | 19 | 105 |
| fireworks | 19 | 0 | 19 |
| groq | 4 | 1 | 3 |
| openai | 36 | 23 | 13 |
| together | 22 | 10 | 12 |
| vertex | 4 | 0 | 4 |
| **Total** | **241** | **56** | **185** |

## Direct decisions

| Reason | Rows | Meaning |
| --- | ---: | --- |
| Bedrock policy | 21 | Account-backed Bedrock routing is outside this epic's configured scope. |
| No exact or unambiguous model ID | 124 | The audit refused fuzzy name matching. This is not proof that OpenRouter cannot serve the model. |
| Source provider not listed | 19 | The candidate model had no matching source-provider endpoint in the current endpoint listing. |
| Visible output empty | 18 | Both 64-token probes spent the budget without visible text, so the default published profile is incompatible. |
| Transient rate limit | 3 | OpenRouter reported upstream 429 overloads. A serial retry still did not produce two successful probes. |

The 18 empty-output rows are mostly reasoning models. They remain direct for
the current profile; they are not marked as permanently unsupported. The three
rate-limited rows remain direct until a later recheck succeeds.

## Method and evidence

The audit tooling is:

- `scripts/openrouter_coverage_audit.py`, which joins the frozen source rows to
  exact catalog candidates and endpoint metadata.
- `scripts/openrouter_route_probe.py`, which runs the bounded streaming probes
  and records usage, finish status, response ID, selected provider metadata,
  visible output, and errors.
- `scripts/openrouter_route_decisions.py`, which materializes one guarded route
  decision per source row. The 56 availability-qualified rows become
  `candidate` records with `canary_state=availability_passed`; all 185 other
  rows become direct records. Applying these records does not activate a
  route, because the scheduler still requires an active record and a passed
  measurement canary.

The temporary raw evidence is retained for this session at:

- `/tmp/our_models_fresh.json`
- `/tmp/or_models_fresh.json`
- `/tmp/or_endpoints_fresh/`
- `/tmp/or_coverage_final_v3.json`
- `/tmp/or_probe_full_v2.json`
- `/tmp/or_probe_retry.json`

The immutable evidence bundle is now preserved under the artifact manifest
[`artifact.manifest.json`](../../artifact.manifest.json), backed by the
`cube-artifacts` MinIO store at version `v1`. The manifest records checksums for
the frozen source inputs, 83 endpoint responses, audit outputs, route decisions,
and the paired canary report.

Re-running the audit requires a new source snapshot and a new OpenRouter
catalog snapshot. The route result must not be copied into production model
configuration. The first paired canary is recorded in
`docs/audits/openrouter-canary-2026-08-10.md`; its output and performance checks
passed, while cost remains unverified.
