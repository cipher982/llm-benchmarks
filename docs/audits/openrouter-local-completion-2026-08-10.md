# OpenRouter local completion

Date: 2026-08-10
Scope: local validation only. Production MongoDB, deployment, and feature flags were not changed.

## Result

The frozen enabled inventory contains 241 source rows. The terminal local reconciliation is:

| State | Rows | Meaning |
| --- | ---: | --- |
| `route-approved` | 6 | Passed identity, endpoint, pricing, pinned probe, and 30-pair canary gates |
| `direct-canary-failed` | 49 | Candidate was tested or explicitly bounded and failed or lacked a local direct credential |
| `direct-policy-excluded` | 21 | Kept direct by policy, including Bedrock |
| `direct-incompatible` | 5 | Protocol or modality mismatch |
| `direct-probe-failed` | 2 | Pinned availability probe failed |
| `direct-unknown` | 158 | No sufficient reviewed evidence for promotion |
| **Total** | **241** | Every row has exactly one terminal state |

The local route map contains only these six source IDs:

- `openai/gpt-3.5-turbo`
- `openai/gpt-4`
- `openai/gpt-4-turbo`
- `openai/gpt-4.1-mini`
- `openai/gpt-4o`
- `openai/gpt-4o-mini`

All other rows retain their original direct lane. Direct fallback is enabled for every route.

## Evidence and canaries

There are 55 candidate canary artifacts, plus the summary file. Six candidates completed 30 successful paired requests. The other 49 are terminal direct decisions:

- 32 could not run the direct half because the corresponding local provider credential was not configured. This is a local evidence limitation, not a claim that OpenRouter cannot serve those models.
- 9 long-running rows were explicitly closed after the bounded local window with `local-canary-timeout-after-bounded-window` evidence. Each failure artifact records its closure timestamp, retry budget, bounded window, and a 24-hour recheck time.
- 8 OpenAI rows completed enough requests to show a failed promotion gate, including performance or successful-pair failures. Their artifacts record the concrete failed gate and derived recheck time.

The approved routes met the canary thresholds. Their route throughput ratios were 0.959 to 1.116, route TTFT ratios were 0.832 to 1.236, and route error rate was zero in all six runs. Pricing was derived from the highest matching provider endpoint rate in the frozen endpoint evidence. That is a conservative same-provider upper bound for this local gate, not an independent proof of OpenRouter billing equivalence.

## Local runtime checks

The local route map resolved six rows to OpenRouter and 235 rows to direct transport. The six active snapshots are marked `terminal_state=route-approved` and point to immutable v3 `derived/canaries/` objects. A forced OpenRouter failure returned the request to the direct lane while preserving the source model ID.

Real local smoke requests passed with `OPENROUTER_ROUTING_ENABLED=1`:

- Routed `gpt-4o-mini`: OpenRouter transport, model `openai/gpt-4o-mini`, two output tokens.
- Direct `gpt-4o-mini`: direct transport, three output tokens.

The full API suite passed: `258 passed in 3.46s`. Focused routing and canary tests also passed during implementation. `uv build` produced both the source distribution and wheel, `git diff --check` passed, and all changed Python files compiled successfully.

## Artifact record

The v3 manifest is stored at:

`s3://artifacts/llm-benchmarks/openrouter-consolidation/v3/manifest.json`

Manifest hash: `d37f5788ae1e3e359cce3e76fcd49943803c6c18015e25c9d7698c90e734aba8`.

All 125 v3 manifest objects were checked at their listed keys, including the manifest itself. The local reconciliation run ID is `reconcile:7d645e6493153d6cbdfe2699`. The route evidence profile hash is `603d20ffe6fbd3282440882b9167777b54d19454310acd2f81daf4d628a04003`.

## Production stop gate

This completion does not authorize production activation. The next production step must explicitly review the six routes, refresh direct and routed price evidence if billing parity is required, then apply the route map through the normal Mongo and deployment workflow. No production write, deployment, or feature-flag change was performed here.

## Review receipts

- Hatch Sol returned READY in run `hatch_20260810T225340.288385000Z_3914ce2ba11b56f0`.
- Cursor Grok returned READY in run `hatch_20260810T225654.179453000Z_55b59b431b9c7963`.
