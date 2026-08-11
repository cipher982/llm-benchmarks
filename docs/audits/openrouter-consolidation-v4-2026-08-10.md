# OpenRouter consolidation v4

Date: 2026-08-10/11. Scope: local validation only. No production MongoDB
write, no deployment, no feature flag change.

## Result

241 frozen source rows, terminal reconciliation `reconcile:e098f045ce40a5e9fa24dc76`:

| State | Rows |
| --- | ---: |
| `route-approved` | 33 |
| `direct-canary-failed` | 59 |
| `direct-probe-failed` | 27 |
| `direct-unknown` | 96 |
| `direct-incompatible` | 5 |
| `direct-policy-excluded` (Bedrock) | 21 |

Routed per provider: anthropic 6/10, deepinfra 19/124, openai 7/36,
vertex 1/4, groq 0/4, together 0/22, fireworks 0/19, cerebras 0/1.

The prior session's v3 pass approved 6 routes (all OpenAI). This pass
approved 33 by removing three pipeline starvation causes, not by weakening
the gates: the alias review covered all 220 non-Bedrock rows against the
full 400-row catalog with per-model endpoint evidence, production
credentials ran the direct canary half for every provider, and canaries ran
as parallel per-candidate processes instead of one serial bounded window.

## Why the other rows stay direct

- `route-serving-reasoning-mismatch` (~20): OpenRouter's lane serves the
  model with thinking enabled and ignores `reasoning` exclusion (qwen3, GLM
  via DeepInfra; claude-opus-5 via Anthropic). Routed visible output is
  empty under the published 64-token profile; direct is not. Routing would
  publish wrong numbers.
- `together-direct-402-no-credit` (13 candidates, and all 22 Together rows
  in production): the Together account is out of credit, so the direct
  canary half cannot run. Production direct Together benchmarks are failing
  with 402 right now.
- `route-tps-below-bound` (~12): measured routed throughput CI below 0.8×
  direct (groq lanes ~0.66-0.68, cerebras 0.58, several gpt-5.x 0.59-0.77,
  claude-fable-5 0.63). Real deficits; the site publishes throughput, so
  these stay direct.
- Reasoning/`-pro` models: not measurable under the published profile in
  either lane (shadow-profile models).
- `direct-unknown` (96): no catalog identity on OpenRouter's 400-row
  discovery surface (mostly DeepInfra's long tail).
- vertex gemini-2.5-flash / -pro: the **direct** lane intermittently emits
  zero visible tokens at 64 tokens, dropping successful pairs below 29/30.
  A direct-lane measurement problem, not a routing one.
- vertex gemini-2.5-flash-image: no routable OpenRouter endpoint (404).

## Code shipped (commits 1f4bd3b..429663c)

- Tag-aware provider matching in the coverage audit (OpenRouter moved the
  machine-readable slug into `tag` for Google Vertex).
- Canary evidence streams accumulate independently; TTFT bound waived only
  when the direct lane measures TTFT on zero pairs, recorded as
  `ttft_waived_direct_unmeasured` and enforced fail-closed in
  `RouteDecision`.
- `route_reasoning_exclude` per-route snapshot flag: routed requests send
  `reasoning {exclude, effort minimal}` only for routes validated with it
  (claude-fable-5 attempt, vertex gemini). Opt-in, recorded in evidence.
- `Google` display-name resolution via expected pinned slug.
- v2 reviewed alias spec (130 mappings, dual hatch receipts) at
  `docs/specs/openrouter-model-aliases.v2.json`.

## Evidence

`s3://artifacts/llm-benchmarks/openrouter-consolidation/v4/` — manifest
sha256 `6adbb41c2f60278bb050c7d8da76e175a5d103d684e7a0d606b93513b4c671bd`.
Includes catalog, endpoints, alias spec + review verdicts, probe reports,
92 canary artifacts (plus superseded retries), 33 route snapshots, the
local route map, decisions, and reconciliation.

Reviews: hatch claude-opus and cursor-grok approved the alias mapping
(runs `hatch_20260810T234201.355557000Z_7b7956473b299352`,
`hatch_20260810T234202.319309000Z_74146841a01db006`); cursor-grok returned
READY on the gate-change commits
(`hatch_20260811T000038.663031000Z_f54cbe2a4fd82973`).

Verification: full API suite 275 passed; live smoke resolved routed
transport with verified pinned provider metadata for deepinfra and vertex
lanes; fail-closed checks (missing snapshot, corrupt evidence hash)
returned direct.

## Production stop gate

Not activated. The next step is explicit: apply the 33 route snapshots via
`bench_route_decisions` (`openrouter_promote_route.py --apply` per route or
a batch equivalent), set `OPENROUTER_ROUTING_ENABLED=1` in the tracked
manual-app compose, deploy, and watch `bench_model_health` freshness plus
`route_state` provenance on new metric rows. Rollback is the env flag, or
`openrouter_revoke_route.py` per route. Route snapshots expire in 24h by
design; re-promotion (or a refreshed canary) is expected at activation
time.
