# OpenRouter migration live evidence, 2026-08-10

This is the result of the first hardened end-to-end run. It is evidence for
the migration decision, not a claim that the limited public catalog is the
complete OpenRouter catalog.

## Input and scope

- Enabled source rows: 241
- Source snapshot SHA-256: `05c0d2050abcc9767cb6b6521a1b49f9afbe547c56f2d1feda14441c0a8c780d`
- OpenRouter discovery rows: 400
- Catalog snapshot SHA-256: `02bf3d644019604f37a388f29bcb7b98e6584df5fa8bbf22f7c4657dd678be66`
- Catalog scope: public discovery, not proven global
- Identity rule: `or-identity-v2`
- Alias rule: `or-alias-v1`
- Reconciliation run: `reconcile:2c0c6ae773cc6e91bbede569`
- Alias review receipts:
  - Cursor Grok: `hatch_20260810T190002.855387000Z_cdf9d525f035c771`
  - Hatch Sol: `hatch_20260810T190518.377100000Z_14d6d714301fc525`

## Decision counts

| Terminal state | Count |
| --- | ---: |
| `route-approved` | 1 |
| `direct-policy-excluded` | 21 |
| `direct-incompatible` | 5 |
| `direct-probe-failed` | 2 |
| `direct-unknown` | 212 |
| **Total** | **241** |

The one approved route is `openai/gpt-4o-mini`. It passed 30 paired requests
with 30 successful route responses, zero errors, throughput ratio 1.1141,
TTFT ratio 1.0451, and cost ratio 1.0000. It is represented by a dry-run
active route artifact with revocation generation 0.

The availability probe observed 55 successful responses from 59 scheduled
candidates. The four unsuccessful or incomplete observations were retained as
direct decisions. The successful probe count is higher than the active route
count because activation also requires direct and routed pricing plus a paired
canary.

## Evidence bundle

Manifest: `artifact.manifest.v2.json`

Artifact prefix:
`artifacts/llm-benchmarks/openrouter-consolidation/v2/`

The manifest hash-addresses the frozen source and catalog, endpoint captures,
reviewed aliases, audit, probe, pricing, canary, active route, final decisions,
and reconciliation report. The reconciliation run is deterministic and
idempotent for the same input hashes.

The promoted route points to the v2 canary URI
`s3://artifacts/llm-benchmarks/openrouter-consolidation/v2/derived/or_canary_gpt-4o-mini_live_v2.json`.
The manifest was re-verified after that URI correction and after adding the
hash-addressed alias evidence manifest.

## Operational boundary

The run did not write production MongoDB because `MONGODB_URI` was not
configured. Applying the reconciliation requires the normal deployment
credential and the explicit apply flag. Until then, all source rows remain
safe on their direct adapters and the generated route is only a reviewable
artifact.
