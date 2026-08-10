# OpenRouter paired canary

Run date: 2026-08-10 UTC
Mode: report-only, no MongoDB writes
Canary ID: `canary:openai:gpt-4o-mini:20260810T173751Z`
Canary artifact: `/tmp/or_canary_gpt-4o-mini_30_balanced_v2.json`

The canary used the production OpenAI and OpenRouter adapters with the default
cloud profile and 64-token budget. It ran 30 paired requests with randomized,
balanced order (15 direct-first and 15 OpenRouter-first). The OpenRouter
request used the exact `openai/gpt-4o-mini` model ID, provider restriction
`openai`, disabled fallbacks, and required parameters. The canary's pricing
inputs are recorded in `/tmp/gpt4o-mini-pricing.json` and reference the
[OpenAI model pricing page](https://developers.openai.com/api/docs/models/gpt-4o-mini)
and the dated OpenRouter endpoint snapshot.

| Check | Result |
| --- | ---: |
| Successful paired requests | 30 / 30 |
| Output validation | passed |
| Route provider metadata | 30 / 30 verified |
| OpenRouter/direct generated-throughput ratio | 1.017x |
| 95% bootstrap TPS ratio CI | 0.974x - 1.041x |
| OpenRouter/direct TTFT ratio | 1.055x |
| 95% bootstrap TTFT ratio CI | 0.915x - 1.088x |
| OpenRouter/direct estimated cost ratio | 1.000x |
| 95% bootstrap cost ratio CI | 1.000x - 1.000x |
| Route error-rate delta | 0.000 |
| Promotion state | passed |

The predeclared gates were at least 29 of 30 usable pairs, TPS lower bound at
least 0.8x, TTFT upper bound at most 1.5x, route error delta at most five
percentage points, verified estimated cost upper bound at most 1.10x, and
100% routed provider metadata verification. All gates passed. The ratios are
paired effects with deterministic percentile bootstrap intervals, not claims
that OpenRouter is faster. The evidence remains report-only until the explicit
promotion command writes an expiring route record.

The canary JSON, pricing inputs, and dry-run active route are retained under
the immutable `artifact.manifest.json` at
`s3://artifacts/llm-benchmarks/openrouter-consolidation/v1`.
The corresponding dry-run route is
`derived/or_route_openai_gpt-4o-mini_active_v3.json` in that manifest.
