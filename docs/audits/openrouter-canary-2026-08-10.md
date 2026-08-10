# OpenRouter paired canary

Run date: 2026-08-10 UTC
Mode: report-only, no MongoDB writes
Canary ID: `canary:openai:gpt-4o-mini:20260810T032957Z`

The canary used the production OpenAI and OpenRouter adapters with the default
cloud profile and 64-token budget. It ran two paired requests with balanced
order: one direct-first pair and one OpenRouter-first pair. The OpenRouter
request used the exact `openai/gpt-4o-mini` model ID, provider restriction
`openai`, disabled fallbacks, and required parameters.

| Check | Result |
| --- | ---: |
| Successful paired requests | 2 / 2 |
| Output validation | passed |
| OpenRouter/direct generated-throughput ratio | 1.36x |
| OpenRouter/direct TTFT ratio | 0.76x |
| Minimum throughput ratio | 0.50x |
| Maximum TTFT ratio | 3.00x |
| Cost comparison | unverified |

OpenRouter reported `observed_provider_slug=openai` and verified provider
metadata on both routed attempts. The result is
`measurement_passed_cost_unverified`, not a promotion record. Cost remains a
required human-reviewed gate because the current direct-provider metric
contract does not expose a comparable cost field. The raw evidence is retained
at `/tmp/or_canary_gpt-4o-mini_balanced.json`.
