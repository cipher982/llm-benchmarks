# OpenRouter provider research, 2026-08-10

This is a provider-level follow-up to the hardened 241-row reconciliation. It
answers a narrower question: which direct-provider rows have concrete
OpenRouter candidates that deserve the next paired canary pass?

## Evidence basis

- Source snapshot: 241 enabled rows, SHA-256
  `05c0d2050abcc9767cb6b6521a1b49f9afbe547c56f2d1feda14441c0a8c780d`.
- OpenRouter snapshot: 400 public-discovery rows, SHA-256
  `02bf3d644019604f37a388f29bcb7b98e6584df5fa8bbf22f7c4657dd678be66`.
- The catalog is recorded as public discovery, not proven global coverage.
- Provider review used the hash-addressed v2 coverage, alias, endpoint, and
  probe artifacts. A candidate is still only a candidate until its paired
  direct/OpenRouter canary passes.

## Findings

The review found 55 source-row candidates with successful pinned availability
evidence, excluding the already-canary-tested `openai/gpt-4o-mini` pilot. They
are not 55 claims of production equivalence.

- First-party families contributed 39 source rows, representing 36 distinct
  OpenRouter IDs after collapsing duplicate hosted rows.
- Gateway providers contributed 29 source rows with reviewed identity and
  successful probes. Some share the same OpenRouter ID as first-party rows or
  as other gateway rows.
- The remaining source rows are split between incomplete catalog evidence,
  missing source-provider identity, protocol incompatibility, failed probes,
  and explicit Bedrock policy exclusion.

Examples of canary-ready candidates include OpenAI GPT, Anthropic Claude,
Gemma, Mistral Small, Llama, Qwen, Phi, Kimi, GPT-OSS, and ThinkingMachines
Inkling variants. The full row-level evidence remains in the v2 artifacts,
rather than being inferred from family names.

## Important exclusions

- Bedrock's 21 rows remain direct by policy. OpenRouter family names alone do
  not prove equivalence to an AWS account, region, or Bedrock build.
- Vertex's four rows remain direct until the Google Vertex transport and
  modality behavior are compared separately. A `google/...` slug alone is not
  enough.
- Fireworks router and fast variants remain unresolved because their hosted
  routing semantics are not the same thing as an exact model identity.
- A successful availability probe is not a migration approval. Pricing,
  paired behavior, health, and rollback evidence are still required.

## Next deterministic work

Run paired canaries in bounded batches over the candidate set, deduplicated by
OpenRouter ID while retaining every source-row mapping. Promote only rows that
pass the existing cost, latency, error, output, and revocation gates. Keep all
other rows direct with their specific evidence reason.
