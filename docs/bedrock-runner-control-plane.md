# Bedrock Runner Control Plane

## Goal

Bedrock benchmarks run from the RND EC2 instance because AWS Bedrock access lives
there. MongoDB on clifford remains the source of truth for model catalog state:
`models.enabled` and `models.deprecated` decide what should run.

The EC2 runner must not connect to MongoDB, tunnels, Tailscale, or any personal
network database port. Its only cross-boundary traffic is short-lived HTTPS to
`bench-ingest`.

## Architecture

```text
RND EC2 Bedrock runner
  GET  https://bench-ingest.drose.io/runner-config?provider=bedrock
  POST https://bench-ingest.drose.io/ingest

bench-ingest on clifford
  reads/writes MongoDB
```

`bench-ingest` is the HTTPS boundary. The runner never receives MongoDB
credentials, database names, collection names, internal hostnames, or provider
secrets.

## Runner Config API

Request:

```http
GET /runner-config?provider=bedrock
Authorization: Bearer <runner-config-token>
```

Response:

```json
{
  "schema_version": 1,
  "provider": "bedrock",
  "interval_minutes": 30,
  "models": ["anthropic.claude-opus-4-7"],
  "model_metadata": {
    "anthropic.claude-opus-4-7": {
      "omit_temperature": true
    }
  },
  "generated_at": "2026-05-11T20:00:00Z"
}
```

Server behavior:

- `provider` is the canonical provider string. Initially only `bedrock` is
  allowed.
- Query `models` for `provider == requested provider`, `enabled == true`, and
  `deprecated != true`.
- Sort by `display_name`, then `model_id`, for stable cycles.
- The catalog should keep one enabled row per displayed model/version. Model IDs
  may contain provider-required dates, but `display_name` and `canonical_id`
  must not include date or timestamp checkpoint suffixes. Do not enable both
  regional/global aliases or dated aliases that would render as the same model
  line.
- Return only the minimal runner worklist, cadence, and per-model request
  metadata needed by the runner. Provider quirks such as `omit_temperature`
  belong in MongoDB model metadata, not in model-name string checks.
- Authenticate with `RUNNER_CONFIG_TOKEN`, separate from `INGEST_API_KEY`.
- Log provider, model count, status, latency, and a stable token identity.
- Rate-limit per token.

## Runner Behavior

Production Bedrock uses dynamic config:

```text
RUNNER_CONFIG_URL=https://bench-ingest.drose.io/runner-config?provider=bedrock
RUNNER_CONFIG_TOKEN=...
INGEST_API_URL=https://bench-ingest.drose.io/ingest
INGEST_API_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

At the start of each daemon cycle, the runner fetches `/runner-config` and uses
that model list for the cycle. Config changes apply to the next cycle; a model
disabled mid-cycle may still finish its in-flight run.

Failure handling:

- `200` with models: run that list and persist it to
  `/var/lib/bedrock-bench/last_config.json`.
- `200` with an empty model list: skip the cycle without error.
- `5xx` or network failure: use the persisted config only within the configured
  grace window.
- `401` or `403`: do not use cache. Treat the token as invalid/revoked.
- Cache older than the grace window: skip cycles and log a stale-config error.

Emergency rollback:

```text
BENCHMARK_MODELS_OVERRIDE=1
BENCHMARK_MODELS=model-a,model-b
```

When the override flag is set, `BENCHMARK_MODELS` wins over dynamic config.
Without the override flag, static `BENCHMARK_MODELS` is only a local/manual
fallback when `RUNNER_CONFIG_URL` is unset.
Override mode also bypasses `/runner-config` `model_metadata`; avoid it for
models that need request quirks such as `omit_temperature` unless you are
accepting those failures for the emergency run.

## Ingest Guardrails

`POST /ingest` validates incoming provider/model pairs against the same enabled
catalog before storing metrics. The enabled set is cached in `bench-ingest` for a
short TTL to avoid a MongoDB read per metric.

Disabled, deprecated, or unknown models are not inserted into
`metrics_cloud_v2`. The API returns a semantic rejected response rather than an
auth-looking failure, so a stale runner does not look like a credential problem.

## Token Storage and Rotation

On RND EC2:

```text
/etc/bedrock-bench/runner.env
owner: ubuntu:ubuntu
mode: 0600
```

The systemd unit loads this file via `EnvironmentFile=`.

Rotation procedure:

1. Issue new `RUNNER_CONFIG_TOKEN` and/or `INGEST_API_KEY`.
2. Update the token in `bench-ingest` and redeploy/restart it.
3. Update `/etc/bedrock-bench/runner.env` on RND EC2 and restart
   `bedrock-bench.service`.
4. Verify config fetches and ingest writes with the new token.
5. Expect the runner to skip work between steps 2 and 3 if the config token was
   rotated. The current implementation supports one active config token, not
   overlap rotation.

## Success Criteria

- Bedrock production runner does not depend on a static `BENCHMARK_MODELS` list.
- Updating `models.enabled` or `models.deprecated` in MongoDB changes the
  Bedrock worklist by the next runner cycle.
- RND EC2 only talks to `bench-ingest` over HTTPS.
- New Bedrock catalog entries, such as Opus 4.7, appear in `/runner-config` once
  enabled in MongoDB, with any required `model_metadata`.
- Disabled, deprecated, and unknown models do not receive new metric rows.
- Runner survives transient config endpoint failures using bounded persisted
  cache.
- Auth failures stop runner work instead of silently using stale config.
- Logs are sufficient to explain config fetches, rejected metrics, and successful
  cycles.
