# LLM Benchmarks API Service

Benchmark runner that calls LLM APIs and measures performance. Deployed as two instances:
- **clifford** (VPS): All providers except Bedrock
- **EC2**: Bedrock only (uses IAM role)

---

## Operating model

**This site is meant to run itself.** The goal is a living system maintained by
AI agents, not by David. He used to hand-curate the model catalogue; that is
being retired because current models can do it continuously and cheaply.

- **Gate on reversibility, not uncertainty.** Enabling or disabling a model,
  requeuing jobs, redeploying — reversible and logged. Do them without asking.
  **Concurrency and cadence are excluded** — see the spend rule below.
- **Spend inference, not human attention.** Ambiguity is resolved with more
  model calls, not a review queue. This covers one-off calls to settle a
  question; it does not cover raising the standing measurement rate.
- **Never build a queue that waits on David.** Routing low confidence to
  "human review" is a design failure.
- **Leave a trail.** Every mutation carries a reason and timestamp so a later
  agent can audit and reverse it.
- **Escalate only for** spending money and destroying published history — and
  even then, notify and continue with everything else.

Full context: `~/git/llmbench/AGENTS.md`. Roadmap: `docs/platform-plan.md`.

---

## DO NOT INCREASE SPEND RATE

**Hard owner constraint, 2026-08-18. This overrides the "do it without asking"
clauses above.** The budget for this site is a few dollars a day. Concurrency
and cadence are not routine reversible changes — they set a recurring bill.

The knobs are **not in this repo**. They are in the deployment compose:
`~/git/me/domains/mytech/infrastructure/manual-apps/llm-bench-dashboard/compose.yaml`

| Knob | Now | What it does |
|---|---|---|
| `FRESH_MINUTES` | `2880` (2 days) | **The spend lever** — sets the cadence, and so jobs/day |
| `BENCHMARK_MAX_COST_PER_RUN_USD` | `0.005` | Ceiling on one run, not the average |
| `OPENROUTER_CONCURRENCY` / `BENCHMARK_CONCURRENCY_OPENROUTER` | `4` | Burst ceiling only — how fast a backlog drains |

```
jobs/day  = enabled_targets / cadence_days
$/day     = rows/day x measured_cost_per_row
```

Cost from **measured rows**, never from the per-run ceiling. Measured
2026-08-18: 1,169 enabled targets (790 endpoint + 379 model-level); Aug 17
produced 13,462 rows against a $45.75 bill = **$0.0034/row** against a $0.005
ceiling. And rows are not jobs — the sampled ratio was ~4.6 rows per job, a
multiplier nobody has yet explained, so treat any jobs/day figure as a lower
bound.

**Before changing any knob, compute the new $/day and put the number in the
commit message.** A change whose cost was never calculated is not reversible in
practice; it is an open-ended charge that surfaces on a bill days later.

**Growing the catalogue is a spend change too.** If discovery admits another 500
endpoints the bill rises proportionally with no config edit at all. Population
is a spend input exactly as much as cadence is — which is how this happened:
the endpoint cutover multiplied the fleet ~4.7x, `e8d3b14` then fixed a stuck
pool and took throughput from 12/hour to ~340/hour, and both changes were
correct. What was missing is that nobody recomputed the bill afterwards.

**What this costs, honestly.** At this budget the sampling policy cannot reach
its `official` tier in reasonable time — 30 samples at a 2-day cadence is ~60
days per endpoint, and `preliminary` (8 samples) is ~16 days. That is a real
conflict between the publication design and the budget. The budget wins until
the owner says otherwise. Do not "fix" it by speeding the scheduler back up.

---

## Repository Structure

**Parent directory structure:**
```
~/git/llmbench/
├── llm-benchmarks/           # THIS REPO - Benchmark runner
└── llm-benchmarks-dashboard/ # Separate repo - Next.js dashboard
```

Always `cd` into the specific subdirectory before git operations.

**Key files:**
- `api/llm_bench/scheduler/` - Mongo-backed scheduler, queue, worker, health, and process-isolated runner
- `api/bench_simple_runner.py` - Bedrock/remote HTTP-ingest runner
- `api/llm_bench/cloud/providers/` - Provider implementations
- `api/llm_bench/models_db.py` - Loads enabled models from MongoDB
- `REASONING_MODELS.md` - OpenAI o1/o3/o4 documentation
- `TROUBLESHOOTING.md` - Error patterns and solutions

---

## Architecture

### Two Runners

| Instance | Providers | Why Separate |
|----------|-----------|--------------|
| **clifford** | anthropic, cerebras, deepinfra, fireworks, groq, openai, together, vertex | Mongo-backed scheduler with isolated provider worker lanes |
| **EC2** | bedrock | Needs AWS IAM role + MongoDB via HTTP bridge |

**HTTP Ingest Bridge:** EC2 POSTs results to `https://bench-ingest.drose.io` (on clifford), which writes to MongoDB. `bench-ingest` is a manual app, not a Coolify app: tracked deploy state is in `~/git/me/domains/mytech/infrastructure/manual-apps/bench-ingest/`, runtime secrets are on clifford at `/home/drose/manual-apps/bench-ingest/.env.secrets`, and deploys use `~/git/me/domains/mytech/bin/manual-app deploy bench-ingest --repo-dir ~/git/llmbench/bench-ingest`.

**Configuration:** clifford scheduler config lives in the tracked manual-app compose at `~/git/me/domains/mytech/infrastructure/manual-apps/llm-bench-dashboard/compose.yaml` — not in this repo's `docker-compose.yml`, which is not what runs in production. `bench-ingest` config/secrets live in the manual-app remote `.env.secrets`. The RND Bedrock runner loads `/etc/bedrock-bench/runner.env`, but its model worklist comes from `bench-ingest` `/runner-config`, not from a static env var.

---

## MongoDB

**Connection:** Use the service env on clifford; do not ask the user for `MONGODB_URI`. The RND Bedrock runner must never receive MongoDB credentials.

**Key collections:**
- `models` - Enabled models (provider, model_id, enabled, deprecated)
- `bench_jobs` - Scheduler queue
- `bench_model_health` - Authoritative freshness/error state
- `metrics_cloud_v2` - Successful runs (run_ts, tokens_per_second)
- `errors_cloud` - Failed runs (ts, message, stage)
- `provider_catalog` - Models discovered from provider APIs (managed by Sauron)

**Quick commands:**
```bash
# Check errors (last 24h)
mongosh "$MONGODB_URI" --eval "db.errors_cloud.aggregate([{\$match: {ts: {\$gte: new Date(Date.now()-86400000)}}}, {\$group: {_id: '\$provider', count: {\$sum: 1}}}])"

# List enabled models
mongosh "$MONGODB_URI" --eval "db.models.find({enabled: true}, {provider: 1, model_id: 1})"

# Disable a model
mongosh "$MONGODB_URI" --eval "db.models.updateOne({provider: 'X', model_id: 'Y'}, {\$set: {enabled: false, disabled_reason: 'Reason', disabled_at: new Date()}})"
```

---

## Common Issues

| Symptom | Quick Fix |
|---------|-----------|
| Bedrock `ProfileNotFound: zh-ml-mlengineer` | Remove stale `AWS_PROFILE` from clifford env vars; Bedrock runs on the RND EC2 instance only |
| Groq `output_tokens not within 10%` | Expected for variable-output providers; validation uses `visible_nonzero`, not a +-10% band |
| OpenAI o3-mini `unsupported parameter: max_tokens` | Update code - needs Responses API (see REASONING_MODELS.md) |
| Vertex `429 quota exceeded` | Increase GCP quota or reduce frequency |

**Full troubleshooting:** See `TROUBLESHOOTING.md`

---

## Deployment

**clifford (manual-app).** Coolify was retired from personal infra on 2026-06-26.
This one command builds and deploys both the dashboard and the runner:
```bash
~/git/me/domains/mytech/bin/manual-app deploy llm-bench-dashboard --repo-dir ~/git/llmbench
```
The deploy gates on the commit being pushed to `origin/main`.

**RND EC2 Bedrock runner (via AWS SSM):**
```bash
aws sso login --profile zh-marketing-preprod-engineer
aws ssm start-session --target i-056bc81c58a387657 --region us-east-1 --profile zh-marketing-preprod-engineer
```

**Logs:**
```bash
ssh clifford 'docker logs -f llm-bench-api-service'   # fixed name, no hash suffix
```

---

## Adding/Disabling Models

**Critical rules:**
- **Bedrock:** Use `us.anthropic.*` / `us.meta.*` prefixes (not `anthropic.*` / `meta.*`)
- **Bedrock display/canonical names:** never include date/timestamp checkpoint suffixes; keep one enabled alias per display model (no duplicate `claude-opus-4.6` rows from regional/global/date variants).
- **Bedrock runner config:** enable/disable models in MongoDB `models`; production does not use `BENCHMARK_MODELS` except with the explicit emergency `BENCHMARK_MODELS_OVERRIDE=1`.
- **OpenAI:** o1/o3/o4 models auto-detected as reasoning models
- **OpenAI-compatible hosted providers:** use provider-reported usage tokens; streamed text chunks can omit hidden/reasoning tokens
- **Bedrock ingest bridge:** `bench-ingest.drose.io` must preserve additive metric fields; schema-v2 runner fields are lost if the bridge rejects or ignores extras.
- **The benchmark surface is any endpoint that takes a query and returns text.**
  That includes guard, moderation, router and compound models — a user asking
  "how fast is Llama Guard" is asking a real question, and latency on a
  classifier is useful information. Do not disable a text-returning model for
  being the "wrong kind" of model.
  Only genuinely non-text endpoints are out: embeddings, TTS, transcription,
  image and video. Do not detect those by name pattern — two passes of that
  still let through `veo`, `kling`, `vidu`, `ideogram`, `parakeet`. A real call
  is the reliable classifier, because a non-text model cannot return text
  tokens at a measurable rate.
- **Provider `/models` does not tell you what is benchmarkable.** Together
  lists dedicated-endpoint-only models with normal pricing and `type: "chat"`,
  and its `running` field is `false` for all 274 models including working ones.
  Only a live call distinguishes them.

**Commands:**
```bash
# Add
mongosh "$MONGODB_URI" --eval "db.models.insertOne({provider: 'openai', model_id: 'gpt-4o', enabled: true, created_at: new Date()})"

# Disable (record a class so a recoverable reason gets a recheck date)
mongosh "$MONGODB_URI" --eval "db.models.updateOne({provider: 'X', model_id: 'Y'}, {\$set: {enabled: false, disabled_class: 'hard_model', disabled_reason: 'Endpoint 404s', disabled_at: new Date()}})"
```

---

## Provider Discovery

**Sauron job:** `llm-bench-provider-discovery` (07:00 UTC daily)

Fetches models from provider APIs → stores in `provider_catalog`. Email
reporting is disabled in `run()`; the agent digest owns notification.

This job was registered `enabled=False` from ~2026-04-29 to 2026-08-04, so
`provider_catalog` silently went three months stale while still looking like a
live source. A disabled Sauron job is indistinguishable from a healthy one —
check `curl http://127.0.0.1:8876/jobs` on clifford before trusting it.

**Providers:** Groq, Together, Cerebras, OpenAI, Anthropic, Fireworks, DeepInfra

**Check catalog:**
```bash
ssh clifford 'mongosh "$MONGODB_URI" --quiet --eval "db.provider_catalog.aggregate([{\$group: {_id: \"\$provider\", count: {\$sum: 1}}}])"'
```

---

## Related Docs

- **Detailed troubleshooting:** `TROUBLESHOOTING.md` (create if complex patterns emerge)
- **Cross-repo context + operating model:** `~/git/llmbench/AGENTS.md`
- **Roadmap and open questions:** `docs/platform-plan.md`
- **Dashboard:** `~/git/llmbench/llm-benchmarks-dashboard/backend/`
- **Sauron:** `~/git/sauron/AGENTS.md`
- **OpenAI Reasoning:** `REASONING_MODELS.md` (this repo)

---

## Agent learnings

Format: `- (YYYY-MM-DD) [category] Insight.`
Categories: `gotcha`, `pattern`, `deploy`, `perf`, `data`.

- (2026-08-04) [gotcha] `log_mongo` copies only keys listed in
  `OPTIONAL_METRIC_FIELDS`. A new metric field set by the runner is silently
  dropped unless it is added there too — 92 rows were written carrying none of
  a new provenance contract. Tests that mock the writer cannot see this; assert
  against `_optional_metric_fields` output instead.
- (2026-08-04) [pattern] Verify an invariant by running it against production
  before trusting it. Of the first three findings, two were the check being
  wrong: queue-age measured `created_at` and read normal exponential backoff as
  stalls, and a terminal-reason check fired on 466 correctly-dead models
  including a provider whose API shut down in 2024.
- (2026-08-04) [gotcha] Sauron's control API needs `X-Sauron-Control-Token`
  (not `X-Control-Token`); the value is `SAURON_CONTROL_TOKEN` in Infisical.
  Sauron has no docker CLI, so a job cannot `docker exec` into a container —
  jobs reach llm-bench through Mongo only.
- (2026-08-04) [gotcha] Do not guess OpenRouter model IDs from memory —
  `openai/gpt-5.3-mini` was invented and every call 400'd. List them first:
  `curl -s https://openrouter.ai/api/v1/models -H "Authorization: Bearer $KEY"`.
  `openai/gpt-5.6-luna` honours `response_format: json_object`; the identity
  normalizer uses it.
- (2026-08-04) [data] The publication pipeline groups by
  `(providerCanonical, display_name)`, so two providers share a chart line only
  when their display names match exactly. `claude-haiku-4.5` vs
  `claude-haiku-4-5` split a three-provider model into two lines. Unifying
  names within a derived identity group took 3+ provider lines from 3 to 6.
  Two guards matter: only unify across providers, and never rename onto a name
  the same provider already publishes — both merge deployments instead of
  adding a provider.
- (2026-08-05) [gotcha] A bounded list is a coverage leak when the bound is
  applied to *which* items are considered rather than how much work one pass
  creates. `scheduler_pass` sliced each provider's models at `--limit` (100);
  DeepInfra had 112, so the same twelve were never scheduled — no error, no
  dead letter, nothing disabled — and one more fell off every time admission
  added a model. Cap the work, order by staleness, never slice the population.
- (2026-08-05) [pattern] A permanently red check is as useless as no check.
  The staleness horizon multiplied the invariant loop's cadence (how often we
  look) instead of the measurement period (how often a model's turn comes
  around, measured at 45 min), so thirteen healthy models were reported starved
  on every run. Measure the real period before setting a threshold against it.
- (2026-08-05) [data] `unknown` was carrying three verdicts: deleted models,
  dedicated-endpoint refusals (Together/Fireworks answer 400, not 404), and
  reasoning models that spend the whole 64-token budget before emitting
  anything visible. The last is not a failure — it is the profile failing to
  measure the model — and is now `budget_exhausted`, terminal without
  quarantine. `llm_error_classifier.py` has no caller anywhere, so nothing ever
  resolves what is left in `unknown`.
- (2026-08-05) [gotcha] `invariants.evaluate` writes to `bench_check_runs` by
  default. Pass `record=False` for any exploratory or fault-injection run: the
  watchdog endpoint and the external dead man both read the newest row, so a
  deliberate failure written there pages for a fault that is already gone.
