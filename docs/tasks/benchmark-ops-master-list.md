# Benchmark Ops Master List

This is the working list after the Bedrock runner restore on 2026-05-11.
Keep it short, operational, and biased toward reducing silent drift.

## Ground Rules

- Clifford MongoDB is the source of truth for model enablement and lifecycle
  state.
- The RND EC2 Bedrock runner must not connect to MongoDB or personal network
  database ports. It talks to `bench-ingest` over HTTPS.
- Bedrock model IDs may contain provider-required date strings, but display and
  canonical names must never end in date/timestamp/checkpoint suffixes.
- Test all actual model versions. Do not disable Opus 4.5/4.6/etc. just because
  the provider ID contains a date. Only collapse duplicate aliases that would
  render as the same display model.
- Prefer small, atomic commits and live verification after each deploy-impacting
  change.

## Task 1: Make `bench-ingest` Deployable From Tracked Source

Problem: `bench-ingest` is live on clifford under `/opt/bench-ingest`, but the
server-specific deploy state is not yet tracked in the normal manual-app source
of truth.

Scope:
- Add `mytech/infrastructure/manual-apps/bench-ingest/` with manifest, compose,
  and route fragment.
- Keep runtime secrets out of git. Clifford should keep them in an untracked
  remote `.env.secrets` or equivalent.
- Preserve the public endpoint `https://bench-ingest.drose.io`.
- Verify both server-local and public `/health`.

Success criteria:
- `manual-app status bench-ingest --json` explains container, route, and health.
- `manual-app deploy bench-ingest --repo-dir ~/git/llmbench/bench-ingest` can
  reproduce the service without hand-editing `/opt/bench-ingest`.
- The public `/runner-config?provider=bedrock` endpoint still returns the
  enabled Bedrock model worklist after deploy.

## Task 2: Bedrock Catalog Hygiene CLI

Problem: Bedrock model enablement is now database-driven, but alias hygiene is
still manual and easy to misread.

Scope:
- Add a report/check command that groups Bedrock rows by normalized display
  identity.
- Flag duplicate enabled aliases for the same display model.
- Flag display/canonical names that end in date/timestamp/checkpoint suffixes.
- Distinguish actual version numbers from provider checkpoint dates.

Success criteria:
- The check passes for the current Bedrock catalog.
- A fixture catches duplicate regional/global aliases for the same display name.
- A fixture preserves actual versions like Opus 4.5, 4.6, and 4.7.

## Task 3: Move Bedrock Provider Quirks Into Metadata

Problem: provider-specific request quirks are still encoded as model-name checks
in runner code.

Scope:
- Add model metadata for known Bedrock invocation quirks, starting with
  "omit temperature" for models that reject sampling controls.
- Update the Bedrock provider to read metadata rather than matching a hardcoded
  Opus 4.7 string.
- Keep the runner behavior unchanged for current models.

Success criteria:
- Opus 4.7 still runs successfully.
- Tests cover both normal temperature and omit-temperature paths.
- Adding a future Bedrock model quirk does not require a provider code change.

## Task 4: Freshness And Drift Alerts

Problem: a stopped Bedrock runner looked like a chart oddity before it looked
like an operational incident.

Scope:
- Add or update Sauron/ops health checks for Bedrock runner freshness.
- Alert on stale `bench_model_health` or missing recent `metrics_cloud_v2` rows
  for enabled Bedrock models.
- Include `/runner-config` model count and latest successful ingest timestamp.

Success criteria:
- A stopped Bedrock runner produces an explicit health failure.
- Disabling/deprecating a model in MongoDB does not produce a false stale alert.
- The check output names the stale provider/model and last successful run.

## Task 5: Dashboard Chart Ordering And Visibility

Problem: provider comparison charts can make the most important line hard to see,
and the page ordering did not obviously match "most providers per model."

Scope:
- Revisit time-series color/line rules so single-provider or stopped/deprecated
  series remain legible.
- Verify chart ordering logic against the intended priority: models with more
  provider coverage should sort before single-provider charts, with sensible
  tie-breakers.
- Make "stopped" visual state useful without making the only line low contrast.

Success criteria:
- Bedrock stopped segments remain readable.
- Single-provider charts do not unexpectedly dominate the top of the page unless
  the sorting rule explicitly says they should.
- Visual tests or screenshot checks cover the affected chart states.

## Task 6: Discovery-To-Catalog Promotion

Problem: new provider models can be discovered, but human promotion into the
enabled catalog still needs safer guardrails.

Scope:
- Define the review flow from `provider_catalog` to `models`.
- Show new Bedrock/provider entries with normalized display identity and likely
  duplicate aliases.
- Keep automatic discovery separate from automatic enablement.

Success criteria:
- New model reports are actionable without direct Mongo spelunking.
- Enabling a new Bedrock model also runs the alias hygiene check.
- Promotion leaves a clear audit trail in Mongo fields.

## Task 7: Cleanup Pass

Problem: the restore added the right control plane, but some old docs, naming,
  and cross-repo seams are still rough.

Scope:
- Update stale EC2 instance IDs, deployment paths, and bridge docs.
- Reduce hardcoded dashboard display-name mappings where processed data can
  provide the canonical/display names.
- Remove obsolete `BENCHMARK_MODELS`-first guidance outside the emergency
  override path.

Success criteria:
- A new agent can answer "where does Bedrock config come from?" from docs alone.
- No docs imply that production Bedrock is controlled by a static env model list.
- Dashboard model naming for current Bedrock Claude models is data-driven or
  explicitly covered by tests.
