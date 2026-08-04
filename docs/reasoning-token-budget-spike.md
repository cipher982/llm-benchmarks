# Reasoning Token Budget Spike

Status: proposal only. This spike did not change runner or production behavior.

Date: 2026-08-04

## What We Measure and Why

The benchmark serves developers choosing a model/provider combination and operators comparing providers that serve the same model. It currently compresses several questions into one tokens-per-second number:

| Audience question | Appropriate measurement |
|---|---|
| How long until a user sees an answer? | Time from request send to first answer token. |
| How long until enough answer text is available to be useful? | Time from request send to a fixed answer-text target, plus the rate at which runs reach that target. |
| How quickly does answer text stream once it starts? | Answer tokens divided by time from first to last measured answer token. |
| How much generation work did the provider deliver per wall-clock second? | Provider-reported output tokens, including billed reasoning when present, divided by full request duration. |
| How much paid work was reasoning rather than answer text? | Provider-reported reasoning tokens and reasoning share, when the provider supplies a trustworthy breakdown. |
| Does this configuration reliably return an answer within its cap? | Outcome distribution, especially answer-target completion and budget-exhaustion rates. |

There is no single correct throughput metric. Each numerator and timing interval answers a different question. The defensible provider comparison is the same model under the same request profile; cross-model comparisons also include tokenizer and model-behavior differences.

The issue is both the token numerator and the time denominator. The current `visible_tokens_per_second` is visible text divided by the full request duration. That is a reasonable *answer yield* metric: hidden reasoning lowers the number because it makes the user wait. It is not answer decode speed. Answer decode speed should exclude time before the first answer token. Conversely, `generated_tokens_per_second` includes reasoning work in its numerator and uses full request time in its denominator. It is useful for delivered provider work and cost context, but it can look healthy while the user receives no answer. Because the denominator includes network latency, queueing, and prefill, it is not pure engine decode throughput either.

The benchmark also cannot determine whether text is actually useful. The current long-story prompt and a fixed answer-token target can establish that answer text arrived, not that it was correct or useful. This proposal therefore uses "answer target" rather than "useful answer" in field names and UI copy. Quality requires a different workload and scorer.

### Where the current abstraction works

Schema v2 already made several correct distinctions:

- `generated_output_tokens` can preserve provider-reported output work, including reasoning.
- `visible_output_tokens` can preserve answer text separately where it is observable.
- `reasoning_tokens` can preserve a provider-reported split.
- `time_to_first_token` is documented as time to first visible text and is nullable when no visible text arrives.
- `token_source`, `request_mode`, `max_output_tokens_attempted`, and `reasoning_effort` provide some provenance.

These fields are enough to avoid calling a reasoning-inclusive count "visible tokens." They are not enough to establish a stable benchmark protocol or represent a completed request that returned no answer.

### Where it breaks down

The runner currently uses a global 64-token request in `scheduler/runner.py`. The shared OpenAI-compatible path then tries 64, 256, and 512 total output tokens. Together and DeepInfra can repeat those budgets with `reasoning.enabled=false`. Only the final attempt is returned. Its `generate_time` excludes time and cost from all preceding attempts.

This causes several measurement problems:

- Rows for the same model and provider can represent different total output caps.
- Rows can silently switch from reasoning enabled or provider default to a requested reasoning-disabled mode.
- `attempts_count` says that prior calls happened, but their durations, token usage, costs, and outcomes are discarded.
- A successful final retry is selected preferentially, biasing the published distribution toward requests that eventually produced text.
- `reasoning_effort="disabled"` records what the runner requested, not what the provider demonstrably applied. Together documents models whose reasoning cannot be disabled, and DeepInfra says unsupported reasoning parameters can have no effect.
- The OpenAI-compatible path is non-streaming, so it cannot measure first answer token, first reasoning token, or answer stream speed even though Together and DeepInfra support streaming.
- `visible_output_tokens` is not one stable unit. Depending on the response, it can be a provider-native subtraction or a `cl100k_base` count of returned text. Subtracting a client-tokenized text count from provider-native usage is not a valid token split.
- Provider total output minus reported reasoning is not always visible answer text. OpenAI documents non-visible formatting and channel tokens that may not appear in the reasoning breakdown.
- The dashboard drops non-positive visible throughput from visible distributions. The runner rejects zero visible text before writing `metrics_cloud_v2`, so reasoning-only budget exhaustion disappears from metric distributions entirely.
- The dashboard's table uses visible answer yield, while its speed distribution and time series use generated work rate. Both are useful, but the similar tokens-per-second naming makes them easy to compare as though they had the same meaning.

A read-only production snapshot at 2026-08-04T13:57Z shows that this is already affecting the time series:

| Provider, prior 30 days | Metric rows | Multiple-attempt rows | Final cap differed from nominal 64 | Final request labeled reasoning disabled |
|---|---:|---:|---:|---:|
| Together | 1,182 | 975 | 860 | 119 |
| DeepInfra | 247 | 83 | 52 | 36 |

The counts were changing while the scheduler ran, so they are a point-in-time sample. They are still sufficient to show that field presence or `metrics_schema_version=2` does not imply protocol comparability. The exact budget-exhaustion validation message appeared in 102 error documents since May across all providers, including 83 for Together or DeepInfra; 42 of the 102 were in the prior 30 days. They are error documents, not metric rows, despite containing a legitimate completed generation in many cases.

## Distinct Cases

These cases require different policies:

| Case | What can be known | Correct treatment |
|---|---|---|
| Non-reasoning model | Returned text is normally the generated output, subject to tokenizer differences and non-text formatting. | Use the normal fixed-output profile. A short EOS completion is a completed-short outcome, not automatically an infrastructure error. |
| Reasoning model with configurable effort | The request can pin an effort, but effort is usually soft guidance rather than a token guarantee. | Pin and label one effort in the benchmark profile. Changing effort creates a different profile and time series. Do not lower it as an unrecorded retry. |
| Hybrid model with reasoning on/off | Both modes may be valid product configurations and can have very different latency and quality. | Benchmark them as separate named profiles if both are worth publishing. Never use reasoning-off as fallback data for a reasoning-on series. |
| Reasoning-only model | Reasoning cannot be disabled without changing models or may not be disableable at all. | Allocate a model-specific total cap and retain reasoning-on results, including budget exhaustion. Do not send a generic disable fallback. |
| Hidden reasoning with a reported usage split | Provider total output and reasoning tokens are native-token counts; answer text is observable separately. | Store provider totals and split, plus a separately tokenized answer-text count. Do not assume total minus reasoning equals exact answer text. |
| Hidden reasoning without a reported split | Provider total output may be known, but the amount attributable to reasoning is not. | Store total generated work, answer text, and `token_split_status=unavailable`. Do not infer a reasoning count by mixing tokenizers. |
| Reasoning returned in `reasoning` or `reasoning_content` | Reasoning and answer chunks can be timed and tokenized independently. The reasoning may be visible to the API caller but is still not answer text. | Capture both streams. Record first generated/reasoning token separately from first answer token. Keep provider usage authoritative for billed output. |
| Reasoning embedded in `content` tags | Returned bytes are visible, but answer/reasoning separation requires a model-specific parser. | Split only with a tested parser and record parser provenance. Otherwise mark the answer split unavailable. |
| Budget exhausted before answer text | The API call completed, generation time and billed output can be valid, and answer count is zero. | Store `budget_exhausted_reasoning_only`. Include it in outcome rates and generated-work metrics; exclude it from answer-stream-rate distributions. |
| Budget exhausted after partial answer | First-answer timing and partial answer counts are valid, but the answer target may not have been met. | Store `budget_exhausted_partial_answer`, target status, and all available timing. Do not silently replace it with a larger-cap sample. |
| Empty output without budget exhaustion | This can be refusal, safety filtering, provider schema drift, an unsupported model surface, or a parser bug. | Store a classified non-answer outcome when the response is trustworthy. Use an error only when the response cannot support a valid measurement. |

"The budget was genuinely too small" is not distinguishable from "this model spends heavily on reasoning" without a declared workload and profile. Once the profile is declared, budget exhaustion is a valid result for that profile. A high exhaustion rate can then trigger recalibration and a new profile version. Recalibration must not rewrite the old result.

## What the Industry Does

### Artificial Analysis

[Artificial Analysis's performance methodology](https://artificialanalysis.ai/methodology/performance-benchmarking) reports several dimensions rather than forcing reasoning into one rate:

- Time to first token is the first reasoning token when reasoning is returned.
- Time to first answer token is measured separately after thinking.
- Output speed is tokens received per second after the first token.
- End-to-end response time includes input processing, reasoning, and answer generation.
- Average reasoning tokens are reported separately.
- Performance tokens use a common `o200k_base` tokenizer rather than provider-native tokenizers.
- When all reasoning tokens are not exposed, output speed uses the last 80% of answer chunks as an approximation of answer generation speed.

This is the closest published methodology to the user-facing distinction needed here. It also shows the limitation of a single "output speed" value: reasoning cost and delay are represented by other metrics. The methodology does not say how reasoning-only budget exhaustion enters its public aggregates, and the last-80% technique is an approximation rather than a provider-reported generation rate.

### OpenRouter

[OpenRouter's provider integration documentation](https://openrouter.ai/docs/guides/community/for-providers) defines live throughput as output tokens divided by generation time, where generation time includes fetch latency, TTFT, streaming time, and provider queueing. This is an end-to-end provider work-rate metric, not post-first-answer decode speed. Its [usage accounting documentation](https://openrouter.ai/docs/cookbook/administration/usage-accounting) reports native completion tokens and a reasoning-token detail when available. Its [reasoning documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) treats reasoning as charged output and notes that visible summaries can be shorter than billed reasoning.

OpenRouter's public documentation does not state whether the throughput numerator always includes hidden reasoning usage or is derived from streamed text for every upstream response shape. It also does not describe a separate answer-token throughput metric. Its published formula supports keeping this repository's generated-work rate, but not presenting that rate as sufficient user-experience measurement.

### vLLM

The current [`vllm bench serve` implementation](https://github.com/vllm-project/vllm/blob/main/vllm/benchmarks/serve.py) reports request throughput, total output tokens divided by benchmark duration, TTFT, TPOT, inter-token latency, and end-to-end request latency. For successful requests it prefers a backend-reported output-token count and otherwise tokenizes generated text. Failed requests receive zero output length and are reported separately.

vLLM's [reasoning-output documentation](https://github.com/vllm-project/vllm/blob/main/docs/features/reasoning_outputs.md) describes parsing generated output into `reasoning` and `content`, optional reasoning budgets, and suppression of returned reasoning without suppressing its generation. The serving benchmark does not expose separate answer and reasoning throughput distributions. It is primarily an inference-engine and serving-capacity benchmark, so total generated work is the natural numerator. That is a different product question from time to answer text.

Backend-specific request functions can supply either provider usage or retokenized returned text, so the exact treatment of hidden, non-returned reasoning is not uniform from the top-level benchmark source alone.

### SGLang

SGLang made the serving-benchmark choice explicit in a 2026 fix. [Commit `989a161`](https://github.com/sgl-project/sglang/commit/989a16187dcf8b71d2eee1242d7181a50a80f24e) changed `bench_serving` to concatenate `reasoning_content` and `content`, count the first reasoning chunk as TTFT, and include reasoning-only responses in generated text and token timing. The [current benchmark source](https://github.com/sgl-project/sglang/blob/ee464fed/python/sglang/benchmark/serving.py) computes output throughput from the combined count and TPOT from request latency minus this first combined token.

This avoids falsely reporting zero engine work, but it intentionally does not measure first answer token or answer-stream speed. It is appropriate for serving-engine throughput and insufficient by itself for a developer asking how long a useful answer takes.

### Provider accounting

Primary provider documentation supports treating reasoning as paid generated work while keeping answer text separate:

- [OpenAI](https://developers.openai.com/api/docs/guides/reasoning) says hidden reasoning is billed as output, `max_output_tokens` caps reasoning, visible output, and non-visible formatting together, and an incomplete `max_output_tokens` response can contain no visible answer. Its [token-counting guide](https://developers.openai.com/API/docs/guides/token-counting) warns that output totals can include non-visible formatting not itemized as reasoning.
- [Anthropic](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking) says `max_tokens` covers thinking plus response text, effort is soft guidance, and full thinking is billed even when only a summary or no thinking text is returned. Its [troubleshooting guide](https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting) recommends either more total tokens or lower effort when thinking consumes the cap.
- [Together](https://docs.together.ai/docs/inference/chat/reasoning) distinguishes reasoning-only, hybrid, and adjustable-effort models, returns reasoning separately for many models, bills it as completion tokens, and warns that reasoning can require tens of thousands of tokens. Its [compatibility guide](https://docs.together.ai/docs/inference/openai-compatibility) says usage detail shapes vary by model.
- [DeepInfra](https://docs.deepinfra.com/chat/reasoning) documents effort and enable/disable controls, bills reasoning as output, and says unsupported controls have no effect on non-supporting models. Its [streaming guide](https://docs.deepinfra.com/chat/streaming) confirms final usage is available on streamed requests, but its public docs are less explicit than Together's about reasoning-token detail fields and per-model response shapes.

The industry evidence supports multiple metrics and explicit configuration. It does not support silently changing output cap or reasoning mode until a request yields visible text.

## Recommendation

### Define one primary request, then derive several metrics

Each published sample should be one API request under a versioned benchmark profile. The profile fixes model, provider, workload, answer target, total output cap, reasoning mode, and effort. One streamed response should produce both user-outcome and provider-work metrics.

The default site view should prioritize:

1. Answer-target completion rate.
2. Median time to first answer token.
3. Median time to the fixed answer target, initially 64 standardized answer tokens.
4. Answer stream rate after the first answer token, where streaming makes it observable.
5. Provider-reported output tokens per full request second, clearly labeled as including reasoning and queueing.
6. Median reasoning tokens and reasoning share where the provider reports them.

The current answer-yield metric can remain available as `answer_tokens / full request duration`, but it should be labeled "answer yield" rather than "visible speed." A reasoning-only result has answer yield zero, generated-work rate nonzero, and no answer-stream rate.

### Do not retry primary measurements with a different protocol

For reasoning budget exhaustion:

- Write the primary request to `metrics_cloud_v2` with a budget-exhaustion outcome.
- Do not write it to `errors_cloud` unless token/timing data is unusable.
- Do not immediately retry it into the same published sample.
- Do not disable reasoning as fallback.
- Optionally schedule a diagnostic request with a larger predeclared cap. Store every diagnostic attempt as a separate row with `sample_role="diagnostic"` and a parent measurement ID. Exclude diagnostics from public distributions.

Retries for transient transport failures may repeat the exact same profile, but they must not select only the final successful latency as though earlier attempts did not happen. Either publish each API request as its own attempt or keep retry behavior in scheduler health data and publish only explicitly identified primary attempts. Protocol-changing retries are never transport retries.

### Pin reasoning behavior per model profile

Use these policies:

| Model capability | Published profile policy |
|---|---|
| Non-reasoning | Reasoning off or unsupported, a 64-answer-token target, and enough fixed output headroom to reach it under the protocol tokenizer. |
| Adjustable effort | Reasoning enabled with one explicit documented effort. Start with the provider/model recommended normal effort, not an emergency low effort. |
| Hybrid | Publish reasoning-on and reasoning-off only as separate profile IDs. If only one is affordable, choose and label one; do not blend them. |
| Reasoning-only | Reasoning enabled with a calibrated total cap and lower cadence if cost requires it. |
| Fixed reasoning budget | Set the reasoning budget explicitly and set total output cap high enough to reserve the 64-answer-token target. |
| Provider default only | Record `reasoning_mode_requested="provider_default"`; do not claim an effective effort the API did not confirm. |

Provider comparisons for the same canonical model should use the same benchmark profile. If a provider cannot honor the profile, report it as unsupported rather than silently changing the request.

### Calibrate output budgets, then freeze them

A global 64-token total cap should be retired. Keep 64 standardized answer tokens as the initial latency target, but give non-reasoning profiles a fixed cap with tokenizer and stream-measurement headroom, initially 128 provider-native tokens. Reasoning profiles need a larger total cap selected before publication.

Use a small set of reviewable cap tiers, for example 512, 2,048, 8,192, and 32,768 tokens, bounded by provider and model limits. For a new reasoning profile:

1. Start from provider guidance or the closest model-family profile.
2. Run at least five diagnostic samples of the exact benchmark workload and reasoning configuration.
3. Select the smallest tier for which all calibration samples produce the 64-answer-token target. A length finish after the target is acceptable for this sustained-generation workload; a length finish before the target is a budget-exhausted result.
4. Use the same selected tier across providers serving the same canonical model.
5. If no affordable tier works reliably, publish the model as budget-sensitive or unsupported for this workload rather than manufacturing a reasoning-off success.
6. Freeze the cap in a profile ID. A later cap or effort change creates a new profile and a visible time-series boundary.

Five samples are enough to catch deterministic under-budget behavior, not to estimate a tail percentile. The ongoing primary outcome rate remains the better tail signal. Sustained budget-exhaustion rate should open a profile review; it should not trigger per-run adaptive cap changes.

Large reasoning caps increase benchmark cost. Reduce cadence for expensive reasoning profiles instead of imposing a cap that usually prevents answers. Cadence is a sampling decision and should not alter the request protocol.

## Schema and Code Changes

The existing visible/generated/legacy split should remain for compatibility, but it is not sufficient. Add schema-v3 fields to `metrics_cloud_v2`; do not create another success-only collection.

### Proposed fields

| Field | Meaning |
|---|---|
| `metrics_schema_version` | Set to `3` for the new contract. |
| `benchmark_protocol_version` | Version of outcome, timing, and publication rules. |
| `benchmark_profile_id` | Stable identity for workload, cap, target, reasoning mode, and effort. |
| `workload_id` | Stable prompt/workload identity. |
| `sample_role` | `primary` or `diagnostic`; only primary enters public aggregates. |
| `attempt_group_id` | Links exact-profile retries and diagnostic requests without collapsing them. |
| `attempt_index` | One-based API request index within the group. |
| `parent_measurement_id` | Links a diagnostic request to the primary measurement that motivated it. |
| `answer_target_tokens` | Standardized answer-text target, initially 64. |
| `max_output_tokens` | Actual total cap sent on this API request. Keep `requested_tokens` only as a legacy field. |
| `reasoning_mode_requested` | `disabled`, `enabled`, `adaptive`, `provider_default`, or `unsupported`. |
| `reasoning_effort_requested` | Explicit effort sent, nullable. Do not call it effective without confirmation. |
| `measurement_outcome` | See the outcome values below. |
| `answer_target_met` | Whether answer text reached the profile target. |
| `provider_output_tokens` | Provider-native total generated output usage, including billed reasoning and other hidden output. |
| `provider_reasoning_tokens` | Provider-native reasoning/thinking usage when explicitly reported. |
| `provider_non_reasoning_output_tokens` | `provider_output_tokens - provider_reasoning_tokens` when both exist. This is not asserted to be exact answer text. |
| `answer_tokens_standardized` | Tokens in answer text counted with one declared cross-provider tokenizer. |
| `answer_tokenizer` | Tokenizer/version used for the standardized answer count. Pin v3 to `tiktoken:o200k_base` and change the protocol if it changes. |
| `returned_reasoning_tokens_standardized` | Client count of returned reasoning text, nullable; not a billing count. |
| `token_split_status` | `provider_reported`, `returned_text_only`, `unavailable`, or `parser_failed`. |
| `request_duration_seconds` | Request send to terminal response. Existing `generate_time` remains its compatibility alias. |
| `time_to_first_generated_token_seconds` | First reasoning or answer delta, when observable. |
| `time_to_first_reasoning_token_seconds` | First separate reasoning delta, nullable. |
| `time_to_first_answer_token_seconds` | First answer-text delta, nullable. Existing `time_to_first_token` remains its compatibility alias. |
| `time_to_answer_target_seconds` | Request send to the answer target, nullable. |
| `answer_stream_duration_seconds` | First answer token to last measured answer token, nullable. |
| `provider_output_tokens_per_second_e2e` | Provider output tokens divided by full request duration. |
| `answer_tokens_per_second_e2e` | Standardized answer tokens divided by full request duration. This is answer yield. |
| `answer_stream_tokens_per_second` | Standardized answer tokens observed after the first answer event divided by time from the first to last answer event. |
| `answer_stream_rate_method` | Provenance for chunk bundling and rate calculation, initially `post_first_event_o200k`. |

Recommended outcome values are:

- `answer_target_met`
- `completed_short`
- `budget_exhausted_partial_answer`
- `budget_exhausted_reasoning_only`
- `non_answer_response`

Transport, authentication, rate-limit, provider 5xx, timeout, and parser failures without trustworthy metrics remain in `errors_cloud`. A completed response with trustworthy duration and usage belongs in `metrics_cloud_v2`, even if its outcome is not answer success.

### Compatibility fields

For schema-v3 rows:

- Keep `output_tokens`, `generated_output_tokens`, `tokens_per_second`, and `generated_tokens_per_second` as aliases for provider-reported generated output and its full-request rate when available.
- Keep `generate_time` as an alias for full request duration.
- Keep `time_to_first_token` as an alias for first answer token, preserving the documented schema-v2 intent.
- Set `requested_tokens` to the actual v3 total cap; `answer_target_tokens` carries the separate answer target.
- Continue writing `visible_output_tokens` and `visible_tokens_per_second` as compatibility aliases for standardized answer count and full-request answer yield. New dashboard code must use the explicit fields, and clients must not aggregate these aliases across v2 and v3.
- Keep `reasoning_tokens` as an alias only for explicitly provider-reported reasoning. Stop inferring it by subtracting counts from different tokenizers.
- Preserve `token_source` and add explicit token unit/provenance through the new fields.

### Runner changes implied

The implementation should make these focused changes:

1. Replace `MAX_TOKENS` as the universal request protocol with a model/profile resolver. Store reviewed benchmark-profile metadata with model runner metadata and require matching profile settings for the same canonical model across providers.
2. Remove budget escalation and reasoning-disable fallback from `openai_compat.py` primary runs.
3. Use streaming for Together, DeepInfra, Fireworks, and Groq where their APIs return final usage. Parse `delta.reasoning`, `delta.reasoning_content`, and `delta.content` as separate channels.
4. Build one metric row per API request. Do not return only the last attempt.
5. Classify terminal responses into measurement outcomes before runner validation.
6. Let budget-exhausted measurements pass validation when duration and generated usage are valid. Require answer text only for answer-rate and answer-speed metrics, not for generated-work metrics.
7. Reserve `errors_cloud` and scheduler failure counters for failures to obtain a valid measurement. Track answer-target outcome separately from service health so an always-reasoning-only profile is visible without making the scheduler treat the provider as down.
8. Extend `OPTIONAL_METRIC_FIELDS`, the dashboard Mongoose schema, API projections, static-file generation, and TypeScript types with the new fields.
9. Update dashboard aggregation to filter `sample_role="primary"` and a single `benchmark_profile_id`. Do not average across caps, efforts, or protocol versions.
10. Show budget-exhausted outcomes in completion-rate and cap-exhaustion views. Leave answer-stream speed and first-answer latency null rather than zero when no answer arrived.

The first implementation does not need a general experiment framework. A small explicit profile object on each reviewed model record, a stable profile ID, and validation that prevents cross-provider drift are enough.

The reviewed model metadata can be one object:

```json
{
  "benchmark_profile": {
    "profile_id": "story-v3-answer64-reasoning-medium-cap8192",
    "workload_id": "long-happy-world-story-v1",
    "answer_target_tokens": 64,
    "max_output_tokens": 8192,
    "reasoning_mode": "enabled",
    "reasoning_effort": "medium"
  }
}
```

Discovery should not infer or overwrite this object. A reviewed profile can be copied to each provider record for the canonical model, with validation that the protocol fields match.

## Migration Story

Do not backfill semantics that were not observed. Historical rows lack first-answer streaming times, and many failed reasoning-only calls retain no metric payload. Existing visible counts also use mixed token-count methods.

Use this migration:

1. Add schema-v3 readers and dashboard support before changing the runner.
2. Run bounded diagnostic samples for the affected reasoning models to select profiles. Mark them `sample_role="diagnostic"`; do not publish them in current distributions.
3. Cut primary collection over to protocol v3 on one recorded date. Continue dual-writing legacy alias fields in each v3 row, but do not run both old and new request protocols for every cron cycle.
4. Make public aggregates select the latest profile for each provider/model and never merge v2 and v3 samples. Show "data since protocol change" until the selected window fills.
5. Keep the old series accessible as "legacy protocol" with its known cap/mode mixture. Add a visible break in time-series charts rather than joining lines across the change.
6. After 14 days, the default 14-day view is naturally all v3. After 30 days, the 30-day distributions are naturally all v3. No destructive Mongo migration is required.
7. Keep old API clients working through the alias fields. New clients should require `benchmark_protocol_version`, `benchmark_profile_id`, and explicit metric names.

Recent schema-v2 rows cannot safely form one clean legacy cohort. If the dashboard needs a historical comparison, it can segment rows by existing `request_mode`, `max_output_tokens_attempted`, and `reasoning_effort`, with missing values labeled unknown. It should not merge those cohorts or claim that all nominally requested 64 tokens represent the same workload.

Changing a cap, answer target, workload prompt, reasoning mode, effort, tokenizer, stream timing rule, or outcome rule creates a new profile or protocol version. Provider infrastructure changes do not; those are what the stable profile is intended to reveal.

## What Is Uncertain

- The current long-story prompt may be useful for sustained generation but is not representative of tasks for which developers choose reasoning models. This proposal deliberately keeps workload design separate so the measurement contract can be fixed first.
- A 64-answer-token target is inherited from the current benchmark, not established as a user-utility threshold. It is a stable latency landmark only.
- Artificial Analysis documents its last-80%-of-answer-chunks method for hidden reasoning, but not how no-answer budget exhaustion affects public aggregates.
- OpenRouter defines its throughput denominator but does not publicly specify the exact output-token source for every reasoning-provider response shape. It is unclear when hidden reasoning is included in that live numerator.
- vLLM's top-level benchmark can receive provider usage or retokenized returned text depending on backend. Hidden reasoning treatment therefore depends on the backend adapter even though the aggregate metric has one name.
- Together documents per-model variation in usage detail shape. DeepInfra documents reasoning billing and controls but is thin on the exact reasoning usage fields returned by every model. Live contract tests are needed before trusting a split.
- A provider can accept a reasoning-control parameter and silently ignore or remap it. Request metadata proves what was asked for, not what the model did. Response evidence and provider capability metadata should be kept separate.
- Provider-native output token rates are not perfectly comparable across model families because tokenizers differ. They remain useful for comparing providers serving the same model and for understanding billing. Standardized answer-token metrics are better for cross-model display.
- Client-side cancellation after exactly 64 answer tokens could normalize answer length, but it can lose final provider usage and produce inconsistent billing. This proposal measures time to the target while allowing the declared request to finish.
- Streaming events can bundle several tokens, so first-token and post-first-token rates are not exact token-level decode traces. Retokenizing answer chunks and recording the rate method is more honest than treating each chunk as one token.
- Calibration with five samples catches obvious under-budget profiles but does not estimate p95 or p99 reasoning demand. The published exhaustion rate should remain visible rather than being optimized away.

These uncertainties argue for explicit provenance and multiple metrics, not for another automatic retry heuristic.
