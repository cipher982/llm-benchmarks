from __future__ import annotations

import hashlib
import json
import os
import signal
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from multiprocessing import Queue
from multiprocessing import get_context
from typing import Any

import dotenv
from pymongo import MongoClient

from llm_bench.cloud.visible_tokens import VISIBLE_TOKEN_MARK
from llm_bench.config import CloudConfig
from llm_bench.logging import log_mongo
from llm_bench.ops.error_rollups import upsert_error_rollup
from llm_bench.ops.error_taxonomy import classify_error
from llm_bench.scheduler import policies
from llm_bench.scheduler.mongo import PUBLISHED_PROFILE_ID
from llm_bench.scheduler.mongo import error_rollups_collection_name
from llm_bench.scheduler.mongo import errors_collection_name
from llm_bench.scheduler.mongo import metrics_collection_name
from llm_bench.scheduler.mongo import mongo_client
from llm_bench.scheduler.mongo import mongo_env
from llm_bench.scheduler.mongo import probe_metrics_collection_name
from llm_bench.scheduler.routing import DIRECT_TRANSPORT
from llm_bench.scheduler.routing import OPENROUTER_TRANSPORT
from llm_bench.scheduler.routing import OR_SERVED_POLICY
from llm_bench.scheduler.routing import RouteDecision
from llm_bench.scheduler.routing import resolve_job_route
from llm_bench.utils import get_current_timestamp

dotenv.load_dotenv(".env")

QUERY_TEXT = "Tell a long and happy story about the history of the world."
# Headroom for thinking, not a target. The stream closes at the 64th visible
# token, so a model that does not reason still generates ~70 tokens here; the
# budget only has to be large enough that a reasoning model reaches visible
# text at all. Measured reasoning use: p50 339, p90 1,446, p95 1,685 tokens.
MAX_TOKENS = int(os.getenv("BENCHMARK_MAX_TOKENS", "2048"))
TEMPERATURE = 0.1

PROVIDER_MODULES: dict[str, str] = {
    "openai": "llm_bench.cloud.providers.openai",
    "anthropic": "llm_bench.cloud.providers.anthropic",
    "bedrock": "llm_bench.cloud.providers.bedrock",
    "vertex": "llm_bench.cloud.providers.vertex",
    "anyscale": "llm_bench.cloud.providers.anyscale",
    "together": "llm_bench.cloud.providers.together",
    "openrouter": "llm_bench.cloud.providers.openrouter",
    "azure": "llm_bench.cloud.providers.azure",
    "runpod": "llm_bench.cloud.providers.runpod",
    "fireworks": "llm_bench.cloud.providers.fireworks",
    "deepinfra": "llm_bench.cloud.providers.deepinfra",
    "groq": "llm_bench.cloud.providers.groq",
    "databricks": "llm_bench.cloud.providers.databricks",
    "lambda": "llm_bench.cloud.providers.lambda",
    "cerebras": "llm_bench.cloud.providers.cerebras",
}

VARIABLE_OUTPUT_PROVIDERS = ("openai", "openrouter", "deepinfra", "fireworks", "together", "groq", "vertex")

_PROVIDER_MODULES_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class RunnerResult:
    status: str
    error_kind: str | None = None
    error_message: str | None = None
    # Which series this sample belongs to. The worker uses it to decide whether
    # the run counts as the model being measured; a probe must not create the
    # appearance of freshness for a model that is not actually being published.
    sample_role: str = "published"
    transport_provider: str = DIRECT_TRANSPORT
    route_attempted: bool = False
    fallback_reason: str | None = None


def load_provider_func(provider: str):
    if provider not in PROVIDER_MODULES:
        raise ValueError(f"Unsupported provider: {provider}")
    if provider in _PROVIDER_MODULES_CACHE:
        return _PROVIDER_MODULES_CACHE[provider]
    module = __import__(PROVIDER_MODULES[provider], fromlist=["generate"])  # type: ignore
    generate_func = module.generate
    _PROVIDER_MODULES_CACHE[provider] = generate_func
    return generate_func


def validate_metrics(provider: str, metrics: dict[str, Any], max_tokens: int) -> tuple[bool, str | None]:
    """Did this run produce the measurement, and is it visible answer text?

    There used to be a second policy asking whether `output_tokens` landed
    within 10% of the requested budget — a check for providers that ignore
    `max_tokens`. The runner now closes the stream at the 64th visible token,
    so every run is truncated deliberately and no run is near its budget. The
    question the benchmark actually needs answered is whether visible text was
    delivered, which is the same question for every provider.
    """
    if not isinstance(metrics, dict):
        return False, "metrics not a dict"
    required = ["output_tokens", "generate_time", "tokens_per_second"]
    for key in required:
        if key not in metrics:
            return False, f"missing metric: {key}"
    if metrics["tokens_per_second"] <= 0:
        return False, "tokens_per_second <= 0"
    out = metrics["output_tokens"]
    if not isinstance(out, int):
        return False, "output_tokens not int"
    if metrics.get("visible_text_empty") is True:
        if metrics.get("response_status") == "incomplete" or metrics.get("finish_reason") in (
            "length",
            "max_output_tokens",
        ):
            return False, "visible output empty after token budget was exhausted; retry with a larger output budget"
        return False, "visible output text is empty"
    visible_out = metrics.get("visible_output_tokens")
    if visible_out is not None and visible_out <= 0:
        return False, f"visible_output_tokens {visible_out} <= 0"
    if out <= 0:
        return False, f"output_tokens {out} <= 0 for {provider}"
    return True, None


def validation_policy(provider: str) -> str:
    return "visible_nonzero"


def log_error_mongo(
    *,
    provider: str,
    model_id: str,
    stage: str,
    message: str,
    tb: str | None = None,
    exc_type: str = "",
    client: MongoClient | None = None,
    db_name: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> str:
    uri, default_db = mongo_env()
    db_name = db_name or default_db
    should_close = False
    if client is None:
        client = MongoClient(uri)
        should_close = True
    try:
        classified = classify_error(message=message, exc_type=exc_type)
        fingerprint = classified.fingerprint(provider=provider, model=model_id, stage=stage)
        doc = {
            "provider": provider,
            "model_name": model_id,
            "ts": datetime.now(timezone.utc),
            "stage": stage,
            "message": message,
            "error_kind": classified.kind.value,
            "http_status": classified.http_status,
            "provider_error_code": classified.provider_error_code,
            "normalized_message": classified.normalized_message,
            "fingerprint": fingerprint,
        }
        if tb:
            doc["traceback"] = tb
        if provenance:
            for key in (
                "source_provider",
                "source_model_id",
                "transport_provider",
                "transport_model_id",
                "route_model_id",
                "route_provider_slug",
                "observed_provider",
                "observed_provider_slug",
                "route_policy",
                "route_snapshot_at",
                "route_probe_id",
                "route_decision_version",
                "route_canary_id",
                "route_canary_state",
                "route_canary_successes",
                "route_canary_required_successes",
                "route_canary_cost_status",
                "route_canary_evidence_uri",
                "route_canary_evidence_sha256",
                "route_canary_promotion_gate",
                "route_expires_at",
                "route_revocation_generation",
                "profile_id",
                "profile_hash",
                "effective_request_hash",
                "route_state",
                "route_reason",
                "transport_attempt",
            ):
                if key in provenance and provenance[key] is not None:
                    doc[key] = provenance[key]
        client[db_name][errors_collection_name()].insert_one(doc)
        upsert_error_rollup(
            collection=client[db_name][error_rollups_collection_name()],
            fingerprint=fingerprint,
            provider=provider,
            model_name=model_id,
            stage=stage,
            error_kind=classified.kind.value,
            normalized_message=classified.normalized_message,
        )
        return classified.kind.value
    finally:
        if should_close:
            client.close()


# Sample roles. Only PUBLISHED reaches the site and counts as freshness; a
# probe is an admission check, not a measurement of a model we publish.
SAMPLE_ROLE_PUBLISHED = "published"
SAMPLE_ROLE_PROBE = "probe"
NON_PUBLISHING_ROLES = frozenset({SAMPLE_ROLE_PROBE})


def publishes(sample_role: str | None) -> bool:
    """Only the published role reaches the site and counts as freshness.

    Stated as an allowlist rather than a denylist because the denylist was
    wrong the moment the shadow and long roles were deleted from it: jobs
    carrying those roles still exist in the queue, and under a denylist a
    resurrected one would write to the published collection and mark the model
    fresh — measured under a profile that no longer exists. An unrecognised
    role is not publishable.
    """
    return sample_role == SAMPLE_ROLE_PUBLISHED


# The measurement protocol a row was produced under, so rows measured under
# different rules are never silently averaged together. Defined in policies
# because the queue reads it too: a terminal verdict about the measurement is
# only evidence while the protocol that produced it still holds.
PROTOCOL_VERSION = policies.MEASUREMENT_PROTOCOL_VERSION
DEFAULT_PROFILE_ID = PUBLISHED_PROFILE_ID
OPENROUTER_ROUTING_ENABLED_ENV = "OPENROUTER_ROUTING_ENABLED"

# One profile. There were three: a 64-token default, a 512-token `cloud-long-v1`
# and a 2048-token `cloud-reasoning-v1` shadow. Both extras existed only because
# a 64-token budget could not measure a reasoning model — it spends the whole
# budget thinking and emits nothing visible. The default now carries the
# headroom and the stream closes at the 64th visible token, so a second budget
# buys nothing: every model is measured the same way, on one series, and the
# question of whether reasoning rows belong on the same axis stops existing.
BENCHMARK_PROFILES: dict[str, dict[str, Any]] = {
    DEFAULT_PROFILE_ID: {"max_tokens": MAX_TOKENS},
}


def profile_max_tokens(profile_id: str) -> int:
    """The token budget a profile asks for, falling back to the default.

    An unknown profile id must not silently become a different measurement, so
    it resolves to the default rather than to zero or to whatever it names.
    """
    return int(BENCHMARK_PROFILES.get(profile_id, BENCHMARK_PROFILES[DEFAULT_PROFILE_ID])["max_tokens"])


# Lanes whose stream loop stops at the 64th visible token. A budget is only
# safe to raise where the runner can decline to consume it: on these lanes the
# budget is headroom for thinking and the run ends at the mark, while a lane
# that reads to end-of-stream would generate — and be billed for — the whole
# 2048 tokens on every model that already answers fine at 64.
EARLY_STOP_PROVIDERS = frozenset({"openrouter"})
UNCAPPED_LANE_MAX_TOKENS = int(os.getenv("BENCHMARK_UNCAPPED_LANE_MAX_TOKENS", "64"))

# What one measurement may cost. A token budget is the wrong unit to bound
# spend in, because the same 2048 tokens cost a thousandth of a cent on a small
# open model and 35 cents on a premium reasoning model: gpt-5.2-pro ran to the
# full budget three times and was 67% of a day's spend on its own. Closing the
# stream early does not always help either — some upstreams finish and bill the
# whole generation whatever the client does, billing 2008 tokens against the 64
# we read.
MAX_COST_PER_RUN_USD = float(os.getenv("BENCHMARK_MAX_COST_PER_RUN_USD", "0.01"))


def affordable_max_tokens(completion_price_per_token: float | None, profile_budget: int) -> int:
    """The budget this model's price allows, floored at the measurement itself.

    Below the 64-visible-token mark there is no measurement to buy, so the floor
    is the mark rather than zero — a model too expensive to profile is measured
    at the minimum instead of silently dropped.

    An unknown price fails closed. It used to return the full budget, which made
    the whole clamp best-effort: a model missing from the catalogue, a sentinel
    price, or one Mongo timeout would send the same 2048-token request this
    exists to prevent, and gpt-5.2-pro at 2048 was 67% of a day's spend. A price
    we could not read is not evidence the model is cheap.

    A price of exactly zero is different from an unknown one: the catalogue says
    the model is free, so there is nothing to clamp and it gets the full budget.
    """
    # The floor is the measurement, but it is still bounded by the profile: a
    # profile deliberately configured below the mark must not be raised past
    # what it asked for.
    floor = min(profile_budget, VISIBLE_TOKEN_MARK)
    if completion_price_per_token is None:
        return floor
    if completion_price_per_token <= 0:
        return profile_budget
    affordable = int(MAX_COST_PER_RUN_USD / completion_price_per_token)
    return max(floor, min(profile_budget, affordable))


def lane_max_tokens(
    transport_provider: str,
    profile_budget: int,
    *,
    completion_price_per_token: float | None = None,
) -> int:
    """Clamp the profile's budget to what this lane can stop consuming, and afford."""
    if transport_provider not in EARLY_STOP_PROVIDERS:
        return min(profile_budget, UNCAPPED_LANE_MAX_TOKENS)
    return affordable_max_tokens(completion_price_per_token, profile_budget)


_PRICE_CACHE: dict[str, float | None] = {}


def completion_price(model_id: str) -> float | None:
    """Dollars per completion token, from the catalogue OpenRouter publishes.

    Cached for the life of the process: prices move on the order of weeks and
    this is read on the path of every run. A few catalogue rows carry sentinel
    negative prices (the auto-router and BYOK entries), which are not a price
    and must not be read as a cheap one.
    """
    if model_id in _PRICE_CACHE:
        return _PRICE_CACHE[model_id]
    try:
        _, db_name = mongo_env()
        client = mongo_client()
        try:
            doc = client[db_name]["openrouter_catalog"].find_one({"openrouter_id": model_id}, {"pricing": 1})
        finally:
            client.close()
    except Exception as exc:  # noqa: BLE001
        # A lookup failure is not an answer about the price, so it is not
        # cached: caching it would let one timeout hold the model at the
        # fail-closed floor for the rest of the process.
        print(f"price lookup failed for {model_id}: {type(exc).__name__}: {exc}", flush=True)
        return None
    raw = (doc or {}).get("pricing", {}).get("completion")
    price: float | None = None
    if raw is not None:
        value = float(raw)
        # Negative is a sentinel the catalogue uses for the auto-router and BYOK
        # entries, not a price. Zero is a real answer: the model is free.
        price = value if value >= 0 else None
    _PRICE_CACHE[model_id] = price
    return price


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def log_success_mongo(config: CloudConfig, metrics: dict[str, Any], *, sample_role: str) -> None:
    """Write one sample to the collection its role belongs in."""
    uri, db_name = mongo_env()
    collection = metrics_collection_name() if publishes(sample_role) else probe_metrics_collection_name()
    log_mongo(
        model_type="cloud",
        config=config,
        metrics=metrics,
        uri=uri,
        db_name=db_name,
        collection_name=collection,
    )


class AttemptFailure(RuntimeError):
    """A generation or validation failure for one transport attempt."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage
        self.message = message


def openrouter_routing_enabled() -> bool:
    """Return the explicit runtime gate for production route activation."""

    return os.getenv(OPENROUTER_ROUTING_ENABLED_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def job_requires_openrouter(job: dict[str, Any]) -> bool:
    """Return whether this queued job should consume the shared OR lane."""

    decision = resolve_job_route(job)
    # OpenRouter-native catalogue rows have no source-provider route snapshot;
    # their direct adapter is OpenRouter itself and still needs the shared quota
    # gate. Existing source rows consume the gate only when routing is enabled.
    return str(job.get("provider") or "") == "openrouter" or (
        openrouter_routing_enabled() and decision.transport_provider == OPENROUTER_TRANSPORT
    )


def is_openrouter_native_job(job: dict[str, Any]) -> bool:
    """Whether quota exhaustion cannot fall back to a different provider."""

    return str(job.get("provider") or "") == "openrouter"


def effective_job_route(job: dict[str, Any]) -> RouteDecision:
    decision = resolve_job_route(job)
    if decision.transport_provider == OPENROUTER_TRANSPORT and not openrouter_routing_enabled():
        return RouteDecision.direct(
            decision.source_provider,
            decision.source_model_id,
            reason="route-activation-disabled",
        )
    return decision


def _transport_provider(decision: RouteDecision) -> str:
    return decision.source_provider if decision.transport_provider == DIRECT_TRANSPORT else decision.transport_provider


def _config_for_decision(
    decision: RouteDecision,
    *,
    run_ts: str,
    timeout_seconds: float | None = None,
) -> CloudConfig:
    provider = _transport_provider(decision)
    model_id = decision.transport_model_id or decision.source_model_id
    misc: dict[str, Any] = {}
    if decision.transport_provider == OPENROUTER_TRANSPORT:
        misc["route_provider_slug"] = decision.route_provider_slug
        misc["route_model_id"] = decision.route_model_id
        misc["route_probe_id"] = decision.route_probe_id
        misc["route_snapshot_at"] = decision.route_snapshot_at
        misc["route_policy"] = decision.route_policy
        if decision.route_reasoning_exclude:
            misc["route_reasoning_exclude"] = True
    if timeout_seconds is not None:
        misc["timeout_seconds"] = timeout_seconds
        misc["bounded_timeout"] = True
    return CloudConfig(
        provider=provider,
        model_name=model_id,
        run_ts=run_ts,
        temperature=TEMPERATURE,
        misc=misc,
        source_provider=decision.source_provider,
        source_model_id=decision.source_model_id,
        transport_provider=provider,
        transport_model_id=model_id,
    )


def _generate_and_validate(
    decision: RouteDecision,
    *,
    run_ts: str,
    run_config: dict[str, Any],
    max_tokens: int,
    timeout_seconds: float | None = None,
) -> tuple[CloudConfig, dict[str, Any]]:
    config = _config_for_decision(decision, run_ts=run_ts, timeout_seconds=timeout_seconds)
    provider = _transport_provider(decision)
    try:
        generate = load_provider_func(provider)
        metrics = generate(config, run_config)
    except Exception as exc:
        raise AttemptFailure("generate", f"{type(exc).__name__}: {exc}") from exc

    if not metrics:
        raise AttemptFailure("validate", "empty metrics")

    validation_provider = _transport_provider(decision)
    ok, why = validate_metrics(validation_provider, metrics, max_tokens)
    if not ok:
        raise AttemptFailure("validate", str(why))

    if decision.transport_provider == OPENROUTER_TRANSPORT:
        observed_slug = metrics.get("observed_provider_slug")
        if metrics.get("provider_metadata_verified") is not True:
            raise AttemptFailure("validate", "OpenRouter provider metadata was not verified")
        # Marketplace lanes measure OpenRouter's default routing, which may
        # rotate upstreams between runs; the observed provider is a fact about
        # this run, not a match to enforce. Pinned lanes still must match.
        if decision.route_policy != OR_SERVED_POLICY and observed_slug != decision.route_provider_slug:
            raise AttemptFailure(
                "validate",
                f"observed provider {observed_slug!r} does not match route {decision.route_provider_slug!r}",
            )
    return config, metrics


def _record_attempt_failure(
    *,
    provider: str,
    model_id: str,
    failure: AttemptFailure,
    stage_prefix: str = "",
    decision: RouteDecision | None = None,
    provenance_extra: dict[str, Any] | None = None,
) -> str:
    stage = f"{stage_prefix}_{failure.stage}" if stage_prefix else failure.stage
    provenance = decision.metric_fields() if decision is not None else {}
    if provenance_extra:
        provenance.update(provenance_extra)
    try:
        return log_error_mongo(
            provider=provider,
            model_id=model_id,
            stage=stage,
            message=failure.message,
            tb=traceback.format_exc(limit=5),
            exc_type=type(failure).__name__,
            provenance=provenance or None,
        )
    except Exception as exc:
        print(
            f"Failed to log {stage} error for {provider}:{model_id}: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return "unknown"


def _annotate_metrics(
    metrics: dict[str, Any],
    decision: RouteDecision,
    *,
    fallback_reason: str | None = None,
) -> None:
    route_fields = decision.metric_fields()
    for key, value in route_fields.items():
        # The completed request is authoritative for observed provider
        # identity. The snapshot's observed identity is only an eligibility
        # expectation and must not overwrite what this sample actually saw.
        if key in {"observed_provider", "observed_provider_slug"} and metrics.get(key):
            continue
        metrics[key] = value
    metrics["provider_metadata_verified"] = bool(
        decision.transport_provider == OPENROUTER_TRANSPORT and metrics.get("observed_provider_slug")
    )
    if fallback_reason:
        metrics["fallback_reason"] = fallback_reason


def run_benchmark_job(job: dict[str, Any]) -> RunnerResult:
    provider = str(job["provider"])
    model_id = str(job["model_id"])
    if job.get("job_kind") == "smoke_hang":
        smoke_seconds = job.get("smoke_seconds")
        seconds = 300 if smoke_seconds is None else int(smoke_seconds)
        print(f"Smoke hang job sleeping for {seconds}s: {provider}:{model_id}", flush=True)
        time.sleep(seconds)
        return RunnerResult(status="success")

    sample_role = str(job.get("sample_role") or SAMPLE_ROLE_PUBLISHED)
    profile_id = str(job.get("benchmark_profile_id") or DEFAULT_PROFILE_ID)
    decision = effective_job_route(job)

    # The lane and the price decide the budget: the profile asks for thinking
    # headroom, only a lane that stops at the measurement mark is allowed to be
    # given it, and only for as many tokens as one run may cost. Keyed on the
    # provider whose code actually runs, not on the route label — a native
    # OpenRouter row routes as "direct" and would otherwise be clamped to 64 on
    # the one lane the headroom exists for.
    def budget_for(route: RouteDecision) -> int:
        """The budget this route may be given. Recomputed per route, never reused.

        A route decides the budget, so a fallback to a different lane has to ask
        again. Computing it once for the OpenRouter attempt and passing the same
        run_config to the direct fallback handed a 2048-token budget — safe only
        because OpenRouter stops at the mark — to an adapter that reads to
        end-of-stream, on a directly billed key.
        """
        route_lane = _transport_provider(route)
        return lane_max_tokens(
            route_lane,
            profile_max_tokens(profile_id),
            completion_price_per_token=(
                completion_price(route.transport_model_id or route.source_model_id)
                if route_lane in EARLY_STOP_PROVIDERS
                else None
            ),
        )

    max_tokens = budget_for(decision)
    deadline_seconds = int(job.get("deadline_seconds") or policies.DEFAULT_DEADLINE_SECONDS)
    run_ts = get_current_timestamp()
    run_config = {
        "query": QUERY_TEXT,
        "max_tokens": max_tokens,
    }

    route_attempted = decision.transport_provider == OPENROUTER_TRANSPORT
    fallback_reason = job.get("route_fallback_reason")
    # Failures from a non-default profile carry the profile on the error row,
    # so quarantine and reliability tooling can tell "the model is broken" from
    # "the model serves 64 tokens but not this profile's budget".
    profile_provenance = {"profile_id": profile_id} if profile_id != DEFAULT_PROFILE_ID else None

    try:
        route_timeout_seconds = None
        if decision.transport_provider == OPENROUTER_TRANSPORT:
            configured = float(os.getenv("OPENROUTER_ROUTE_TIMEOUT_SECONDS", "45"))
            route_timeout_seconds = min(configured, max(5.0, deadline_seconds / 3.0))
        model_config, metrics = _generate_and_validate(
            decision,
            run_ts=run_ts,
            run_config=run_config,
            max_tokens=max_tokens,
            timeout_seconds=route_timeout_seconds,
        )
    except AttemptFailure as failure:
        if not route_attempted:
            error_kind = _record_attempt_failure(
                provider=provider, model_id=model_id, failure=failure, provenance_extra=profile_provenance
            )
            print(f"Error {provider}:{model_id} - {failure.message}", flush=True)
            return RunnerResult(
                status="error",
                error_kind=error_kind,
                error_message=failure.message,
                transport_provider=decision.transport_provider,
                fallback_reason=fallback_reason,
            )

        _record_attempt_failure(
            provider=provider,
            model_id=model_id,
            failure=failure,
            stage_prefix="route",
            decision=decision,
            provenance_extra=profile_provenance,
        )
        fallback_reason = failure.message
        decision = RouteDecision.direct(provider, model_id, reason="route-fallback")
        # A different lane, so a different budget. The direct adapters read to
        # end-of-stream and must not inherit OpenRouter's thinking headroom.
        max_tokens = budget_for(decision)
        run_config = {**run_config, "max_tokens": max_tokens}
        try:
            model_config, metrics = _generate_and_validate(
                decision,
                run_ts=run_ts,
                run_config=run_config,
                max_tokens=max_tokens,
            )
        except AttemptFailure as direct_failure:
            error_kind = _record_attempt_failure(
                provider=provider,
                model_id=model_id,
                failure=direct_failure,
                decision=decision,
                provenance_extra={**(profile_provenance or {}), "fallback_reason": fallback_reason},
            )
            print(f"Error {provider}:{model_id} - {direct_failure.message}", flush=True)
            return RunnerResult(
                status="error",
                error_kind=error_kind,
                error_message=direct_failure.message,
                transport_provider=decision.transport_provider,
                route_attempted=True,
                fallback_reason=fallback_reason,
            )

    _annotate_metrics(metrics, decision, fallback_reason=fallback_reason)
    metrics.setdefault("validation_policy", validation_policy(_transport_provider(decision)))
    # Provenance every sample carries, so a row can always be traced back to the
    # rules that produced it and to the group of attempts it came from. The
    # spike found 975 of 1,182 Together rows came from multi-attempt retries
    # with no way to tell from the row itself.
    metrics["sample_role"] = sample_role
    metrics["benchmark_profile_id"] = profile_id
    metrics["profile_hash"] = stable_hash(
        {
            "profile_id": profile_id,
            "query": QUERY_TEXT,
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
            "protocol_version": PROTOCOL_VERSION,
        }
    )
    metrics["effective_request_hash"] = stable_hash(
        {
            "source_provider": decision.source_provider,
            "source_model_id": decision.source_model_id,
            "transport_provider": decision.transport_provider,
            "transport_model_id": decision.transport_model_id,
            "route_provider_slug": decision.route_provider_slug,
            "query": QUERY_TEXT,
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
            "protocol_version": PROTOCOL_VERSION,
        }
    )
    metrics["requested_max_tokens"] = max_tokens
    metrics["protocol_version"] = PROTOCOL_VERSION
    metrics["attempt_group"] = str(job.get("_id"))
    metrics["attempt"] = int(job.get("attempt") or 1)
    try:
        log_success_mongo(model_config, metrics, sample_role=sample_role)
    except Exception as exc:
        reason = f"log failure: {type(exc).__name__}: {exc}"
        error_kind = log_error_mongo(
            provider=provider,
            model_id=model_id,
            stage="log",
            message=reason,
            exc_type=type(exc).__name__,
        )
        print(f"Error {provider}:{model_id} - {reason}", flush=True)
        return RunnerResult(status="error", error_kind=error_kind, error_message=reason)

    print(
        f"Success[{sample_role}] {provider}:{model_id} "
        f"(tps={metrics.get('tokens_per_second'):.2f}, out={metrics.get('output_tokens')})",
        flush=True,
    )
    return RunnerResult(
        status="success",
        sample_role=sample_role,
        transport_provider=decision.transport_provider,
        route_attempted=route_attempted,
        fallback_reason=fallback_reason,
    )


def _child_main(job: dict[str, Any], result_queue: Queue) -> None:
    try:
        if hasattr(os, "setsid"):
            os.setsid()
    except Exception:
        pass
    try:
        result_queue.put(run_benchmark_job(job))
    except BaseException as exc:
        result_queue.put(
            RunnerResult(
                status="error",
                error_kind="unknown",
                error_message=f"{type(exc).__name__}: {exc}",
            )
        )
        raise


def _kill_process_group(process) -> None:
    pid = process.pid
    if pid is None:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        try:
            process.terminate()
        except Exception:
            return
    process.join(timeout=5)
    if process.is_alive():
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:
            process.kill()


def run_job_in_child(job: dict[str, Any], *, deadline_seconds: float) -> RunnerResult:
    ctx = get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=_child_main, args=(job, result_queue))
    started = time.monotonic()
    timeout_decision = effective_job_route(job)
    process.start()
    remaining = max(0.0, float(deadline_seconds) - (time.monotonic() - started))
    process.join(timeout=remaining)
    if process.is_alive():
        _kill_process_group(process)
        process.join(timeout=5)
        provider = str(job.get("provider"))
        model_id = str(job.get("model_id"))
        message = f"benchmark timed out after {deadline_seconds}s"
        timeout_provenance = timeout_decision.metric_fields()
        timeout_profile_id = str(job.get("benchmark_profile_id") or DEFAULT_PROFILE_ID)
        if timeout_profile_id != DEFAULT_PROFILE_ID:
            timeout_provenance["profile_id"] = timeout_profile_id
        try:
            log_error_mongo(
                provider=provider,
                model_id=model_id,
                stage="timeout",
                message=message,
                exc_type="TimeoutError",
                provenance=timeout_provenance,
            )
        except Exception as exc:
            print(f"Failed to log timeout for {provider}:{model_id}: {exc}", flush=True)
        return RunnerResult(
            status="timeout",
            error_kind="timeout",
            error_message=message,
            route_attempted=timeout_decision.transport_provider == OPENROUTER_TRANSPORT,
            transport_provider=timeout_decision.transport_provider,
        )

    if process.exitcode and process.exitcode != 0:
        return RunnerResult(status="error", error_kind="unknown", error_message=f"child exited {process.exitcode}")

    if result_queue.empty():
        return RunnerResult(status="error", error_kind="unknown", error_message="child exited without result")

    result = result_queue.get()
    if isinstance(result, RunnerResult):
        return result
    if isinstance(result, dict):
        return RunnerResult(**result)
    return RunnerResult(status="error", error_kind="unknown", error_message=f"invalid child result: {result!r}")
