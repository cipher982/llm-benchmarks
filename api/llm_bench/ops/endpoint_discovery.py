"""Discover OpenRouter endpoints — the unit the site actually benchmarks.

An endpoint is one `(model, serving deployment)` pair, identified by the tag
OpenRouter publishes: `groq`, `deepinfra/fp8`, `google-vertex/us-east5`. It is
not a provider. `deepinfra/bf16` and `deepinfra/turbo` are different endpoints
of the same model from the same company, and they serve at different speeds.

Spec: docs/specs/endpoint-as-target.md

Two properties of this data drive most of the code below:

* **A base slug is not an endpoint.** OpenRouter matches every variant of a
  provider family when a bare slug appears in `provider.only`, so pinning
  `deepinfra` still load-balances `bf16` against `turbo`. Only the full tag
  identifies one deployment.
* **Quantization is part of identity.** `openai/gpt-oss-120b` is served at fp4
  by five endpoints and at bf16 by three. fp4 is markedly faster and materially
  worse, so those do not belong on one ranking axis.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime
from typing import Any

from pymongo.database import Database

from llm_bench.ops.openrouter_discovery import DEFAULT_BASE_URL
from llm_bench.ops.openrouter_discovery import is_router
from llm_bench.ops.openrouter_discovery import utcnow
from llm_bench.scheduler.mongo import collection_name
from llm_bench.scheduler.mongo import health_collection_name

# Endpoints the site reaches on its own credentials. Publishing them through
# OpenRouter as well would double-count the same deployment under two lanes.
DIRECT_PROVIDER_TAG_PREFIXES = (
    "openai",
    "amazon-bedrock",
    "google-vertex",
)

# OpenRouter marks degraded endpoints with a negative status. Admitting by mere
# presence schedules known-bad targets: four of gpt-oss-120b's twenty endpoints
# were at -2 when this was written.
MIN_ADMISSIBLE_STATUS = 0

# An endpoint must be absent from this many *complete* discovery passes before
# it is retired. A single failed or rate-limited pass is not evidence that a
# deployment disappeared — the same ratchet that decayed coverage to 11.7% in
# August (see CLAUDE.md 2026-08-04).
MISSING_PASSES_BEFORE_RETIREMENT = 3


def endpoints_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_ENDPOINTS", "bench_endpoints")


def endpoint_discovery_runs_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_ENDPOINT_DISCOVERY_RUNS", "bench_endpoint_discovery_runs")


def provider_canonical(tag: str) -> str:
    """The provider family a tag belongs to.

    This is the verification key, never the identity key. OpenRouter's response
    metadata reports a provider display name and never echoes the tag, so
    `deepinfra/fp8` and `deepinfra/turbo` are indistinguishable in a completed
    generation. Variant identity is asserted by our request; only the family can
    be confirmed from theirs.
    """

    return tag.split("/", 1)[0].strip().lower()


def is_direct_provider_tag(tag: str) -> bool:
    return provider_canonical(tag) in DIRECT_PROVIDER_TAG_PREFIXES


def quantization_of(endpoint: dict[str, Any]) -> str:
    """Normalised quantization, defaulting to explicit ignorance.

    `unknown` is a real and common value — Groq reports it — and it must never
    be silently grouped with a known one.
    """

    raw = endpoint.get("quantization")
    text = str(raw).strip().lower() if raw is not None else ""
    return text or "unknown"


def is_admissible(endpoint: dict[str, Any]) -> tuple[bool, str | None]:
    """Whether an endpoint can be scheduled, and why not when it cannot."""

    tag = str(endpoint.get("tag") or "").strip()
    if not tag:
        return False, "endpoint-has-no-tag"
    if is_direct_provider_tag(tag):
        return False, "served-by-a-direct-lane"
    try:
        status = int(endpoint.get("status", 0))
    except (TypeError, ValueError):
        return False, "endpoint-status-unreadable"
    if status < MIN_ADMISSIBLE_STATUS:
        return False, f"openrouter-endpoint-status-{status}"
    return True, None


def fetch_endpoints(
    model_id: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 60.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> list[dict[str, Any]]:
    """Fetch one model's endpoint list.

    Raises on transport failure so the caller can distinguish "this model has no
    endpoints" from "we could not read this model", which is exactly the
    distinction the retirement hysteresis depends on.
    """

    base_url = base_url or os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    url = f"{base_url.rstrip('/')}/models/{urllib.parse.quote(model_id)}/endpoints"
    headers = {"Accept": "application/json", "User-Agent": "llm-bench-endpoint-discovery/1"}
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if api_key:
        # Authenticated reads return throughput/latency percentiles and are not
        # rate-limited into the ground; an anonymous sweep lost 67 of 325 models.
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    with opener(request, timeout=timeout_seconds) as response:
        payload = json.load(response)
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        raise ValueError(f"OpenRouter endpoints payload for {model_id} has no data object")
    endpoints = data.get("endpoints")
    if not isinstance(endpoints, list):
        raise ValueError(f"OpenRouter endpoints payload for {model_id} has no endpoints list")
    return [e for e in endpoints if isinstance(e, dict)]


def endpoint_doc(model_id: str, endpoint: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    """One catalogue document for one endpoint."""

    tag = str(endpoint["tag"]).strip()
    pricing = endpoint.get("pricing") or {}
    throughput = endpoint.get("throughput_last_30m") or {}
    return {
        "model_id": model_id,
        "endpoint_tag": tag,
        "provider_canonical": provider_canonical(tag),
        "provider_name": endpoint.get("provider_name") or provider_canonical(tag),
        "quantization": quantization_of(endpoint),
        "context_length": endpoint.get("context_length"),
        "max_completion_tokens": endpoint.get("max_completion_tokens"),
        "supported_parameters": endpoint.get("supported_parameters") or [],
        "or_status": endpoint.get("status"),
        "or_uptime_1d": endpoint.get("uptime_last_1d"),
        # Endpoint-level completion price. Budget clamping that reads
        # model-level pricing can hand a premium endpoint a cheap endpoint's
        # token budget, multiplied across the whole fleet.
        "completion_price_per_token": _as_float(pricing.get("completion")),
        "prompt_price_per_token": _as_float(pricing.get("prompt")),
        # OpenRouter's own aggregate over real traffic. Not a controlled
        # benchmark, so it never publishes — but it is a free prior for
        # scheduling and an independent cross-check on our numbers.
        "or_throughput_p50": throughput.get("p50"),
        "last_seen_at": now,
    }


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def endpoint_update(
    doc: dict[str, Any], existing: dict[str, Any] | None, *, now: datetime
) -> tuple[dict[str, Any], bool]:
    """The upsert for one discovered endpoint, and whether it restores a retired one.

    `enabled` must appear in exactly one operator. A restore put it in `$set`
    while `$setOnInsert` still carried it for the fresh-insert case, and Mongo
    rejects an update that names one path under two operators — even on a
    document that exists, where `$setOnInsert` would do nothing. From
    2026-08-31 every discovery pass threw on the first restorable endpoint,
    so no endpoint was admitted or restored for two days, new models had
    nothing to schedule, and the reaper re-retired the same 27 endpoints
    each tick that discovery could no longer bring back.
    """
    update: dict[str, Any] = {
        "$set": {**doc, "missing_passes": 0},
        "$setOnInsert": {"first_seen_at": now},
    }
    restored = bool(existing and existing.get("disabled_by") == "endpoint-discovery")
    if restored:
        update["$set"]["enabled"] = True
        update["$unset"] = {"disabled_reason": "", "disabled_at": "", "disabled_by": ""}
    else:
        update["$setOnInsert"]["enabled"] = True
    return update, restored


def refresh_endpoints(
    db: Database,
    *,
    model_ids: list[str],
    now: datetime | None = None,
    fetcher: Callable[[str], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Reconcile the endpoint catalogue for the given models.

    A model whose fetch fails is left entirely alone: its endpoints keep their
    `missing_passes` count and their enabled state. Only a model that was read
    successfully can contribute evidence that one of its endpoints is gone.
    """

    now = now or utcnow()
    fetcher = fetcher or fetch_endpoints
    endpoints = db[endpoints_collection_name()]
    started_at = now

    seen_by_model: dict[str, set[str]] = {}
    admitted = 0
    rejected: dict[str, int] = {}
    failed_models: list[str] = []

    for model_id in model_ids:
        if is_router(model_id):
            continue
        try:
            rows = fetcher(model_id)
        except Exception as exc:  # noqa: BLE001 - one bad model must not end the pass
            failed_models.append(f"{model_id}: {type(exc).__name__}")
            continue

        seen: set[str] = set()
        for endpoint in rows:
            ok, reason = is_admissible(endpoint)
            if not ok:
                rejected[reason or "unknown"] = rejected.get(reason or "unknown", 0) + 1
                continue
            doc = endpoint_doc(model_id, endpoint, now=now)
            seen.add(doc["endpoint_tag"])
            query = {"model_id": model_id, "endpoint_tag": doc["endpoint_tag"]}
            existing = endpoints.find_one(query, {"enabled": 1, "disabled_by": 1})
            update, restored = endpoint_update(doc, existing, now=now)
            endpoints.update_one(query, update, upsert=True)
            if restored:
                db[health_collection_name()].update_one(
                    {"provider": "openrouter", "model_id": model_id, "endpoint_tag": doc["endpoint_tag"]},
                    {"$set": {"enabled": True}},
                )
            admitted += 1
        seen_by_model[model_id] = seen

    # Retirement, only for models we actually read this pass.
    retired = 0
    for model_id, seen in seen_by_model.items():
        stale = endpoints.find(
            {"model_id": model_id, "enabled": True, "endpoint_tag": {"$nin": sorted(seen)}},
            {"endpoint_tag": 1, "missing_passes": 1},
        )
        for row in stale:
            misses = int(row.get("missing_passes") or 0) + 1
            update: dict[str, Any] = {"missing_passes": misses}
            if misses >= MISSING_PASSES_BEFORE_RETIREMENT:
                update.update(
                    {
                        "enabled": False,
                        "disabled_reason": (f"absent from {misses} consecutive complete OpenRouter endpoint listings"),
                        "disabled_at": now,
                        "disabled_by": "endpoint-discovery",
                    }
                )
                retired += 1
            endpoints.update_one({"_id": row["_id"]}, {"$set": update})
            if misses >= MISSING_PASSES_BEFORE_RETIREMENT:
                db[health_collection_name()].update_one(
                    {"provider": "openrouter", "model_id": model_id, "endpoint_tag": row["endpoint_tag"]},
                    {"$set": {"enabled": False}},
                )

    record = {
        "started_at": started_at,
        "finished_at": utcnow(),
        "status": "ok" if not failed_models else "partial",
        "models_requested": len(model_ids),
        "models_read": len(seen_by_model),
        "models_failed": failed_models[:50],
        "endpoints_admitted": admitted,
        "endpoints_rejected": rejected,
        "endpoints_retired": retired,
    }
    db[endpoint_discovery_runs_collection_name()].insert_one(dict(record))
    return record
