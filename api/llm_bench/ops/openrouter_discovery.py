"""Additive discovery for OpenRouter-native benchmark models.

OpenRouter is both a transport for existing source-provider rows and a
catalogue of models that do not exist in any direct-provider lane.  The latter
must enter the same provider_catalog -> admission -> models pipeline as every
other provider; merely recording them in the legacy ``openrouter_catalog``
collection leaves them permanently disabled.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime
from datetime import timezone
from typing import Any
from typing import Callable

from pymongo.database import Database

from llm_bench.scheduler.mongo import collection_name

PROVIDER = "openrouter"
SOURCE_VERSION = 1
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def discovery_runs_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_DISCOVERY_RUNS", "bench_discovery_runs")


def base_model_id(model_id: str) -> str:
    """Collapse OpenRouter routing variants into one benchmark identity."""

    return model_id.split(":", 1)[0]


def is_router(model_id: str) -> bool:
    """OpenRouter's own routers (`openrouter/auto`, `openrouter/free`, ...).

    These are not models. Each call lands on whatever upstream the router picks
    that second — one `openrouter/auto-beta` sample came back from Google, the
    next from xAI — so a throughput number for one is a number for nothing.
    """

    return model_id.split("/", 1)[0].lower() == "openrouter"


def canonical_models(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one row per model, preferring the unsuffixed catalogue row."""

    by_base: dict[str, dict[str, Any]] = {}
    for row in rows:
        model_id = str(row.get("id") or "").strip()
        if not model_id or "/" not in model_id or is_router(model_id):
            continue
        base = base_model_id(model_id)
        if base not in by_base or model_id == base:
            by_base[base] = dict(row, id=base)
    return [by_base[key] for key in sorted(by_base)]


def fetch_models(
    *,
    base_url: str | None = None,
    timeout_seconds: float = 60.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> tuple[list[dict[str, Any]], int | None]:
    """Fetch the current text catalogue and return rows plus its total count."""

    base_url = base_url or os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)
    url = f"{base_url.rstrip('/')}/models?{urllib.parse.urlencode({'output_modalities': 'text'})}"
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "llm-bench-openrouter-discovery/1"},
        method="GET",
    )
    with opener(request, timeout=timeout_seconds) as response:
        payload = json.load(response)
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("OpenRouter model catalogue has no data list")
    return [row for row in rows if isinstance(row, dict) and row.get("id")], payload.get("total_count")


def _display_name(row: dict[str, Any], model_id: str) -> str:
    """OpenRouter's human label, not its machine slug.

    This preferred `canonical_slug` and called it a display name. The slug is
    the dated identifier — `z-ai/glm-4.7-20251222` — while `name` is the label
    OpenRouter itself renders, `Z.ai: GLM 4.7`. Admission copies whatever comes
    back here onto `models.display_name`, so the leaderboard showed 195 of 236
    rows as raw ids while the readable name sat unread in the same document.

    The vendor prefix is stripped here because the site attributes the provider
    separately; `model_naming` owns the same parse for the rows already written.
    """
    from llm_bench.ops.model_naming import parse_catalogue_name

    _, label = parse_catalogue_name(row.get("name"))
    return str(label or row.get("canonical_slug") or model_id)


def _catalog_doc(row: dict[str, Any], *, now: datetime) -> dict[str, Any]:
    model_id = str(row["id"])
    architecture = row.get("architecture") or {}
    return {
        "openrouter_id": model_id,
        "base_model_id": base_model_id(model_id),
        "name": row.get("name") or model_id,
        "canonical_slug": row.get("canonical_slug"),
        "context_length": row.get("context_length"),
        "pricing": row.get("pricing") or {},
        "architecture": architecture,
        "supported_parameters": row.get("supported_parameters") or [],
        "input_modalities": architecture.get("input_modalities") or [],
        "output_modalities": architecture.get("output_modalities") or ["text"],
        "last_seen_at": now,
    }


def refresh_catalog(
    db: Database,
    *,
    now: datetime | None = None,
    fetcher: Callable[[], tuple[list[dict[str, Any]], int | None]] | None = None,
) -> dict[str, Any]:
    """Refresh OpenRouter and mirror current models into ``provider_catalog``.

    The run ledger is written on both success and failure.  A failed or partial
    read never becomes evidence that an existing model disappeared.
    """

    now = now or utcnow()
    fetcher = fetcher or fetch_models
    runs = db[discovery_runs_collection_name()]
    started_at = now
    try:
        raw_rows, total_count = fetcher()
        models = canonical_models(raw_rows)
        complete = total_count is None or len(raw_rows) >= int(total_count)
    except Exception as exc:  # noqa: BLE001 - discovery must leave an audit row
        runs.insert_one(
            {
                "provider": PROVIDER,
                "started_at": started_at,
                "finished_at": utcnow(),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "raw_count": 0,
                "accepted_count": 0,
                "new_count": 0,
                "pagination_complete": False,
                "source_version": SOURCE_VERSION,
            }
        )
        return {"status": "failed", "error": str(exc)}

    provider_catalog = db.provider_catalog
    known_catalog = {
        row["model_id"] for row in provider_catalog.find({"provider": PROVIDER}, {"model_id": 1, "_id": 0})
    }
    model_ids = {str(row["id"]) for row in models}
    known_models = {row["model_id"] for row in db.models.find({"provider": PROVIDER}, {"model_id": 1, "_id": 0})}
    provider_rows = []
    legacy_catalog = db.openrouter_catalog
    for row in models:
        model_id = str(row["id"])
        provider_rows.append(
            {
                "provider": PROVIDER,
                "model_id": model_id,
                "name": _display_name(row, model_id),
                "openrouter_id": model_id,
                "source": "openrouter_catalog",
                "source_version": SOURCE_VERSION,
                "last_seen_at": now,
                "openrouter": _catalog_doc(row, now=now),
            }
        )
        legacy_catalog.update_one(
            {"openrouter_id": model_id},
            {
                "$set": _catalog_doc(row, now=now),
                "$setOnInsert": {"first_seen_at": now},
            },
            upsert=True,
        )

    for row in provider_rows:
        provider_catalog.update_one(
            {"provider": PROVIDER, "model_id": row["model_id"]},
            {
                "$set": row,
                "$setOnInsert": {"first_seen_at": now},
            },
            upsert=True,
        )

    finished_at = utcnow()
    runs.insert_one(
        {
            "provider": PROVIDER,
            "started_at": started_at,
            "finished_at": finished_at,
            "status": "completed" if complete else "failed",
            "error": None if complete else "catalogue total exceeds response rows",
            "raw_count": len(raw_rows),
            "accepted_count": len(models),
            "new_count": len(model_ids - known_catalog),
            "pagination_complete": complete,
            "catalog_total_count": total_count,
            "models_missing_from_models_collection": len(model_ids - known_models),
            "source_version": SOURCE_VERSION,
        }
    )
    return {
        "status": "completed" if complete else "failed",
        "raw_count": len(raw_rows),
        "accepted_count": len(models),
        "new_catalog_models": len(model_ids - known_catalog),
        "new_benchmark_models": len(model_ids - known_models),
        "pagination_complete": complete,
    }
