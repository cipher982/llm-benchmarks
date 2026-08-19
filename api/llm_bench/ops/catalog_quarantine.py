"""Dry-run-first quarantine for repeatedly unavailable catalog models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Any

import typer
from pymongo import MongoClient

from llm_bench.ops.reliability_cli import _hard_errors_since
from llm_bench.ops.reliability_cli import _last_success_ts


# Which providers to sweep, when the caller does not say. This was the literal
# pair ("together", "vertex"), written when those were the providers that
# misbehaved. The catalogue has since become 332 OpenRouter models out of 388,
# and OpenRouter was never in the list — so the tool reported "No quarantine
# candidates" while 120 enabled models sat permanently unservable, which is
# exactly the reassuring silence it exists to break.
#
# Derived from the catalogue instead, so it cannot go stale again the next time
# the provider mix changes.
def default_providers(db) -> list[str]:
    models_collection = os.getenv("MONGODB_COLLECTION_MODELS", "models")
    return sorted(
        str(provider)
        for provider in db[models_collection].distinct("provider", {"enabled": True, "deprecated": {"$ne": True}})
        if provider
    )


app = typer.Typer(help="Quarantine catalog models after repeated hard-model failures.")


@dataclass(frozen=True)
class QuarantineCandidate:
    provider: str
    model_id: str
    failures: int
    first_error: datetime
    last_error: datetime
    last_success: datetime | None


def should_quarantine(
    failures: list[dict[str, Any]],
    *,
    min_failures: int,
    min_span_minutes: int,
    health_error_kind: str | None = None,
    last_success: datetime | None = None,
) -> bool:
    if not failures:
        return False
    # A hard error with 400/404, 'not found', 'deprecated', or capability refusal is terminal:
    # the scheduler halts future retries, so it will never accumulate multiple timestamps.
    if health_error_kind in ("hard_model", "hard_capability"):
        latest = failures[-1]
        if latest.get("http_status") in (400, 404):
            return True
        msg = (latest.get("normalized_message") or latest.get("message") or "").lower()
        if (
            "not found" in msg
            or "no endpoints found" in msg
            or "deprecated" in msg
            or "does not exist" in msg
            or "does not support" in msg
            or "is not supported" in msg
            or "expected <" in msg
            or "invalid_request_error" in msg
        ):
            return True
    if len(failures) < min_failures:
        return False
    timestamps = [item.get("ts") for item in failures if isinstance(item.get("ts"), datetime)]
    if len(timestamps) < 2:
        if last_success is None and len(failures) >= min_failures:
            return True
        return False
    return (max(timestamps) - min(timestamps)) >= timedelta(minutes=min_span_minutes)


def find_candidates(
    db,
    *,
    providers: list[str],
    min_failures: int,
    min_span_minutes: int,
    lookback_days: int,
) -> list[QuarantineCandidate]:
    models_collection = os.getenv("MONGODB_COLLECTION_MODELS", "models")
    now = datetime.now(UTC)
    candidates: list[QuarantineCandidate] = []
    for model in db[models_collection].find(
        {
            "provider": {"$in": providers},
            "enabled": True,
            "deprecated": {"$ne": True},
        },
        {"provider": 1, "model_id": 1, "_id": 0},
    ):
        provider = model.get("provider")
        model_id = model.get("model_id")
        if not provider or not model_id:
            continue
        last_success = _last_success_ts(
            db=db,
            provider=provider,
            model=model_id,
            lookback_days=lookback_days,
        )
        since = last_success or now - timedelta(days=lookback_days)
        failures = _hard_errors_since(
            db=db,
            provider=provider,
            model=model_id,
            since=since,
            kind=("hard_model", "hard_capability"),
        )
        health_coll = os.getenv("MONGODB_COLLECTION_MODEL_HEALTH", "bench_model_health")
        health = db[health_coll].find_one(
            {"provider": provider, "model_id": model_id, "endpoint_tag": {"$in": [None, ""]}},
            {"last_error_kind": 1, "last_error_message": 1, "consecutive_failures": 1},
        )
        health_kind = health.get("last_error_kind") if isinstance(health, dict) else None
        if not failures and last_success is None and health_kind not in ("budget_exhausted", None):
            failures = _hard_errors_since(
                db=db,
                provider=provider,
                model=model_id,
                since=since,
                kind=("hard_model", "hard_capability", "unknown", "transient_provider"),
            )
        if not should_quarantine(
            failures,
            min_failures=min_failures,
            min_span_minutes=min_span_minutes,
            health_error_kind=health_kind,
            last_success=last_success,
        ):
            continue
        timestamps = [item["ts"] for item in failures if isinstance(item.get("ts"), datetime)]
        candidates.append(
            QuarantineCandidate(
                provider=provider,
                model_id=model_id,
                failures=len(failures),
                first_error=min(timestamps),
                last_error=max(timestamps),
                last_success=last_success,
            )
        )
    return sorted(candidates, key=lambda item: (item.provider, item.model_id))


@app.command()
def quarantine(
    provider: list[str] | None = typer.Option(None, "--provider", "-p"),
    min_failures: int = typer.Option(2, "--min-failures"),
    min_span_minutes: int = typer.Option(0, "--min-span-minutes"),
    lookback_days: int = typer.Option(30, "--lookback-days"),
    dry_run: bool = typer.Option(True, "--dry-run/--apply"),
) -> None:
    """Report or disable enabled Together/Vertex models with persistent 4xx failures.

    The default is dry-run. Use --apply only after reviewing the printed list.
    """
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "llm-bench")
    if not uri:
        raise typer.BadParameter("MONGODB_URI must be set")
    client = MongoClient(uri)
    try:
        db = client[db_name]
        providers = sorted(set(provider)) if provider else default_providers(db)
        candidates = find_candidates(
            db,
            providers=providers,
            min_failures=min_failures,
            min_span_minutes=min_span_minutes,
            lookback_days=lookback_days,
        )
        if not candidates:
            typer.echo("No quarantine candidates.")
            return
        models_collection = os.getenv("MONGODB_COLLECTION_MODELS", "models")
        action = "would disable" if dry_run else "disabling"
        for candidate in candidates:
            typer.echo(
                f"{action} {candidate.provider}/{candidate.model_id}: "
                f"{candidate.failures} hard_model failures "
                f"({candidate.first_error.isoformat()} to {candidate.last_error.isoformat()})"
            )
            if not dry_run:
                db[models_collection].update_one(
                    {
                        "provider": candidate.provider,
                        "model_id": candidate.model_id,
                        "enabled": True,
                    },
                    {
                        "$set": {
                            "enabled": False,
                            "disabled_class": "hard_model",
                            "disabled_reason": (
                                f"Provider hard failure ({candidate.failures} "
                                f"failure{'s' if candidate.failures > 1 else ''}); quarantined by catalog_quarantine"
                            ),
                            "disabled_at": datetime.now(UTC),
                            "disabled_by": "catalog_quarantine",
                            "catalog_quarantine": True,
                        }
                    },
                )
                health_coll = os.getenv("MONGODB_COLLECTION_MODEL_HEALTH", "bench_model_health")
                db[health_coll].update_many(
                    {"provider": candidate.provider, "model_id": candidate.model_id},
                    {"$set": {"enabled": False}},
                )
                endpoints_coll = os.getenv("MONGODB_COLLECTION_ENDPOINTS", "bench_endpoints")
                db[endpoints_coll].update_many(
                    {"model_id": candidate.model_id},
                    {
                        "$set": {
                            "enabled": False,
                            "disabled_reason": f"Parent model quarantined ({candidate.provider}/{candidate.model_id})",
                            "disabled_at": datetime.now(UTC),
                        }
                    },
                )
    finally:
        client.close()


if __name__ == "__main__":
    app()
