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

DEFAULT_PROVIDERS = ("together", "vertex")
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
) -> bool:
    if len(failures) < min_failures:
        return False
    timestamps = [item.get("ts") for item in failures if isinstance(item.get("ts"), datetime)]
    if len(timestamps) < 2:
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
            kind="hard_model",
        )
        if not should_quarantine(
            failures,
            min_failures=min_failures,
            min_span_minutes=min_span_minutes,
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
    providers = sorted(set(provider or DEFAULT_PROVIDERS))

    client = MongoClient(uri)
    try:
        db = client[db_name]
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
                            "disabled_reason": "Repeated hard_model failures; quarantined by catalog_quarantine",
                            "disabled_at": datetime.now(UTC),
                            "catalog_quarantine": True,
                        }
                    },
                )
    finally:
        client.close()


if __name__ == "__main__":
    app()
