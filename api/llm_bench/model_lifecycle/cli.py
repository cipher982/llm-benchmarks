from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import typer
from pymongo import MongoClient
import dotenv

from .classifier import LifecycleDecision, LifecycleStatus, classify_snapshot
from .collector import LifecycleSnapshot, collect_lifecycle_snapshots


app = typer.Typer(help="Model lifecycle monitoring utilities")

dotenv.load_dotenv()


STATUS_WEIGHTS: Dict[LifecycleStatus, int] = {
    LifecycleStatus.LIKELY_DEPRECATED: 90,
    LifecycleStatus.FAILING: 75,
    LifecycleStatus.STALE: 60,
    LifecycleStatus.NEVER_SUCCEEDED: 55,
    LifecycleStatus.MONITOR: 40,
    LifecycleStatus.ACTIVE: 10,
    LifecycleStatus.DISABLED: 5,
    LifecycleStatus.DEPRECATED: 0,
}


def _risk_weight(status: LifecycleStatus) -> int:
    return STATUS_WEIGHTS.get(status, 0)


def _status_bucket(decision: LifecycleDecision) -> str:
    return decision.status.value


def _format_dt(value: Optional[datetime]) -> str:
    if value is None:
        return "—"
    return value.astimezone(timezone.utc).strftime("%Y-%m-%d")


def _serialize_decision(snapshot: LifecycleSnapshot, decision: LifecycleDecision, computed_at: datetime) -> Dict:
    successes = snapshot.successes
    errors = snapshot.errors
    metadata = snapshot.metadata

    return {
        "provider": snapshot.provider,
        "model_id": snapshot.model_id,
        "display_name": snapshot.display_name,
        "status": decision.status.value,
        "confidence": decision.confidence,
        "reasons": decision.reasons,
        "recommended_actions": decision.recommended_actions,
        "computed_at": computed_at.astimezone(timezone.utc),
        "catalog_state": snapshot.catalog_state.value,
        "metrics": {
            "last_success": successes.last_success,
            "successes_7d": successes.successes_7d,
            "successes_30d": successes.successes_30d,
            "successes_120d": successes.successes_120d,
            "last_error": errors.last_error,
            "errors_7d": errors.errors_7d,
            "errors_30d": errors.errors_30d,
            "hard_failures_7d": errors.hard_failures_7d,
            "hard_failures_30d": errors.hard_failures_30d,
            "recent_messages": [
                {
                    "timestamp": message.timestamp,
                    "message": message.message,
                    "kind": message.kind,
                }
                for message in errors.recent_messages
            ],
        },
        "metadata": {
            "display_name": metadata.display_name,
            "enabled": metadata.enabled,
            "deprecated": metadata.deprecated,
            "deprecation_date": metadata.deprecation_date,
            "successor_model": metadata.successor_model,
            "deprecation_reason": metadata.deprecation_reason,
        },
    }


def _write_results(docs: List[Dict], collection_name: str) -> None:
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "llm-bench")
    if not uri:
        raise RuntimeError("MONGODB_URI must be set to persist lifecycle results")

    client = MongoClient(uri)
    try:
        coll = client[db_name][collection_name]
        now = datetime.now(timezone.utc)
        for doc in docs:
            update_doc = dict(doc)
            update_doc.setdefault("computed_at", now)
            coll.update_one(
                {"provider": doc["provider"], "model_id": doc["model_id"]},
                {
                    "$set": update_doc,
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )
    finally:
        client.close()


@app.command()
def report(
    provider: List[str] = typer.Option(None, "--provider", "-p", help="Filter to provider(s)."),
    include_active: bool = typer.Option(False, help="Include active models in the report."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write report as JSON to path."),
    json_output: bool = typer.Option(False, "--json", help="Print JSON to stdout instead of table."),
    apply: bool = typer.Option(False, "--apply", help="Persist results into MongoDB collection."),
    collection: str = typer.Option("model_status", help="Collection name to use with --apply."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation when using --apply."),
) -> None:
    """Generate lifecycle status report."""

    now = datetime.now(timezone.utc)

    snapshots = collect_lifecycle_snapshots(provider_filter=provider or None, now=now)
    results: List[Dict] = []

    for snapshot in snapshots:
        decision = classify_snapshot(snapshot, now=now)
        if not include_active and decision.status == LifecycleStatus.ACTIVE:
            continue
        results.append(_serialize_decision(snapshot, decision, now))

    results.sort(key=lambda doc: (_risk_weight(LifecycleStatus(doc["status"])), doc["provider"], doc["model_id"]), reverse=True)

    if output or json_output:
        payload = json.dumps(results, default=str, indent=2)
        if output:
            output.write_text(payload)
        else:
            typer.echo(payload)
    else:
        _print_table(results)

    if apply:
        if not yes:
            typer.echo("--apply requested; skipping because --yes was not provided.")
        else:
            _write_results(results, collection)
            typer.echo(f"Persisted {len(results)} lifecycle entries into collection '{collection}'.")


@app.command()
def summary(
    provider: List[str] = typer.Option(None, "--provider", "-p", help="Filter to provider(s)."),
    include_active: bool = typer.Option(False, help="Include active models in the summary."),
    json_output: bool = typer.Option(False, "--json", help="Print JSON instead of a table."),
) -> None:
    """Show aggregate lifecycle signal counts per provider."""

    now = datetime.now(timezone.utc)
    snapshots = collect_lifecycle_snapshots(provider_filter=provider or None, now=now)

    totals: Dict[str, Dict[str, int]] = {}
    sample_reasons: Dict[str, Dict[str, str]] = {}

    for snapshot in snapshots:
        decision = classify_snapshot(snapshot, now=now)
        if not include_active and decision.status == LifecycleStatus.ACTIVE:
            continue

        provider_key = snapshot.provider
        bucket = _status_bucket(decision)

        provider_totals = totals.setdefault(provider_key, {})
        provider_totals[bucket] = provider_totals.get(bucket, 0) + 1

        if decision.reasons:
            provider_reasons = sample_reasons.setdefault(provider_key, {})
            provider_reasons.setdefault(bucket, decision.reasons[0])

    rows: List[Dict[str, object]] = []
    for provider_name, counts in sorted(totals.items()):
        total = sum(counts.values())
        rows.append(
            {
                "provider": provider_name,
                "total": total,
                "counts": counts,
                "sample_reason": sample_reasons.get(provider_name, {}),
            }
        )

    if json_output:
        typer.echo(json.dumps(rows, indent=2))
    else:
        _print_summary_table(rows)


def _print_summary_table(rows: List[Dict[str, object]]) -> None:
    if not rows:
        typer.echo("No providers matched the current filters.")
        return

    headers = ["PROVIDER", "TOTAL", "LIKELY_DEPRECATED", "FAILING", "STALE", "NEVER_SUCCEEDED", "MONITOR", "DISABLED"]
    col_widths = [len(h) for h in headers]

    def get_count(row: Dict[str, object], key: str) -> int:
        counts = row.get("counts", {})
        if isinstance(counts, dict):
            return int(counts.get(key, 0))
        return 0

    table: List[List[str]] = []
    for row in rows:
        line = [
            str(row["provider"]),
            str(row["total"]),
            str(get_count(row, LifecycleStatus.LIKELY_DEPRECATED.value)),
            str(get_count(row, LifecycleStatus.FAILING.value)),
            str(get_count(row, LifecycleStatus.STALE.value)),
            str(get_count(row, LifecycleStatus.NEVER_SUCCEEDED.value)),
            str(get_count(row, LifecycleStatus.MONITOR.value)),
            str(get_count(row, LifecycleStatus.DISABLED.value)),
        ]
        table.append(line)
        for idx, value in enumerate(line):
            col_widths[idx] = max(col_widths[idx], len(value))

    header_row = "  ".join(headers[idx].ljust(col_widths[idx]) for idx in range(len(headers)))
    typer.echo(header_row)
    typer.echo("  ".join("-" * width for width in col_widths))

    for line in table:
        typer.echo("  ".join(value.ljust(col_widths[idx]) for idx, value in enumerate(line)))


def _print_table(rows: List[Dict]) -> None:
    if not rows:
        typer.echo("No models matched the current filters.")
        return

    headers = [
        "STATUS",
        "CONF",
        "PROVIDER",
        "MODEL",
        "DISPLAY",
        "LAST OK",
        "LAST ERR",
        "OK30",
        "ERR7",
        "HARD7",
        "TOP REASON",
    ]
    col_widths = [len(h) for h in headers]

    table: List[List[str]] = []
    for doc in rows:
        metrics = doc["metrics"]
        line = [
            doc["status"],
            doc["confidence"],
            doc["provider"],
            doc["model_id"],
            doc["display_name"],
            _format_dt(metrics.get("last_success")),
            _format_dt(metrics.get("last_error")),
            str(metrics.get("successes_30d", 0)),
            str(metrics.get("errors_7d", 0)),
            str(metrics.get("hard_failures_7d", 0)),
            doc["reasons"][0] if doc["reasons"] else "",
        ]
        table.append(line)
        for idx, value in enumerate(line):
            col_widths[idx] = max(col_widths[idx], len(value))

    header_row = "  ".join(h.ljust(col_widths[idx]) for idx, h in enumerate(headers))
    typer.echo(header_row)
    typer.echo("  ".join("-" * width for width in col_widths))

    for line in table:
        typer.echo("  ".join(value.ljust(col_widths[idx]) for idx, value in enumerate(line)))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
