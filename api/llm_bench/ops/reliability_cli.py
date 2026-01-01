from __future__ import annotations

import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import typer
from pymongo import MongoClient


UTC = timezone.utc
app = typer.Typer(help="Reliability operations (recommendations-only by default).")


def _mongo() -> Tuple[str, str]:
    uri = os.getenv("MONGODB_URI")
    db = os.getenv("MONGODB_DB", "llm-bench")
    if not uri:
        raise RuntimeError("MONGODB_URI must be set")
    return uri, db


def _coll(name_env: str, default: str) -> str:
    return os.getenv(name_env, default)


def _ensure_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _ts_field(doc: Dict[str, Any]) -> Optional[datetime]:
    value = doc.get("gen_ts") or doc.get("run_ts") or doc.get("_ts")
    if isinstance(value, datetime):
        return _ensure_utc(value)
    return None


def _last_success_ts(
    *,
    db,
    provider: str,
    model: str,
    lookback_days: int,
) -> Optional[datetime]:
    metrics_coll = _coll("MONGODB_COLLECTION_CLOUD", "metrics_cloud_v2")
    since = datetime.now(UTC) - timedelta(days=lookback_days)
    doc = db[metrics_coll].find_one(
        {"provider": provider, "model_name": model, "$or": [{"gen_ts": {"$gte": since}}, {"run_ts": {"$gte": since}}]},
        {"gen_ts": 1, "run_ts": 1},
        sort=[("gen_ts", -1), ("run_ts", -1)],
    )
    if not isinstance(doc, dict):
        return None
    return _ts_field(doc)


def _hard_errors_since(
    *,
    db,
    provider: str,
    model: str,
    since: Optional[datetime],
    kind: str,
    limit: int = 5000,
) -> List[Dict[str, Any]]:
    errors_coll = _coll("MONGODB_COLLECTION_ERRORS", "errors_cloud")
    query: Dict[str, Any] = {"provider": provider, "model_name": model, "error_kind": kind}
    if since is not None:
        query["ts"] = {"$gte": since}
    cur = db[errors_coll].find(query, {"ts": 1, "normalized_message": 1, "message": 1, "http_status": 1}).sort("ts", 1).limit(limit)
    return list(cur)


def _unique_messages(docs: Sequence[Dict[str, Any]], *, limit: int = 3) -> List[str]:
    out: List[str] = []
    for doc in reversed(list(docs)):
        msg = (doc.get("normalized_message") or doc.get("message") or "").strip()
        if not msg:
            continue
        if msg in out:
            continue
        out.append(msg)
        if len(out) >= limit:
            break
    return out


@dataclass(frozen=True, slots=True)
class Recommendation:
    provider: str
    model_id: str
    action: str
    confidence: str
    reasons: List[str]
    evidence: Dict[str, Any]


@app.command()
def recommend(
    provider: List[str] = typer.Option(None, "--provider", "-p", help="Filter to provider(s)."),
    include_disabled: bool = typer.Option(False, help="Include disabled models in analysis."),
    lookback_success_days: int = typer.Option(90, help="Look back this many days for a success (for high confidence)."),
    min_hard_failures: int = typer.Option(5, help="Minimum hard_model failures needed to recommend disable."),
    min_span_minutes: int = typer.Option(120, help="Require failures to span at least this many minutes."),
    json_output: bool = typer.Option(False, "--json", help="Print JSON instead of a table."),
) -> None:
    """Generate recommendations based on error_kind + success history.

    This command is recommendations-only. It does not mutate the `models` catalog.
    """
    uri, db_name = _mongo()
    models_coll = _coll("MONGODB_COLLECTION_MODELS", "models")

    provider_filter = sorted(set(provider or [])) if provider else None

    client = MongoClient(uri)
    try:
        db = client[db_name]

        model_query: Dict[str, Any] = {"deprecated": {"$ne": True}}
        if not include_disabled:
            model_query["enabled"] = True
        if provider_filter:
            model_query["provider"] = {"$in": provider_filter}

        recs: List[Recommendation] = []
        now = datetime.now(UTC)

        for model_doc in db[models_coll].find(model_query, {"provider": 1, "model_id": 1, "enabled": 1, "_id": 0}):
            prov = model_doc.get("provider")
            mid = model_doc.get("model_id")
            if not prov or not mid:
                continue

            last_success = _last_success_ts(db=db, provider=prov, model=mid, lookback_days=lookback_success_days)
            hard_model_errors = _hard_errors_since(db=db, provider=prov, model=mid, since=last_success, kind="hard_model")
            hard_cap_errors = _hard_errors_since(db=db, provider=prov, model=mid, since=last_success, kind="hard_capability")

            if hard_cap_errors:
                sample = _unique_messages(hard_cap_errors, limit=3)
                recs.append(
                    Recommendation(
                        provider=prov,
                        model_id=mid,
                        action="investigate_capability",
                        confidence="medium",
                        reasons=["Hard capability errors detected (likely endpoint/capability mismatch; do not disable automatically)."],
                        evidence={
                            "last_success": last_success.isoformat() if last_success else None,
                            "hard_capability_failures": len(hard_cap_errors),
                            "sample_messages": sample,
                        },
                    )
                )

            if len(hard_model_errors) < min_hard_failures:
                continue

            first_ts = _ensure_utc(hard_model_errors[0].get("ts"))
            last_ts = _ensure_utc(hard_model_errors[-1].get("ts"))
            span_ok = False
            span_minutes = None
            if first_ts and last_ts:
                span_minutes = int((last_ts - first_ts).total_seconds() / 60)
                span_ok = span_minutes >= min_span_minutes

            if not span_ok:
                continue

            sample = _unique_messages(hard_model_errors, limit=3)
            confidence = "high" if last_success else "low"
            reasons = [
                f"{len(hard_model_errors)} hard_model failures since last success.",
                f"Failures span ~{span_minutes} minutes.",
            ]
            if not last_success:
                reasons.append("No recent success in lookback window; treat as manual review (could be bad ID or configuration).")

            recs.append(
                Recommendation(
                    provider=prov,
                    model_id=mid,
                    action="recommend_disable",
                    confidence=confidence,
                    reasons=reasons,
                    evidence={
                        "last_success": last_success.isoformat() if last_success else None,
                        "hard_model_failures": len(hard_model_errors),
                        "first_error": first_ts.isoformat() if first_ts else None,
                        "last_error": last_ts.isoformat() if last_ts else None,
                        "sample_messages": sample,
                    },
                )
            )

        if json_output:
            typer.echo(json.dumps([asdict(r) for r in recs], indent=2, default=str))
            return

        if not recs:
            typer.echo("No recommendations.")
            return

        # Minimal table output
        typer.echo("ACTION                 CONF   PROVIDER     MODEL")
        typer.echo("---------------------  -----  ----------   ------------------------------")
        for r in recs:
            typer.echo(f"{r.action:21s}  {r.confidence:5s}  {r.provider:10s}   {r.model_id}")
    finally:
        client.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
