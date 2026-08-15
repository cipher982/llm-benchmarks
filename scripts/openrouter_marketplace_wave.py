#!/usr/bin/env python3
"""Build, canary, and promote an OpenRouter or-served migration wave.

The source catalogue remains keyed by its original provider/model identity. The
transport changes to OpenRouter only after a routed-only canary observes usable
text and verified, stable provider metadata. This command is intentionally
bounded and idempotent; it never enables or disables model rows.

Examples::

    uv run python scripts/openrouter_marketplace_wave.py build \
      --alias-file docs/specs/openrouter-model-aliases.v2.json \
      --alias-file /private/tmp/openrouter-v4/v5_eyeball_verdicts.tsv \
      --output /tmp/openrouter-wave/decisions.json

    uv run python scripts/openrouter_marketplace_wave.py canary \
      --decisions /tmp/openrouter-wave/decisions.json \
      --output-dir /tmp/openrouter-wave/canaries

    uv run python scripts/openrouter_marketplace_wave.py promote \
      --decisions /tmp/openrouter-wave/decisions.json \
      --canary-dir /tmp/openrouter-wave/canaries \
      --output-dir /tmp/openrouter-wave/routes \
      --evidence-base s3://artifacts/llm-benchmarks/openrouter-consolidation/2026-08-15/marketplace-canaries
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime
from datetime import timezone
from math import ceil
from pathlib import Path
from typing import Any

from llm_bench.scheduler.mongo import models_collection_name
from llm_bench.scheduler.mongo import route_decisions_collection_name
from llm_bench.scheduler.mongo import route_revocation_generation
from llm_bench.scheduler.routing import DIRECT_PROVIDERS
from llm_bench.scheduler.routing import OR_SERVED_POLICY
from llm_bench.scheduler.routing import ROUTE_DECISION_VERSION
from pymongo import MongoClient

from scripts.openrouter_budget import DEFAULT_BATCH_MAX_USD
from scripts.openrouter_budget import DEFAULT_DAILY_MAX_USD
from scripts.openrouter_budget import reserve_daily_budget
from scripts.openrouter_or_served_canary import run_routed_canary
from scripts.openrouter_promote_route import promote_or_served_route

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_ATTEMPTS = 12
DEFAULT_MAX_TOKENS = 512
DEFAULT_DEADLINE_SECONDS = 300
DEFAULT_MIN_SUCCESS_RATE = 0.95


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(provider: str, model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{provider}__{model_id}")


def provider_key(value: Any) -> str:
    return " ".join(str(value).casefold().replace("_", " ").replace("-", " ").split())


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def request_json(url: str, *, api_key: str | None = None, timeout: float = 60.0) -> Any:
    headers = {"Accept": "application/json", "User-Agent": "llm-bench-marketplace-wave/1"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - configured API URL
        return json.load(response)


def fetch_catalog(base_url: str, *, api_key: str | None) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/models?output_modalities=text"
    payload = request_json(url, api_key=api_key)
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("OpenRouter model catalog has no data list")
    return {
        "url": url,
        "fetched_at": utc_now(),
        "total_count": payload.get("total_count"),
        "data": [row for row in rows if isinstance(row, dict) and row.get("id")],
    }


def fetch_providers(base_url: str, *, api_key: str | None) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/providers"
    payload = request_json(url, api_key=api_key)
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("OpenRouter provider catalog has no data list")
    return {
        "url": url,
        "fetched_at": utc_now(),
        "data": [row for row in rows if isinstance(row, dict) and row.get("slug")],
    }


def fetch_endpoints(base_url: str, model_id: str, *, api_key: str | None) -> dict[str, Any]:
    author, slug = model_id.split("/", 1)
    url = (
        f"{base_url.rstrip('/')}/models/{urllib.parse.quote(author, safe='')}/"
        f"{urllib.parse.quote(slug, safe='')}/endpoints"
    )
    payload = request_json(url, api_key=api_key)
    data = payload.get("data") if isinstance(payload, dict) else None
    endpoints = data.get("endpoints") if isinstance(data, dict) else None
    if not isinstance(endpoints, list):
        raise ValueError(f"endpoint listing for {model_id} has no endpoints list")
    return {"url": url, "fetched_at": utc_now(), "data": data}


def load_aliases(paths: list[Path]) -> dict[str, str]:
    """Load reviewed JSON aliases and the v5 TSV review format."""

    aliases: dict[str, str] = {}
    for path in paths:
        if path.suffix.lower() == ".tsv":
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    if row.get("verdict") in {"route", "route-close"} and row.get("target_or_model_id"):
                        aliases[f"{row['source_provider']}/{row['source_model_id']}"] = row["target_or_model_id"]
            continue
        payload = load_json(path)
        rows = payload.get("aliases") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"alias file {path} has no aliases list")
        for row in rows:
            if not isinstance(row, dict) or not row.get("source_key") or not row.get("target_or_model_id"):
                continue
            aliases[str(row["source_key"])] = str(row["target_or_model_id"])
    return aliases


def _latest_active_routes(db) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    cursor = (
        db[route_decisions_collection_name()]
        .find({"state": "active"})
        .sort([("route_snapshot_at", -1), ("updated_at", -1)])
    )
    for row in cursor:
        key = (str(row.get("source_provider") or ""), str(row.get("source_model_id") or ""))
        if all(key) and key not in latest:
            latest[key] = row
    return latest


def _first_provider(endpoint_payload: dict[str, Any]) -> tuple[str | None, str | None]:
    data = endpoint_payload.get("data") or {}
    endpoints = data.get("endpoints") or []
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            continue
        name = endpoint.get("provider_name") or endpoint.get("provider")
        if name:
            return str(name), provider_key(name)
    return None, None


def _pricing(catalog_by_id: dict[str, dict[str, Any]], model_id: str) -> dict[str, Any]:
    pricing = catalog_by_id.get(model_id, {}).get("pricing") or {}
    return {
        "input_per_token": float(pricing.get("prompt", 0) or 0),
        "output_per_token": float(pricing.get("completion", 0) or 0),
    }


def build_report(
    *,
    db,
    aliases: dict[str, str],
    catalog: dict[str, Any],
    providers: dict[str, Any],
    endpoint_fetcher,
    source_providers: set[str] | None = None,
) -> dict[str, Any]:
    catalog_rows = catalog["data"]
    catalog_by_id = {str(row["id"]): row for row in catalog_rows}
    active_routes = _latest_active_routes(db)
    enabled = db[models_collection_name()].find(
        {"enabled": True, "deprecated": {"$ne": True}},
        {"provider": 1, "model_id": 1, "_id": 0},
    )
    decisions: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    snapshot_at = utc_now()
    for row in enabled:
        source_provider = str(row.get("provider") or "")
        source_model_id = str(row.get("model_id") or "")
        if not source_provider or not source_model_id or source_provider in DIRECT_PROVIDERS:
            continue
        if source_providers and source_provider not in source_providers:
            continue

        key = (source_provider, source_model_id)
        existing = active_routes.get(key)
        target = (
            str(existing.get("route_model_id"))
            if existing and existing.get("route_model_id")
            else aliases.get(f"{source_provider}/{source_model_id}")
        )
        if not target:
            skipped.append({"source": f"{source_provider}/{source_model_id}", "reason": "no-reviewed-or-model-id"})
            continue
        if target not in catalog_by_id:
            skipped.append(
                {
                    "source": f"{source_provider}/{source_model_id}",
                    "reason": f"or-model-not-in-text-catalog:{target}",
                }
            )
            continue

        observed_provider = str(existing.get("observed_provider") or "") if existing else ""
        observed_slug = str(existing.get("observed_provider_slug") or "") if existing else ""
        endpoint_evidence: dict[str, Any] | None = None
        if not observed_slug:
            try:
                endpoint_evidence = endpoint_fetcher(target)
                observed_provider, observed_slug = _first_provider(endpoint_evidence)
            except Exception as exc:  # one model must not stop the wave
                skipped.append(
                    {
                        "source": f"{source_provider}/{source_model_id}",
                        "reason": f"endpoint-fetch-failed:{type(exc).__name__}",
                    }
                )
                continue
        if not observed_slug:
            skipped.append({"source": f"{source_provider}/{source_model_id}", "reason": "endpoint-provider-missing"})
            continue

        route = {
            "source_provider": source_provider,
            "source_model_id": source_model_id,
            "route_decision_version": ROUTE_DECISION_VERSION,
            "route_snapshot_at": snapshot_at,
            "route_probe_id": f"coverage:{stable_hash(f'{snapshot_at}:{source_provider}/{source_model_id}')[:20]}",
            "transport_provider": "openrouter",
            "route_model_id": target,
            "route_provider_slug": observed_slug,
            "observed_provider": observed_provider or observed_slug,
            "observed_provider_slug": observed_slug,
            "provider_metadata_verified": True,
            "route_policy": OR_SERVED_POLICY,
            "route_reasoning_exclude": bool(existing and existing.get("route_reasoning_exclude")),
            "route_revocation_generation": int(existing.get("route_revocation_generation", 0)) if existing else 0,
            "state": "candidate",
            "canary_state": "availability_passed",
            "canary_successes": 0,
            "canary_required_successes": DEFAULT_ATTEMPTS,
            "audit_decision": "route-or",
            "audit_reason": "marketplace-wave-reviewed-identity",
            "pricing": {"openrouter": _pricing(catalog_by_id, target)},
        }
        if existing:
            route["previous_route_snapshot_at"] = existing.get("route_snapshot_at")
            route["previous_route_policy"] = existing.get("route_policy")
        if endpoint_evidence:
            route["endpoint_evidence"] = endpoint_evidence
        decisions.append(route)

    decisions.sort(key=lambda item: (item["source_provider"], item["source_model_id"]))
    return {
        "schema_version": 1,
        "generated_at": snapshot_at,
        "route_decision_version": ROUTE_DECISION_VERSION,
        "route_policy": OR_SERVED_POLICY,
        "source_providers": sorted({row["source_provider"] for row in decisions}),
        "catalog": {
            "url": catalog.get("url"),
            "fetched_at": catalog.get("fetched_at"),
            "total_count": catalog.get("total_count"),
            "returned_count": len(catalog_rows),
            "sha256": stable_hash(catalog_rows),
        },
        "providers": {
            "url": providers.get("url"),
            "fetched_at": providers.get("fetched_at"),
            "returned_count": len(providers.get("data") or []),
            "slugs": sorted(str(row["slug"]) for row in providers.get("data") or [] if row.get("slug")),
        },
        "counts": {"candidate": len(decisions), "skipped": len(skipped)},
        "skipped": skipped,
        "decisions": decisions,
    }


def _open_db() -> tuple[MongoClient, Any]:
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise ValueError("MONGODB_URI is required")
    client = MongoClient(uri)
    return client, client[os.environ.get("MONGODB_DB", "llm-bench")]


def command_build(args: argparse.Namespace) -> int:
    client, db = _open_db()
    try:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        catalog = fetch_catalog(args.base_url, api_key=api_key)
        providers = fetch_providers(args.base_url, api_key=api_key)
        aliases = load_aliases(args.alias_file)
        source_providers = set(args.source_provider) if args.source_provider else None
        report = build_report(
            db=db,
            aliases=aliases,
            catalog=catalog,
            providers=providers,
            endpoint_fetcher=lambda model_id: fetch_endpoints(args.base_url, model_id, api_key=api_key),
            source_providers=source_providers,
        )
    finally:
        client.close()
    write_json(args.output, report)
    print(json.dumps(report["counts"], sort_keys=True))
    print(f"wrote {args.output}")
    return 0


def command_canary(args: argparse.Namespace) -> int:
    report = load_json(args.decisions)
    candidates = [row for row in report.get("decisions", []) if row.get("state") == "candidate"]
    if args.source_provider:
        allowed = set(args.source_provider)
        candidates = [row for row in candidates if row.get("source_provider") in allowed]
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        source_provider = str(candidate["source_provider"])
        source_model_id = str(candidate["source_model_id"])
        pricing = candidate.get("pricing") or {}
        rates = pricing.get("openrouter") or {}
        max_rate = max(float(rates.get("input_per_token", 0) or 0), float(rates.get("output_per_token", 0) or 0))
        estimated_input_tokens = max(1, ceil(len("Tell a long and happy story about the history of the world.") / 4))
        estimated_cost = args.attempts * (estimated_input_tokens + args.max_tokens) * max_rate
        output = args.output_dir / f"{safe_filename(source_provider, source_model_id)}__canary.json"
        try:
            ledger = reserve_daily_budget(
                args.daily_ledger,
                amount_usd=min(args.max_cost_usd, max(0.001, estimated_cost)),
                batch_max_usd=args.batch_max_cost_usd,
                daily_max_usd=args.daily_max_cost_usd,
                operation=f"or-served-canary:{source_provider}/{source_model_id}",
            )
            result = run_routed_canary(
                report,
                provider=source_provider,
                model_id=source_model_id,
                attempts_count=args.attempts,
                max_tokens=args.max_tokens,
                deadline_seconds=args.deadline_seconds,
                min_success_rate=args.min_success_rate,
            )
            result["budget"] = {
                "estimated_cost_usd": estimated_cost,
                "daily_reserved_usd": ledger["reserved_usd"],
                "daily_ledger": str(args.daily_ledger),
            }
        except Exception as exc:  # preserve a decision for this row
            result = {
                "schema_version": 1,
                "mode": "report-only-routed-canary",
                "generated_at": utc_now(),
                "source_provider": source_provider,
                "source_model_id": source_model_id,
                "route_model_id": candidate.get("route_model_id"),
                "route_policy": OR_SERVED_POLICY,
                "evaluation": {
                    "mode": "report-only-routed-canary",
                    "canary_state": "failed",
                    "promotion_valid": False,
                    "successful": 0,
                    "required": args.attempts,
                    "reasons": [f"wave-exception:{type(exc).__name__}:{exc}"],
                },
            }
        write_json(output, result)
        evaluation = result.get("evaluation") or {}
        summary = {
            "source": f"{source_provider}/{source_model_id}",
            "route": candidate.get("route_model_id"),
            "state": evaluation.get("canary_state"),
            "promotion_valid": evaluation.get("promotion_valid"),
            "observed_provider_slug": evaluation.get("observed_provider_slug"),
            "output": str(output),
        }
        results.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    write_json(args.output_dir / "_summary.json", {"candidates": len(candidates), "results": results})
    return 0


def command_promote(args: argparse.Namespace) -> int:
    report = load_json(args.decisions)
    candidates = {
        (str(row.get("source_provider")), str(row.get("source_model_id"))): row
        for row in report.get("decisions", [])
        if isinstance(row, dict)
    }
    client = None
    db = None
    if args.apply:
        client, db = _open_db()
    promoted: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    try:
        for path in sorted(args.canary_dir.glob("*__canary.json")):
            canary = load_json(path)
            key = (str(canary.get("source_provider")), str(canary.get("source_model_id")))
            candidate = candidates.get(key)
            evaluation = canary.get("evaluation") or {}
            if not candidate or evaluation.get("promotion_valid") is not True:
                if candidate:
                    failed.append({"source": f"{key[0]}/{key[1]}", "reason": evaluation.get("reasons")})
                continue
            generation = int(candidate.get("route_revocation_generation", 0) or 0)
            if db is not None:
                generation = route_revocation_generation(
                    db,
                    provider=key[0],
                    model_id=key[1],
                )
            evidence_uri = f"{args.evidence_base.rstrip('/')}/{path.name}"
            try:
                route = promote_or_served_route(
                    candidate,
                    canary,
                    evidence_path=path,
                    evidence_uri=evidence_uri,
                    expires_hours=args.expires_hours,
                    revocation_generation=generation,
                )
            except Exception as exc:
                failed.append({"source": f"{key[0]}/{key[1]}", "reason": f"promotion:{type(exc).__name__}:{exc}"})
                continue
            write_json(args.output_dir / f"{safe_filename(key[0], key[1])}__route.json", route)
            if db is not None:
                collection = db[route_decisions_collection_name()]
                collection.update_one(
                    {
                        "source_provider": route["source_provider"],
                        "source_model_id": route["source_model_id"],
                        "route_snapshot_at": route["route_snapshot_at"],
                    },
                    {"$set": route, "$setOnInsert": {"created_at": datetime.now(timezone.utc)}},
                    upsert=True,
                )
                collection.update_many(
                    {
                        "source_provider": route["source_provider"],
                        "source_model_id": route["source_model_id"],
                        "state": "active",
                        "route_snapshot_at": {"$ne": route["route_snapshot_at"]},
                    },
                    {
                        "$set": {
                            "state": "superseded",
                            "superseded_at": datetime.now(timezone.utc),
                            "superseded_by": route["route_snapshot_at"],
                            "superseded_reason": "replaced-by-or-served-wave",
                        }
                    },
                )
            promoted.append(
                {
                    "source": f"{key[0]}/{key[1]}",
                    "route": route["route_model_id"],
                    "observed_provider_slug": route["observed_provider_slug"],
                }
            )
            print(json.dumps(promoted[-1], sort_keys=True), flush=True)
    finally:
        if client is not None:
            client.close()
    summary = {"promoted": len(promoted), "failed": len(failed), "routes": promoted, "failures": failed}
    write_json(args.output_dir / "_summary.json", summary)
    print(json.dumps({"promoted": len(promoted), "failed": len(failed)}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--alias-file", type=Path, action="append", required=True)
    build.add_argument("--source-provider", action="append")
    build.add_argument("--base-url", default=DEFAULT_BASE_URL)
    build.add_argument("--output", type=Path, required=True)
    build.set_defaults(func=command_build)

    canary = subparsers.add_parser("canary")
    canary.add_argument("--decisions", type=Path, required=True)
    canary.add_argument("--output-dir", type=Path, required=True)
    canary.add_argument("--daily-ledger", type=Path, required=True)
    canary.add_argument("--source-provider", action="append")
    canary.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    canary.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    canary.add_argument("--deadline-seconds", type=int, default=DEFAULT_DEADLINE_SECONDS)
    canary.add_argument("--min-success-rate", type=float, default=DEFAULT_MIN_SUCCESS_RATE)
    canary.add_argument("--max-cost-usd", type=float, default=2.0)
    canary.add_argument("--batch-max-cost-usd", type=float, default=DEFAULT_BATCH_MAX_USD)
    canary.add_argument("--daily-max-cost-usd", type=float, default=DEFAULT_DAILY_MAX_USD)
    canary.set_defaults(func=command_canary)

    promote = subparsers.add_parser("promote")
    promote.add_argument("--decisions", type=Path, required=True)
    promote.add_argument("--canary-dir", type=Path, required=True)
    promote.add_argument("--output-dir", type=Path, required=True)
    promote.add_argument("--evidence-base", required=True)
    promote.add_argument("--expires-hours", type=float, default=72.0)
    promote.add_argument("--apply", action="store_true")
    promote.add_argument("--yes", action="store_true")
    promote.set_defaults(func=command_promote)

    args = parser.parse_args()
    if getattr(args, "apply", False) and not args.yes:
        raise ValueError("--apply requires --yes")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
