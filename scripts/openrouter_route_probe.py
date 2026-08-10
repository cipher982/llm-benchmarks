#!/usr/bin/env python3
"""Run bounded, streaming OpenRouter route-availability probes.

This is a report-only probe. It never writes MongoDB or changes the scheduler.
It sends the production-shaped Chat Completions request with a provider
restriction, disabled fallbacks, required parameters, and routing metadata.
The output is evidence input for ``openrouter_coverage_audit.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request
from urllib.request import urlopen

from openrouter_budget import DEFAULT_BATCH_MAX_USD
from openrouter_budget import DEFAULT_DAILY_MAX_USD
from openrouter_budget import reserve_daily_budget
from openrouter_coverage_audit import PROVIDER_SLUGS
from openrouter_coverage_audit import norm
from openrouter_coverage_audit import utc_now

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
PROMPT = "Reply with OK."
PROBE_SCHEMA_VERSION = 2


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def provider_slug(provider: str, mapping: dict[str, Any]) -> str | None:
    explicit = mapping.get(provider)
    if isinstance(explicit, str) and explicit:
        return explicit
    defaults = PROVIDER_SLUGS.get(provider, ())
    return defaults[0] if defaults else None


def parse_sse(response) -> dict[str, Any]:
    content_parts: list[str] = []
    providers: list[str] = []
    generation_id = response.headers.get("X-Generation-Id")
    finish_reason = None
    usage = None
    metadata = None
    saw_done = False
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        generation_id = generation_id or chunk.get("id")
        if chunk.get("provider"):
            providers.append(str(chunk["provider"]))
        if isinstance(chunk.get("openrouter_metadata"), dict):
            metadata = chunk["openrouter_metadata"]
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(str(delta["content"]))
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    selected = []
    if isinstance(metadata, dict):
        endpoints = metadata.get("endpoints") or {}
        selected = [item for item in endpoints.get("available", []) if isinstance(item, dict) and item.get("selected")]
    observed = str(selected[0].get("provider") if selected else (providers[-1] if providers else ""))
    return {
        "generation_id": generation_id,
        "observed_provider": observed or None,
        "metadata": metadata,
        "metadata_selected": selected,
        "provider_header": response.headers.get("X-Provider-Name"),
        "content_chars": len("".join(content_parts)),
        "finish_reason": finish_reason,
        "usage": usage,
        "saw_done": saw_done,
    }


def probe_once(
    row: dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    max_tokens: int,
    timeout: float,
    route_slug: str,
    attempt: int,
    profile_id: str,
    profile_hash: str,
) -> dict[str, Any]:
    source_key = str(row["source_key"])
    model_id = str(row["or_model_id"])
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "provider": {
            "only": [route_slug],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }
    request = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Metadata": "enabled",
            "User-Agent": "llm-bench-coverage-audit/1",
        },
        method="POST",
    )
    result: dict[str, Any] = {
        "source_key": source_key,
        "or_model_id": model_id,
        "attempt": attempt,
        "profile_id": profile_id,
        "profile_hash": profile_hash,
        "route_provider_slug": route_slug,
        "started_at": utc_now(),
        "request": {
            "model": model_id,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "provider": body["provider"],
        },
    }
    result["effective_request_hash"] = stable_hash(body)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed API base URL from CLI
            parsed = parse_sse(response)
        observed = parsed.get("observed_provider") or ""
        observed_slug = route_slug if norm(observed) == norm(route_slug) else None
        usage = parsed.get("usage") or {}
        result.update(
            {
                "status": "success" if parsed.get("saw_done") else "incomplete",
                "observed_provider": observed or None,
                "observed_provider_slug": observed_slug,
                "provider_metadata_verified": bool(parsed.get("metadata_selected")) and observed_slug is not None,
                "usable_output": parsed.get("content_chars", 0) > 0 and int(usage.get("completion_tokens", 0) or 0) > 0,
                "response_id": parsed.get("generation_id"),
                "finish_reason": parsed.get("finish_reason"),
                "usage": usage,
                "metadata": parsed.get("metadata"),
                "provider_header": parsed.get("provider_header"),
            }
        )
    except HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")[:500]
        result.update({"status": "error", "reason_class": f"http-{exc.code}", "error": body_text})
    except Exception as exc:  # noqa: BLE001 - preserve per-row probe outcomes
        result.update({"status": "error", "reason_class": type(exc).__name__, "error": str(exc)[:500]})
    result["finished_at"] = utc_now()
    return result


def summarize(row: dict[str, Any], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [
        attempt
        for attempt in attempts
        if attempt.get("status") == "success"
        and attempt.get("usable_output")
        and attempt.get("provider_metadata_verified")
    ]
    route_slug = str(row["route_provider_slug"])
    observed = next((a.get("observed_provider") for a in successful if a.get("observed_provider")), None)
    observed_slug = next((a.get("observed_provider_slug") for a in successful if a.get("observed_provider_slug")), None)
    error_classes = [str(attempt.get("reason_class") or "") for attempt in attempts]
    if successful and len(successful) == len(attempts):
        reason_class = None
    elif any(value == "http-429" for value in error_classes):
        reason_class = "transient-rate-limit"
    elif attempts and all(
        attempt.get("status") == "success"
        and int((attempt.get("usage") or {}).get("completion_tokens", 0) or 0) > 0
        and int(attempt.get("content_chars", 0) or 0) == 0
        for attempt in attempts
    ):
        reason_class = "visible-output-empty"
    else:
        reason_class = "probe-failed-or-incomplete"
    return {
        "source_key": row["source_key"],
        "or_model_id": row["or_model_id"],
        "status": "success" if len(successful) == len(attempts) else "partial-or-failed",
        "reason_class": reason_class,
        "route_provider_slug": route_slug,
        "observed_provider": observed,
        "observed_provider_slug": observed_slug,
        "provider_metadata_verified": len(successful) == len(attempts),
        "usable_output": len(successful) == len(attempts),
        "successful_attempts": len(successful),
        "attempts": attempts,
        "identity_evidence_verified": bool((row.get("evidence") or {}).get("identity", {}).get("verified")),
        "profile_id": attempts[0].get("profile_id") if attempts else None,
        "profile_hash": attempts[0].get("profile_hash") if attempts else None,
        "effective_request_hashes": sorted(
            {str(attempt["effective_request_hash"]) for attempt in attempts if attempt.get("effective_request_hash")}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--replay-json", type=Path, help="Reclassify a prior probe report without making requests")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider-map-json", type=Path)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-probes", type=int, default=200)
    parser.add_argument("--max-cost-usd", type=float, default=10.0)
    parser.add_argument("--batch-max-cost-usd", type=float, default=DEFAULT_BATCH_MAX_USD)
    parser.add_argument("--daily-max-cost-usd", type=float, default=DEFAULT_DAILY_MAX_USD)
    parser.add_argument("--daily-ledger-json", type=Path)
    parser.add_argument("--estimated-cost-per-probe", type=float, default=0.02)
    parser.add_argument("--profile-id", default="cloud-default-v1")
    args = parser.parse_args()

    if args.replay_json:
        previous = load_json(args.replay_json)
        replayed_rows = [
            summarize(row, row.get("attempts", []))
            for row in previous.get("rows", [])
            if isinstance(row, dict) and row.get("source_key")
        ]
        previous["generated_at"] = utc_now()
        previous["replayed_from"] = str(args.replay_json)
        previous["rows"] = sorted(replayed_rows, key=lambda item: str(item["source_key"]))
        previous["schema_version"] = PROBE_SCHEMA_VERSION
        write_json(args.output, previous)
        print(f"replayed {len(replayed_rows)} rows")
        print(f"wrote {args.output}")
        return 0

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required")
    if args.max_probes < 1 or args.max_cost_usd <= 0 or args.estimated_cost_per_probe <= 0:
        raise ValueError("probe budget values must be positive")
    if not args.daily_ledger_json:
        raise ValueError("live probes require --daily-ledger-json shared budget ledger")
    if args.max_cost_usd > args.batch_max_cost_usd:
        raise ValueError("--max-cost-usd cannot exceed --batch-max-cost-usd")
    profile = {"profile_id": args.profile_id, "max_tokens": args.max_tokens, "prompt": PROMPT, "stream": True}
    profile_hash = stable_hash(profile)
    audit = load_json(args.audit_json)
    provider_map = load_json(args.provider_map_json) if args.provider_map_json else {}
    if not isinstance(provider_map, dict):
        raise ValueError("provider map must be an object")
    candidates = []
    for row in audit.get("rows", []):
        if row.get("reason_class") != "needs-pinned-probe" or not row.get("or_model_id"):
            continue
        route_slug = provider_slug(str(row["provider"]), provider_map)
        if route_slug:
            candidates.append({**row, "route_provider_slug": route_slug})
    max_by_budget = min(args.max_probes, int(args.max_cost_usd / args.estimated_cost_per_probe))
    if args.limit:
        max_by_budget = min(max_by_budget, args.limit)
    unprobed = candidates[max_by_budget:]
    candidates = candidates[:max_by_budget]
    estimated_requests = len(candidates) * args.attempts
    reserved_cost = min(args.max_cost_usd, estimated_requests * args.estimated_cost_per_probe)
    ledger = reserve_daily_budget(
        args.daily_ledger_json,
        amount_usd=reserved_cost,
        batch_max_usd=args.batch_max_cost_usd,
        daily_max_usd=args.daily_max_cost_usd,
        operation="availability-probe",
    )

    summaries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = {
            pool.submit(
                lambda item=row: summarize(
                    item,
                    [
                        probe_once(
                            item,
                            api_key=api_key,
                            base_url=args.base_url,
                            max_tokens=args.max_tokens,
                            timeout=args.timeout,
                            route_slug=str(item["route_provider_slug"]),
                            attempt=attempt,
                            profile_id=args.profile_id,
                            profile_hash=profile_hash,
                        )
                        for attempt in range(1, args.attempts + 1)
                    ],
                )
            ): row
            for row in candidates
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "source_key": row["source_key"],
                    "or_model_id": row["or_model_id"],
                    "status": "error",
                    "reason_class": type(exc).__name__,
                    "route_provider_slug": row["route_provider_slug"],
                    "provider_metadata_verified": False,
                    "usable_output": False,
                }
            summaries.append(result)
            print(f"probe {index}/{len(candidates)} {row['source_key']}", file=sys.stderr, flush=True)

    report = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "mode": "report-only-probe",
        "audit_snapshot": str(args.audit_json),
        "attempts_per_row": args.attempts,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "profile": profile,
        "profile_hash": profile_hash,
        "budget": {
            "max_probes": args.max_probes,
            "max_cost_usd": args.max_cost_usd,
            "estimated_cost_per_probe": args.estimated_cost_per_probe,
            "estimated_requests": estimated_requests,
            "scheduled_rows": len(candidates),
            "unprobed_rows": len(unprobed),
            "status": "exhausted" if unprobed else "within-budget",
            "daily_ledger": str(args.daily_ledger_json),
            "daily_reserved_usd": ledger["reserved_usd"],
        },
        "budget_exhausted_source_keys": [str(row["source_key"]) for row in unprobed],
        "rows": sorted(summaries, key=lambda item: str(item["source_key"])),
    }
    write_json(args.output, report)
    success = sum(1 for row in summaries if row.get("status") == "success")
    print(json.dumps({"candidate_rows": len(candidates), "successful_rows": success}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
