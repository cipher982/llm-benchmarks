#!/usr/bin/env python3
"""Derive bounded canary pricing inputs from frozen provider endpoint evidence.

This helper is intentionally conservative. It only creates pricing for a
reviewed route candidate whose source provider is the same provider slug pinned
in the OpenRouter endpoint evidence. When several endpoints for that provider
exist, it uses the highest prompt and completion rate as a cost upper bound.
The direct and routed entries therefore represent the same provider's
published rate, not an assertion that OpenRouter's platform margin is zero.
"""

from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _provider_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).casefold()).strip("-")


def _rate(pricing: dict[str, Any], key: str) -> float:
    value = pricing.get(key)
    if value is None:
        raise ValueError(f"endpoint pricing is missing {key}")
    rate = float(value)
    if rate < 0:
        raise ValueError(f"endpoint pricing has negative {key}")
    return rate


def load_endpoint_evidence(path: Path) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            if "/._" in member.name or member.name.startswith("._"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            payload = json.load(handle)
            route_id = str(payload.get("url", "")).split("/models/", 1)[-1].removesuffix("/endpoints")
            data = payload.get("data", {}).get("data", {})
            if route_id and isinstance(data, dict):
                evidence[route_id] = data
    return evidence


def derive(candidate: dict[str, Any], data: dict[str, Any], *, source_uri: str) -> dict[str, Any]:
    source_provider = str(candidate.get("source_provider") or "")
    route_slug = str(candidate.get("route_provider_slug") or "")
    if _provider_key(source_provider) != _provider_key(route_slug):
        raise ValueError(
            f"cannot derive same-provider pricing for {source_provider}/{candidate.get('source_model_id')} "
            f"with route slug {route_slug}"
        )
    endpoints = [
        endpoint
        for endpoint in data.get("endpoints", [])
        if isinstance(endpoint, dict)
        and endpoint.get("status", 0) == 0
        and _provider_key(endpoint.get("provider_name")) == _provider_key(route_slug)
    ]
    status_fallback = False
    if not endpoints:
        endpoints = [
            endpoint
            for endpoint in data.get("endpoints", [])
            if isinstance(endpoint, dict) and _provider_key(endpoint.get("provider_name")) == _provider_key(route_slug)
        ]
        status_fallback = bool(endpoints)
    if not endpoints:
        raise ValueError(f"no {route_slug} endpoint for {data.get('id')}")
    prompt = max(_rate(endpoint.get("pricing", {}), "prompt") for endpoint in endpoints)
    completion = max(_rate(endpoint.get("pricing", {}), "completion") for endpoint in endpoints)
    cached_values = [
        float(endpoint["pricing"]["input_cache_read"])
        for endpoint in endpoints
        if endpoint.get("pricing", {}).get("input_cache_read") is not None
    ]
    cached = max(cached_values) if cached_values else prompt
    evidence = {
        "source_uri": source_uri,
        "route_model_id": candidate.get("route_model_id"),
        "route_provider_slug": route_slug,
        "endpoint_count": len(endpoints),
        "pricing_basis": (
            "same-provider-endpoint-upper-bound-status-fallback"
            if status_fallback
            else "same-provider-endpoint-upper-bound"
        ),
        "endpoint_statuses": sorted({int(endpoint.get("status", 0)) for endpoint in endpoints}),
        "provider_names": sorted({str(endpoint.get("provider_name")) for endpoint in endpoints}),
    }
    rates = {
        "input_per_token": prompt,
        "cached_input_per_token": cached,
        "output_per_token": completion,
    }
    return {
        "currency": "USD",
        "unit": "per_token",
        "direct": {**rates, "evidence": evidence},
        "openrouter": {**rates, "evidence": evidence},
        "evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--endpoints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    decisions = load_json(args.decisions).get("decisions", [])
    evidence = load_endpoint_evidence(args.endpoints)
    generated = 0
    skipped: list[str] = []
    for candidate in decisions:
        if candidate.get("state") != "candidate":
            continue
        provider = str(candidate.get("source_provider") or "")
        model_id = str(candidate.get("source_model_id") or "")
        route_id = str(candidate.get("route_model_id") or "")
        try:
            pricing = derive(candidate, evidence[route_id], source_uri=str(args.endpoints))
        except (KeyError, ValueError) as exc:
            skipped.append(f"{provider}/{model_id}: {exc}")
            continue
        filename = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{provider}__{model_id}") + ".json"
        write_json(args.output_dir / filename, pricing)
        generated += 1
    summary = {"generated": generated, "skipped": skipped}
    write_json(args.output_dir / "_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
