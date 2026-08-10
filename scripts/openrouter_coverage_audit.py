#!/usr/bin/env python3
"""Build a conservative, report-only OpenRouter coverage audit.

This tool never writes MongoDB and never enables, disables, or queues a model.
It joins a frozen enabled-model snapshot to an OpenRouter model snapshot and,
when available, endpoint-listing snapshots. Name similarity is used only to
report ambiguity. A row becomes ``route-or`` only when an exact or reviewed
alias identity, primary identity evidence, and a successful observed-provider
probe are supplied in the evidence input.

The first pass is intentionally useful without probes: it identifies exact
candidate IDs, endpoint metadata, and the rows that must remain direct. A later
probe pass can add evidence without changing this decision logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request
from urllib.request import urlopen

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# ``stream`` is a transport mode and is not consistently listed in
# OpenRouter's endpoint ``supported_parameters``. The live probe must verify
# streaming. ``max_tokens`` is the request parameter used by this bench and is
# safe to use as the catalog-level compatibility filter.
REQUIRED_PARAMETERS = ("max_tokens",)
IDENTITY_RULE_VERSION = "or-identity-v2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# OpenRouter uses display names in some endpoint responses and slugs in routing
# requests. This is deliberately a reviewed, small mapping. It is not a model
# identity matcher and does not authorize a route by itself.
PROVIDER_SLUGS: dict[str, tuple[str, ...]] = {
    "anthropic": ("anthropic",),
    "openai": ("openai",),
    "deepinfra": ("deepinfra",),
    "fireworks": ("fireworks", "fireworks-ai"),
    "together": ("together", "togetherai"),
    "groq": ("groq",),
    "cerebras": ("cerebras",),
    "vertex": ("google-vertex", "vertex"),
    "bedrock": (),
}

PROVIDER_DISPLAY_ALIASES: dict[str, tuple[str, ...]] = {
    "anthropic": ("anthropic",),
    "openai": ("openai",),
    "deepinfra": ("deepinfra",),
    "fireworks": ("fireworks", "fireworks ai"),
    "together": ("together", "together ai", "togetherai"),
    "groq": ("groq",),
    "cerebras": ("cerebras",),
    "vertex": ("google vertex", "google-vertex", "vertex"),
    "bedrock": ("amazon bedrock", "bedrock"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_json(url: str, *, api_key: str | None, timeout: float) -> Any:
    headers = {"Accept": "application/json", "User-Agent": "llm-bench-coverage-audit/1"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, headers=headers, method="GET")
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is CLI/config input
        return json.load(response)


def fetch_catalog(*, base_url: str, api_key: str | None, timeout: float) -> dict[str, Any]:
    fetched_at = utc_now()
    url = f"{base_url.rstrip('/')}/models"
    payload = request_json(url, api_key=api_key, timeout=timeout)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("OpenRouter models response has no data list")
    return {
        "fetched_at": fetched_at,
        "url": url,
        "total_count": payload.get("total_count"),
        "links": payload.get("links"),
        "data": payload["data"],
    }


def model_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        payload = payload["data"]
    if not isinstance(payload, list):
        raise ValueError("model snapshot must be a list or an object with data")
    return [row for row in payload if isinstance(row, dict) and row.get("id")]


def catalog_scope(payload: Any, *, observed_count: int) -> tuple[str, list[str]]:
    """Validate whether a catalog can support a negative match decision.

    A truncated or paginated catalog is useful for positive candidates but it
    cannot prove that a source model is absent. The caller must keep those rows
    as ``direct-unknown`` until a complete global snapshot is available.
    """

    if not isinstance(payload, dict):
        return "unknown", ["catalog-payload-not-object"]
    problems: list[str] = []
    total = payload.get("total_count")
    if not isinstance(total, int):
        problems.append("catalog-total-count-missing")
    elif total != observed_count:
        problems.append("catalog-count-mismatch")
    links = payload.get("links")
    if isinstance(links, dict) and links.get("next"):
        problems.append("catalog-pagination-present")
    scope = str(payload.get("catalog_scope") or "public-discovery").strip().lower()
    if scope != "global":
        problems.append("catalog-scope-not-global")
    repeated = payload.get("stable_repeated_count")
    if repeated is not True:
        problems.append("catalog-count-not-stable")
    repeated_counts = payload.get("stable_repeated_counts")
    if (
        not isinstance(repeated_counts, list)
        or len(repeated_counts) < 2
        or len({value for value in repeated_counts if isinstance(value, int)}) != 1
        or any(not isinstance(value, int) for value in repeated_counts)
        or any(value != observed_count for value in repeated_counts)
    ):
        problems.append("catalog-repeat-evidence-missing")
    if problems:
        return ("public-discovery" if scope == "public-discovery" else "incomplete"), problems
    return "global", problems


def enabled_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        payload = payload["data"]
    if not isinstance(payload, list):
        raise ValueError("enabled model snapshot must be a list or an object with data")
    return [
        row
        for row in payload
        if isinstance(row, dict)
        and row.get("provider")
        and row.get("model_id")
        and row.get("enabled", True) is not False
        and row.get("deprecated") is not True
    ]


def source_key(row: dict[str, Any]) -> str:
    return f"{row['provider']}/{row['model_id']}"


def norm(value: str) -> str:
    return str(value).strip().casefold()


def slug(or_id: str) -> str:
    return or_id.rsplit("/", 1)[-1]


def diagnostic_stem(model_id: str) -> str:
    """Build a version-dot-safe diagnostic stem, never an approval key.

    Source IDs can contain organization prefixes, Bedrock regions, provider
    suffixes, and version dots such as ``Qwen2.5``. Splitting on dots destroys
    those versions, so only slash separators are structural here. This value is
    retained for reviewer ranking and debugging, never for route activation.
    """

    core = str(model_id).split(":", 1)[0]
    core = core.split("/")[-1]
    core = re.sub(r"^(?:us|eu|ap|sa|ca|me|af)\.", "", core, flags=re.IGNORECASE)
    core = re.sub(r"^(?:anthropic|amazon|meta|mistral)\.", "", core, flags=re.IGNORECASE)
    core = core.casefold()
    core = re.sub(r"-v\d+(?:\.\d+)?$", "", core)
    core = re.sub(r"-(?:\d{8}|\d{4})$", "", core)
    return core.strip("-._")


def exact_candidates(model_id: str, catalog: list[dict[str, Any]]) -> list[str]:
    """Return only case-insensitive full-ID matches, never fuzzy matches."""
    wanted = norm(model_id)
    return [str(row["id"]) for row in catalog if norm(str(row["id"])) == wanted]


def unique_slug_candidates(model_id: str, catalog: list[dict[str, Any]]) -> list[str]:
    """Find a unique slug match for IDs without an organization prefix.

    This is a candidate only. It is not enough to route without endpoint and
    probe evidence because unrelated organizations can reuse a slug.
    """
    if "/" in model_id:
        return []
    wanted = norm(model_id)
    matches = [str(row["id"]) for row in catalog if norm(slug(str(row["id"]))) == wanted]
    return sorted(set(matches))


def alias_records(key: str, aliases: dict[str, Any]) -> list[dict[str, Any]]:
    """Read both the v1 reviewed schema and the old keyed-map fixture shape."""

    if isinstance(aliases.get("aliases"), list):
        return [item for item in aliases["aliases"] if isinstance(item, dict) and str(item.get("source_key")) == key]
    value = aliases.get(key)
    if not value:
        return []
    values = [value] if isinstance(value, str) else value
    if isinstance(values, list) and all(isinstance(item, str) for item in values):
        return [{"source_key": key, "target_or_model_id": item} for item in values]
    return [item for item in values if isinstance(item, dict)] if isinstance(values, list) else []


def alias_candidates(key: str, aliases: dict[str, Any], catalog: list[dict[str, Any]]) -> list[str]:
    records = alias_records(key, aliases)
    catalog_ids = {norm(str(row["id"])): str(row["id"]) for row in catalog}
    result: list[str] = []
    for record in records:
        target = record.get("target_or_model_id") or record.get("target")
        values = [target] if isinstance(target, str) else target
        for item in values or []:
            if norm(str(item)) in catalog_ids:
                result.append(catalog_ids[norm(str(item))])
    return result


def reviewed_alias_evidence(
    aliases: dict[str, Any],
    *,
    refs: Any,
    reviewers: Any,
    rule_version: Any,
) -> tuple[bool, dict[str, Any]]:
    """Validate alias references against the signed-in-place evidence manifest."""

    manifest = aliases.get("evidence_manifest")
    receipts = aliases.get("review_receipts")
    if not isinstance(manifest, dict) or not isinstance(receipts, dict):
        return False, {"reason": "alias-evidence-manifest-missing"}
    if rule_version != aliases.get("rule_version"):
        return False, {"reason": "alias-rule-version-mismatch"}
    if not isinstance(refs, list) or not refs:
        return False, {"reason": "alias-evidence-refs-missing"}
    resolved_refs: list[dict[str, Any]] = []
    for ref in refs:
        item = manifest.get(ref) if isinstance(ref, str) else None
        if not isinstance(item, dict) or not isinstance(item.get("uri"), str):
            return False, {"reason": "alias-evidence-ref-unresolved", "ref": ref}
        if not SHA256_RE.fullmatch(str(item.get("sha256") or "")):
            return False, {"reason": "alias-evidence-hash-invalid", "ref": ref}
        resolved_refs.append({"ref": ref, "uri": item["uri"], "sha256": item["sha256"]})
    if not isinstance(reviewers, list) or len(reviewers) < 2:
        return False, {"reason": "alias-reviewer-quorum-missing"}
    resolved_reviewers: list[dict[str, Any]] = []
    for reviewer in reviewers:
        receipt = receipts.get(reviewer) if isinstance(reviewer, str) else None
        try:
            reviewed_count = int(receipt.get("reviewed_count", 0) or 0) if isinstance(receipt, dict) else 0
        except (TypeError, ValueError):
            reviewed_count = 0
        if (
            not isinstance(receipt, dict)
            or not str(receipt.get("run_id", "")).startswith("hatch_")
            or not isinstance(receipt.get("uri"), str)
            or not SHA256_RE.fullmatch(str(receipt.get("sha256") or ""))
            or receipt.get("verdict") != "approved"
            or reviewed_count < 1
        ):
            return False, {"reason": "alias-review-receipt-unresolved", "reviewer": reviewer}
        resolved_reviewers.append(
            {
                "reviewer": reviewer,
                "run_id": receipt["run_id"],
                "uri": receipt["uri"],
                "sha256": receipt["sha256"],
                "verdict": receipt["verdict"],
                "reviewed_count": reviewed_count,
            }
        )
    return True, {"evidence": resolved_refs, "reviewers": resolved_reviewers}


def identity_evidence_for(
    key: str,
    *,
    source_row: dict[str, Any],
    aliases: dict[str, Any],
    catalog_row: dict[str, Any],
    route_model_id: str,
    catalog_snapshot_hash: str | None = None,
) -> dict[str, Any]:
    """Return primary evidence for a source-to-OR identity assertion.

    A unique slug is never evidence. Exact IDs can use a matching canonical OR
    record only when the record carries a canonical slug or official metadata;
    reviewed aliases must carry explicit evidence references and reviewer
    attestations.
    """

    records = [
        record for record in alias_records(key, aliases) if (record.get("target_or_model_id") or record.get("target"))
    ]
    for record in records:
        targets = record.get("target_or_model_id") or record.get("target")
        if isinstance(targets, str):
            targets = [targets]
        if route_model_id in {str(item) for item in targets or []}:
            refs = record.get("evidence_refs") or record.get("evidence")
            reviewers = record.get("reviewers")
            valid, resolved = reviewed_alias_evidence(
                aliases,
                refs=refs,
                reviewers=reviewers,
                rule_version=record.get("rule_version"),
            )
            if valid:
                return {
                    "verified": True,
                    "method": "reviewed-alias",
                    "evidence_refs": refs,
                    "reviewers": reviewers,
                    "rule_version": record.get("rule_version"),
                    "resolved_evidence": resolved["evidence"],
                    "resolved_reviewers": resolved["reviewers"],
                }
    source_metadata = source_row.get("identity_evidence")
    if isinstance(source_metadata, dict) and source_metadata.get("verified") is True:
        refs = source_metadata.get("evidence_refs") or source_metadata.get("evidence")
        if isinstance(refs, list) and refs:
            return {"verified": True, "method": "source-metadata", "evidence_refs": refs}
    canonical = catalog_row.get("canonical_slug")
    official_fields = (catalog_row.get("name"), catalog_row.get("description"), catalog_row.get("architecture"))
    if (
        norm(str(canonical or "")) == norm(route_model_id)
        and any(official_fields)
        and catalog_snapshot_hash
        and SHA256_RE.fullmatch(catalog_snapshot_hash)
    ):
        return {
            "verified": True,
            "method": "exact-id-canonical-or-record",
            "evidence_refs": [str(catalog_row.get("links", {}).get("details") or route_model_id)],
            "evidence_hashes": [catalog_snapshot_hash],
        }
    return {"verified": False, "method": "none", "evidence_refs": []}


def endpoint_provider_slug(endpoint: dict[str, Any]) -> str | None:
    for key in ("provider_slug", "slug"):
        value = endpoint.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def endpoint_provider_name(endpoint: dict[str, Any]) -> str:
    for key in ("provider_name", "provider", "name"):
        value = endpoint.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def provider_matches(source_provider: str, endpoint: dict[str, Any]) -> bool:
    observed_slug = norm(endpoint_provider_slug(endpoint) or "")
    if observed_slug and observed_slug in {norm(x) for x in PROVIDER_SLUGS.get(source_provider, ())}:
        return True
    observed_name = norm(endpoint_provider_name(endpoint)).replace("_", " ")
    aliases = {norm(x) for x in PROVIDER_DISPLAY_ALIASES.get(source_provider, ())}
    return observed_name in aliases


def supported_parameters(endpoint: dict[str, Any]) -> set[str]:
    values = endpoint.get("supported_parameters") or []
    return {norm(value) for value in values if isinstance(value, str)}


def endpoint_evidence(source_provider: str, endpoints_payload: Any) -> dict[str, Any]:
    if isinstance(endpoints_payload, dict):
        data = endpoints_payload.get("data", endpoints_payload)
    else:
        data = endpoints_payload
    # The saved fetch artifact wraps the API response in ``data`` once, and
    # the API response itself wraps the model object in ``data`` again.
    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        data = data["data"]
    endpoints = data.get("endpoints", []) if isinstance(data, dict) else []
    if not isinstance(endpoints, list):
        endpoints = []
    rows = [row for row in endpoints if isinstance(row, dict)]
    matches = [row for row in rows if provider_matches(source_provider, row)]
    compatible = [row for row in matches if set(REQUIRED_PARAMETERS) <= supported_parameters(row)]
    return {
        "endpoint_count": len(rows),
        "provider_match_count": len(matches),
        "compatible_count": len(compatible),
        "provider_matches": [
            {
                "provider_name": endpoint_provider_name(row),
                "provider_slug": endpoint_provider_slug(row),
                "supported_parameters": sorted(supported_parameters(row)),
                "name": row.get("name"),
            }
            for row in matches
        ],
    }


def load_probe_evidence(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = load_json(path)
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("probe evidence must be a list or an object with rows")
    return {str(row["source_key"]): row for row in rows if isinstance(row, dict) and row.get("source_key")}


def decide(
    row: dict[str, Any],
    *,
    catalog: list[dict[str, Any]],
    aliases: dict[str, Any],
    endpoint_payload: Any | None,
    probe: dict[str, Any] | None,
    catalog_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider = str(row["provider"])
    model_id = str(row["model_id"])
    key = source_key(row)
    base = {
        "source_key": key,
        "provider": provider,
        "model_id": model_id,
        "display_name": row.get("display_name"),
        "decision": "keep-direct",
        "reason_class": "unknown",
        "or_model_id": None,
        "route_provider_slug": None,
        "observed_provider": None,
        "evidence": {},
    }
    if provider == "bedrock":
        base["reason_class"] = "bedrock-out-of-scope"
        return base

    exact = exact_candidates(model_id, catalog)
    explicit = alias_candidates(key, aliases, catalog)
    slug_matches = unique_slug_candidates(model_id, catalog)
    candidates = sorted(set(exact + explicit))
    base["evidence"] = {
        "identity_rule_version": IDENTITY_RULE_VERSION,
        "diagnostic_stem": diagnostic_stem(model_id),
        "exact_id_candidates": exact,
        "explicit_alias_candidates": explicit,
        "unique_slug_candidates": slug_matches,
    }
    if len(candidates) != 1:
        base["reason_class"] = (
            "ambiguous-model-id"
            if len(slug_matches) > 1 or len(candidates) > 1
            else (
                "no-exact-or-ambiguous-model-id"
                if (catalog_meta or {}).get("scope") == "global"
                else "catalog-evidence-incomplete"
            )
        )
        if len(slug_matches) > 1:
            base["evidence"]["ambiguous_slug_candidates"] = slug_matches
        return base

    or_id = candidates[0]
    base["or_model_id"] = or_id
    catalog_row = next((item for item in catalog if str(item.get("id")) == or_id), {})
    identity = identity_evidence_for(
        key,
        source_row=row,
        aliases=aliases,
        catalog_row=catalog_row,
        route_model_id=or_id,
        catalog_snapshot_hash=(catalog_meta or {}).get("snapshot_sha256"),
    )
    base["evidence"]["identity"] = identity
    if isinstance(catalog_row.get("pricing"), dict):
        base["evidence"]["or_catalog_pricing"] = catalog_row["pricing"]
    if not identity.get("verified"):
        base["reason_class"] = "identity-evidence-missing"
        return base
    if endpoint_payload is None:
        base["reason_class"] = "endpoint-evidence-missing"
        return base

    endpoint_info = endpoint_evidence(provider, endpoint_payload)
    base["evidence"]["endpoint"] = endpoint_info
    if endpoint_info["provider_match_count"] == 0:
        base["reason_class"] = "source-provider-not-listed"
        return base
    if endpoint_info["compatible_count"] == 0:
        base["reason_class"] = "protocol-incompatible"
        return base
    base["reason_class"] = "needs-pinned-probe"
    if not probe:
        return base

    base["evidence"]["probe"] = probe
    if probe.get("status") != "success":
        base["reason_class"] = str(probe.get("reason_class") or "probe-failed")
        return base
    observed = norm(str(probe.get("observed_provider_slug") or ""))
    requested = norm(str(probe.get("route_provider_slug") or ""))
    if not observed or not requested or observed != requested:
        base["reason_class"] = "observed-provider-mismatch"
        return base
    if not probe.get("usable_output") or not probe.get("provider_metadata_verified"):
        base["reason_class"] = "probe-evidence-incomplete"
        return base
    if probe.get("identity_evidence_verified") is not True and not identity.get("verified"):
        base["reason_class"] = "identity-evidence-missing"
        return base
    base.update(
        {
            "decision": "route-or",
            "reason_class": "verified-pinned-route",
            "route_provider_slug": probe.get("route_provider_slug"),
            "observed_provider": probe.get("observed_provider"),
        }
    )
    return base


def fetch_endpoint_snapshots(
    *,
    catalog: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    aliases: dict[str, Any],
    base_url: str,
    api_key: str,
    output_dir: Path,
    timeout: float,
    delay: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ids: set[str] = set()
    for row in rows:
        candidates = exact_candidates(str(row["model_id"]), catalog)
        candidates += alias_candidates(source_key(row), aliases, catalog)
        ids.update(candidates)

    results: dict[str, Any] = {}
    for index, or_id in enumerate(sorted(ids), start=1):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", or_id)
        path = output_dir / f"{safe}.json"
        if path.exists():
            results[or_id] = load_json(path)
            continue
        author, model_slug = or_id.split("/", 1)
        url = f"{base_url.rstrip('/')}/models/{quote(author, safe='')}/{quote(model_slug, safe='')}/endpoints"
        try:
            payload = request_json(url, api_key=api_key, timeout=timeout)
            wrapped = {"fetched_at": utc_now(), "url": url, "data": payload}
        except HTTPError as exc:
            wrapped = {"fetched_at": utc_now(), "url": url, "error": f"HTTP {exc.code}"}
        except Exception as exc:  # noqa: BLE001 - report-only evidence must preserve per-model errors
            wrapped = {"fetched_at": utc_now(), "url": url, "error": f"{type(exc).__name__}: {exc}"}
        write_json(path, wrapped)
        results[or_id] = wrapped
        print(f"endpoint {index}/{len(ids)} {or_id}", file=sys.stderr, flush=True)
        if delay:
            time.sleep(delay)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-json", type=Path, required=True)
    parser.add_argument("--catalog-json", type=Path)
    parser.add_argument("--fetch-catalog", action="store_true")
    parser.add_argument("--endpoint-dir", type=Path)
    parser.add_argument("--fetch-endpoints", action="store_true")
    parser.add_argument("--aliases-json", type=Path)
    parser.add_argument("--probe-json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None, help="OpenRouter key; prefer OPENROUTER_API_KEY")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument(
        "--catalog-scope",
        choices=("public-discovery", "global"),
        default="public-discovery",
        help="Declare whether this snapshot is complete enough for negative matches.",
    )
    args = parser.parse_args()

    api_key = args.api_key
    if api_key is None:
        import os

        api_key = os.environ.get("OPENROUTER_API_KEY")

    rows = enabled_rows(load_json(args.models_json))
    aliases = load_json(args.aliases_json) if args.aliases_json else {}
    if not isinstance(aliases, dict):
        raise ValueError("aliases JSON must be an object keyed by provider/model_id")

    if args.fetch_catalog:
        catalog_snapshot = fetch_catalog(base_url=args.base_url, api_key=api_key, timeout=args.timeout)
        catalog = model_rows(catalog_snapshot)
        if args.catalog_json:
            write_json(args.catalog_json, catalog_snapshot)
    elif args.catalog_json:
        catalog_snapshot = load_json(args.catalog_json)
        catalog = model_rows(catalog_snapshot)
    else:
        raise ValueError("provide --catalog-json or --fetch-catalog")

    if isinstance(catalog_snapshot, dict):
        catalog_snapshot = dict(catalog_snapshot)
        catalog_snapshot.setdefault("catalog_scope", args.catalog_scope)
    scope, scope_problems = catalog_scope(catalog_snapshot, observed_count=len(catalog))
    catalog_meta = {
        "scope": scope,
        "problems": scope_problems,
        "snapshot_sha256": sha256_file(Path(args.catalog_json)) if args.catalog_json else None,
    }

    endpoint_payloads: dict[str, Any] = {}
    if args.fetch_endpoints:
        if not args.endpoint_dir or not api_key:
            raise ValueError("--fetch-endpoints requires --endpoint-dir and OPENROUTER_API_KEY")
        endpoint_payloads = fetch_endpoint_snapshots(
            catalog=catalog,
            rows=rows,
            aliases=aliases,
            base_url=args.base_url,
            api_key=api_key,
            output_dir=args.endpoint_dir,
            timeout=args.timeout,
            delay=args.delay,
        )
    elif args.endpoint_dir:
        for path in args.endpoint_dir.glob("*.json"):
            payload = load_json(path)
            url = payload.get("url", "") if isinstance(payload, dict) else ""
            match = re.search(r"/models/([^/]+/[^/]+)/endpoints", url)
            if match:
                endpoint_payloads[match.group(1)] = payload.get("data", payload)

    probes = load_probe_evidence(args.probe_json)
    audit_rows = []
    for row in sorted(rows, key=lambda item: source_key(item)):
        candidates = exact_candidates(str(row["model_id"]), catalog)
        candidates += alias_candidates(source_key(row), aliases, catalog)
        # Unique slug matches are retained in the audit row for diagnostics,
        # but never authorize an endpoint fetch or a route.
        candidate_ids = sorted(set(candidates))
        endpoint_payload = endpoint_payloads.get(candidate_ids[0]) if len(candidate_ids) == 1 else None
        audit_rows.append(
            decide(
                row,
                catalog=catalog,
                aliases=aliases,
                endpoint_payload=endpoint_payload,
                probe=probes.get(source_key(row)),
                catalog_meta=catalog_meta,
            )
        )

    counts: dict[str, int] = {}
    by_provider: dict[str, dict[str, int]] = {}
    for row in audit_rows:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
        provider_counts = by_provider.setdefault(row["provider"], {})
        provider_counts[row["decision"]] = provider_counts.get(row["decision"], 0) + 1

    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "mode": "report-only",
        "source_snapshot": str(args.models_json),
        "source_count": len(rows),
        "catalog_snapshot": str(args.catalog_json) if args.catalog_json else "fetched",
        "catalog_count": len(catalog),
        "catalog_total_count": catalog_snapshot.get("total_count") if isinstance(catalog_snapshot, dict) else None,
        "catalog_scope": catalog_meta,
        "required_parameters": list(REQUIRED_PARAMETERS),
        "decisions": counts,
        "by_provider": by_provider,
        "rows": audit_rows,
    }
    write_json(args.output, report)
    print(json.dumps({"source_count": len(rows), "catalog_count": len(catalog), "decisions": counts}, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
