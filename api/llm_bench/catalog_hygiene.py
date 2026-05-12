from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from typing import Iterable

CHECKPOINT_SUFFIX_RE = re.compile(
    r"(?i)(?:[-_](?:20\d{2}[-_]?\d{2}[-_]?\d{2}|20\d{6})(?:[-_]v\d+(?::\d+)?)?|[-_]20\d{6}(?:[-_].*)?)$"
)
BEDROCK_PROVIDER_PREFIX_RE = re.compile(r"^(?:us\.)?(?:anthropic|meta|amazon|mistral|cohere)\.")
BEDROCK_VERSION_SUFFIX_RE = re.compile(r"(?i)(?:[-_]v\d+(?::\d+)?)$")


@dataclass(frozen=True)
class CatalogIssue:
    code: str
    severity: str
    provider: str
    model_id: str
    message: str
    field: str | None = None
    value: str | None = None
    normalized_identity: str | None = None


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _strip_checkpoint_suffix(value: str) -> str:
    current = value
    while True:
        stripped = CHECKPOINT_SUFFIX_RE.sub("", current)
        if stripped == current:
            return stripped
        current = stripped


def normalize_display_identity(provider: str, model_id: str, display_name: str | None = None) -> str:
    """Return a stable identity for duplicate alias checks.

    Provider IDs may contain routing prefixes, provider date checkpoints, and
    Bedrock API version suffixes. Those should not split the display identity.
    Real model versions such as "opus 4.5" or "llama3.3" are preserved.
    """
    source = display_name or model_id
    normalized = source.strip()
    if provider == "bedrock":
        normalized = BEDROCK_PROVIDER_PREFIX_RE.sub("", normalized)
        normalized = BEDROCK_VERSION_SUFFIX_RE.sub("", normalized)
    normalized = _strip_checkpoint_suffix(normalized)
    return _slug(normalized)


def has_checkpoint_display_suffix(value: str) -> bool:
    """Return true when a display/canonical value ends with a checkpoint date."""
    return bool(CHECKPOINT_SUFFIX_RE.search(_slug(value)))


def _enabled_active(doc: dict[str, Any]) -> bool:
    return bool(doc.get("enabled", True)) and not bool(doc.get("deprecated", False))


def analyze_catalog_rows(rows: Iterable[dict[str, Any]], *, provider: str = "bedrock") -> list[CatalogIssue]:
    issues: list[CatalogIssue] = []
    active_rows: list[dict[str, Any]] = []

    for row in rows:
        row_provider = row.get("provider")
        if row_provider != provider:
            continue
        model_id = str(row.get("model_id") or "")
        if not model_id:
            continue
        if _enabled_active(row):
            active_rows.append(row)

        for field in ("display_name", "canonical_id", "model_canonical"):
            value = row.get(field)
            if isinstance(value, str) and has_checkpoint_display_suffix(value):
                issues.append(
                    CatalogIssue(
                        code="checkpoint_suffix",
                        severity="error",
                        provider=provider,
                        model_id=model_id,
                        field=field,
                        value=value,
                        message=f"{field} ends with a provider checkpoint date; keep dates in model_id only",
                        normalized_identity=normalize_display_identity(provider, model_id, value),
                    )
                )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in active_rows:
        model_id = str(row.get("model_id") or "")
        identity = normalize_display_identity(provider, model_id, row.get("display_name"))
        grouped.setdefault(identity, []).append(row)

    for identity, group in sorted(grouped.items()):
        model_ids = sorted(str(row.get("model_id")) for row in group)
        if len(set(model_ids)) <= 1:
            continue
        for row in group:
            issues.append(
                CatalogIssue(
                    code="duplicate_enabled_alias",
                    severity="error",
                    provider=provider,
                    model_id=str(row.get("model_id")),
                    message=(
                        "enabled Bedrock alias duplicates normalized display identity "
                        f"{identity}: {', '.join(model_ids)}"
                    ),
                    normalized_identity=identity,
                )
            )

    return issues


def issue_to_dict(issue: CatalogIssue) -> dict[str, Any]:
    return {
        "code": issue.code,
        "severity": issue.severity,
        "provider": issue.provider,
        "model_id": issue.model_id,
        "field": issue.field,
        "value": issue.value,
        "normalized_identity": issue.normalized_identity,
        "message": issue.message,
    }
