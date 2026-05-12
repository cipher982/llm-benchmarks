from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_bench.catalog_hygiene import normalize_display_identity

from .matcher import ModelMatch


@dataclass(frozen=True)
class PromotionPlan:
    provider: str
    model_id: str
    display_name: str
    warnings: list[str]
    insert_doc: dict[str, Any]


def _restore_version_dots(identity: str) -> str:
    return (
        identity.replace("claude-3-5-", "claude-3.5-")
        .replace("claude-3-7-", "claude-3.7-")
        .replace("claude-opus-4-5", "claude-opus-4.5")
        .replace("claude-opus-4-6", "claude-opus-4.6")
        .replace("claude-opus-4-7", "claude-opus-4.7")
        .replace("claude-sonnet-4-5", "claude-sonnet-4.5")
        .replace("claude-haiku-4-5", "claude-haiku-4.5")
    )


def display_name_for_match(match: ModelMatch) -> str:
    if match.provider == "bedrock":
        return _restore_version_dots(normalize_display_identity(match.provider, match.model_id))
    return match.model_id


def build_promotion_plan(match: ModelMatch, existing_models: list[dict[str, Any]]) -> PromotionPlan:
    display_name = display_name_for_match(match)
    identity = normalize_display_identity(match.provider, match.model_id, display_name)
    warnings: list[str] = []

    for row in existing_models:
        if row.get("provider") != match.provider:
            continue
        existing_id = row.get("model_id")
        if not existing_id:
            continue
        existing_identity = normalize_display_identity(
            match.provider,
            str(existing_id),
            row.get("display_name"),
        )
        if existing_identity == identity:
            state = "enabled" if row.get("enabled") and not row.get("deprecated") else "cataloged"
            warnings.append(f"{state} alias already exists for {identity}: {existing_id}")

    insert_doc = {
        "provider": match.provider,
        "model_id": match.model_id,
        "display_name": display_name,
        "enabled": False,
        "deprecated": False,
        "promotion_status": "candidate",
        "promotion_source": "openrouter",
        "promotion_openrouter_id": match.openrouter_id,
        "promotion_confidence": match.confidence,
        "promotion_reasoning": match.reasoning,
    }

    return PromotionPlan(
        provider=match.provider,
        model_id=match.model_id,
        display_name=display_name,
        warnings=warnings,
        insert_doc=insert_doc,
    )
