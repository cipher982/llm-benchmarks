from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from pymongo.collection import Collection


UTC = timezone.utc


@dataclass(frozen=True, slots=True)
class ProviderPause:
    provider: str
    paused_until: datetime
    pause_reason: str
    pause_note: str = ""


def get_pause(collection: Collection, provider: str, *, now: Optional[datetime] = None) -> Optional[ProviderPause]:
    now = now or datetime.now(UTC)
    doc = collection.find_one({"provider": provider}, {"paused_until": 1, "pause_reason": 1, "pause_note": 1})
    if not isinstance(doc, dict):
        return None
    paused_until = doc.get("paused_until")
    if not isinstance(paused_until, datetime):
        return None
    if paused_until <= now:
        return None
    return ProviderPause(
        provider=provider,
        paused_until=paused_until.astimezone(UTC) if paused_until.tzinfo else paused_until.replace(tzinfo=UTC),
        pause_reason=str(doc.get("pause_reason") or "unknown"),
        pause_note=str(doc.get("pause_note") or ""),
    )


def pause_provider(
    *,
    collection: Collection,
    provider: str,
    reason: str,
    duration: timedelta,
    note: str = "",
    now: Optional[datetime] = None,
) -> ProviderPause:
    now = now or datetime.now(UTC)
    paused_until = now + duration

    # Only extend pauses; never shorten.
    existing = collection.find_one({"provider": provider}, {"paused_until": 1})
    if isinstance(existing, dict) and isinstance(existing.get("paused_until"), datetime):
        existing_until = existing["paused_until"]
        if existing_until.tzinfo is None:
            existing_until = existing_until.replace(tzinfo=UTC)
        if existing_until > paused_until:
            paused_until = existing_until

    collection.update_one(
        {"provider": provider},
        {
            "$setOnInsert": {"provider": provider, "created_at": now},
            "$set": {
                "paused_until": paused_until,
                "pause_reason": reason,
                "pause_note": note[:500],
                "updated_at": now,
            },
        },
        upsert=True,
    )

    return ProviderPause(provider=provider, paused_until=paused_until, pause_reason=reason, pause_note=note)

