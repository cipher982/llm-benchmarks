from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pymongo.collection import Collection


def _append_unique_sample(samples: list[str], value: str, *, limit: int = 3) -> list[str]:
    if not value:
        return samples[:limit]
    if value in samples:
        return samples[:limit]
    samples = [value, *samples]
    return samples[:limit]


def upsert_error_rollup(
    *,
    collection: Collection,
    fingerprint: str,
    provider: str,
    model_name: str,
    stage: str,
    error_kind: str,
    normalized_message: str,
    now: Optional[datetime] = None,
) -> None:
    now = now or datetime.now(timezone.utc)

    existing = collection.find_one({"fingerprint": fingerprint}, {"sample_messages": 1})
    existing_samples = []
    if isinstance(existing, dict):
        existing_samples = existing.get("sample_messages") or []

    samples = _append_unique_sample(list(existing_samples), normalized_message, limit=3)

    collection.update_one(
        {"fingerprint": fingerprint},
        {
            "$setOnInsert": {
                "fingerprint": fingerprint,
                "provider": provider,
                "model_name": model_name,
                "stage": stage,
                "error_kind": error_kind,
                "first_seen": now,
            },
            "$set": {
                "provider": provider,
                "model_name": model_name,
                "stage": stage,
                "error_kind": error_kind,
                "last_seen": now,
                "sample_messages": samples,
            },
            "$inc": {"count": 1},
        },
        upsert=True,
    )

