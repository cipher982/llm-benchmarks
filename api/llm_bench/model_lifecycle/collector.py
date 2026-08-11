from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from enum import Enum
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

from pymongo import MongoClient

UTC = timezone.utc

# Mirrors scheduler.mongo.PUBLISHED_PROFILE_ID; this module deliberately
# imports nothing from the scheduler so it stays usable standalone.
_PUBLISHED_PROFILE_ID = "cloud-default-v1"


def _ensure_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _within(window: timedelta, now: datetime, ts: Optional[datetime]) -> bool:
    if ts is None:
        return False
    # Row timestamps are normalized to aware UTC before they reach here, so
    # `now` must be too or a naive caller value raises on subtraction.
    return _ensure_utc(now) - _ensure_utc(ts) <= window


class CatalogState(str, Enum):
    PRESENT = "present"
    MISSING = "missing"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ModelMetadata:
    provider: str
    model_id: str
    display_name: str
    enabled: bool = True
    deprecated: bool = False
    deprecation_date: Optional[datetime] = None
    successor_model: Optional[str] = None
    deprecation_reason: Optional[str] = None


@dataclass(slots=True)
class SuccessMetrics:
    last_success: Optional[datetime] = None
    successes_7d: int = 0
    successes_30d: int = 0
    successes_120d: int = 0

    def age_days(self, now: datetime) -> Optional[float]:
        if self.last_success is None:
            return None
        delta = now - self.last_success
        return delta.total_seconds() / 86400


@dataclass(slots=True)
class ErrorMessage:
    timestamp: Optional[datetime]
    message: str
    kind: str


@dataclass(slots=True)
class ErrorMetrics:
    last_error: Optional[datetime] = None
    errors_7d: int = 0
    errors_30d: int = 0
    hard_failures_7d: int = 0
    hard_failures_30d: int = 0
    recent_messages: List[ErrorMessage] = field(default_factory=list)

    def age_days(self, now: datetime) -> Optional[float]:
        if self.last_error is None:
            return None
        delta = now - self.last_error
        return delta.total_seconds() / 86400


@dataclass(slots=True)
class LifecycleSnapshot:
    provider: str
    model_id: str
    metadata: ModelMetadata
    successes: SuccessMetrics
    errors: ErrorMetrics
    catalog_state: CatalogState = CatalogState.UNKNOWN
    transport_provider: str = "direct"

    @property
    def display_name(self) -> str:
        return self.metadata.display_name


KEYWORDS_HARD = (
    "not found",
    "deprecated",
    "unsupported",
    "invalid",
    "identifier",
    "disabled",
    "does not exist",
    "gone",
)
KEYWORDS_AUTH = (
    "authentication",
    "api key",
    "auth token",
    "unauthorized",
    "forbidden",
    "credentials",
)
KEYWORDS_RATE = (
    "rate limit",
    "quota",
    "exhausted",
    "429",
)


def _categorize_message(message: str) -> str:
    lower = message.lower()
    if any(keyword in lower for keyword in KEYWORDS_HARD):
        return "hard"
    if any(keyword in lower for keyword in KEYWORDS_AUTH):
        return "auth"
    if any(keyword in lower for keyword in KEYWORDS_RATE):
        return "rate"
    return "other"


def collect_lifecycle_snapshots(
    *,
    provider_filter: Optional[Iterable[str]] = None,
    now: Optional[datetime] = None,
    client: Optional[MongoClient] = None,
) -> List[LifecycleSnapshot]:
    """Collect aggregated lifecycle signals for every provider/model pair."""

    now = now or datetime.now(UTC)

    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI must be set")

    db_name = os.getenv("MONGODB_DB", "llm-bench")
    metrics_coll_name = os.getenv("MONGODB_COLLECTION_CLOUD", "metrics_cloud_v2")
    errors_coll_name = os.getenv("MONGODB_COLLECTION_ERRORS", "errors_cloud")
    models_coll_name = os.getenv("MONGODB_COLLECTION_MODELS", "models")

    providers: Optional[Sequence[str]] = None
    if provider_filter is not None:
        providers = sorted({p for p in provider_filter})

    close_client = False
    if client is None:
        client = MongoClient(uri)
        close_client = True

    try:
        db = client[db_name]
        metadata_map = _load_model_metadata(db[models_coll_name], providers)
        success_map = _load_success_metrics(db[metrics_coll_name], providers, now)
        error_map = _load_error_metrics(db[errors_coll_name], providers, now)

        keys = (
            {(provider, model_id, "direct") for provider, model_id in metadata_map.keys()}
            | set(success_map.keys())
            | set(error_map.keys())
        )
        snapshots: List[LifecycleSnapshot] = []

        for provider, model_id, transport_provider in sorted(keys):
            metadata = metadata_map.get(
                (provider, model_id),
                ModelMetadata(provider=provider, model_id=model_id, display_name=model_id),
            )
            successes = success_map.get((provider, model_id, transport_provider), SuccessMetrics())
            errors = error_map.get((provider, model_id, transport_provider), ErrorMetrics())

            snapshots.append(
                LifecycleSnapshot(
                    provider=provider,
                    model_id=model_id,
                    metadata=metadata,
                    successes=successes,
                    errors=errors,
                    transport_provider=transport_provider,
                )
            )

        return snapshots
    finally:
        if close_client:
            client.close()


def _load_model_metadata(collection, providers: Optional[Sequence[str]]) -> Dict[Tuple[str, str], ModelMetadata]:
    query: Dict[str, object] = {}
    if providers:
        query["provider"] = {"$in": list(providers)}

    projection = {
        "provider": 1,
        "model_id": 1,
        "display_name": 1,
        "enabled": 1,
        "deprecated": 1,
        "deprecation_date": 1,
        "successor_model": 1,
        "deprecation_reason": 1,
        "_id": 0,
    }

    metadata: Dict[Tuple[str, str], ModelMetadata] = {}
    for doc in collection.find(query, projection):
        provider = doc.get("provider")
        model_id = doc.get("model_id")
        if not provider or not model_id:
            continue

        display_name = doc.get("display_name") or model_id
        enabled = bool(doc.get("enabled", True))
        deprecated = bool(doc.get("deprecated", False))
        deprecation_date_raw = doc.get("deprecation_date")
        successor_model = doc.get("successor_model")
        deprecation_reason = doc.get("deprecation_reason")

        deprecation_date: Optional[datetime] = None
        if isinstance(deprecation_date_raw, datetime):
            deprecation_date = _ensure_utc(deprecation_date_raw)
        elif isinstance(deprecation_date_raw, str) and deprecation_date_raw:
            try:
                deprecation_date = datetime.fromisoformat(deprecation_date_raw)
                deprecation_date = _ensure_utc(deprecation_date)
            except ValueError:
                deprecation_date = None

        metadata[(provider, model_id)] = ModelMetadata(
            provider=provider,
            model_id=model_id,
            display_name=display_name,
            enabled=enabled,
            deprecated=deprecated,
            deprecation_date=deprecation_date,
            successor_model=successor_model,
            deprecation_reason=deprecation_reason,
        )

    return metadata


def _load_success_metrics(
    collection,
    providers: Optional[Sequence[str]],
    now: datetime,
) -> Dict[Tuple[str, str, str], SuccessMetrics]:
    seven_ago = now - timedelta(days=7)
    thirty_ago = now - timedelta(days=30)
    one_twenty_ago = now - timedelta(days=120)

    if providers:
        match_stage: Dict[str, object] = {
            "provider": {"$in": list(providers)},
            "model_name": {"$exists": True},
        }
    else:
        match_stage = {
            "provider": {"$exists": True},
            "model_name": {"$exists": True},
        }
    # Lifecycle verdicts are about the published series. A model succeeding
    # only under cloud-long-v1 must not look alive here while its 64-token
    # series is dead. Pre-profile rows lack the field and count as published.
    match_stage["$or"] = [
        {"benchmark_profile_id": {"$exists": False}},
        {"benchmark_profile_id": _PUBLISHED_PROFILE_ID},
    ]

    pipeline = [
        {"$match": match_stage},
        {
            "$addFields": {
                "_ts": {"$ifNull": ["$gen_ts", "$run_ts"]},
            }
        },
        {"$match": {"_ts": {"$ne": None}}},
        {
            "$project": {
                "provider": 1,
                "model_name": 1,
                "transport_provider": {"$ifNull": ["$transport_provider", "direct"]},
                "_ts": 1,
                "success_7d": {"$cond": [{"$gte": ["$_ts", seven_ago]}, 1, 0]},
                "success_30d": {"$cond": [{"$gte": ["$_ts", thirty_ago]}, 1, 0]},
                "success_120d": {"$cond": [{"$gte": ["$_ts", one_twenty_ago]}, 1, 0]},
            }
        },
        {
            "$group": {
                "_id": {
                    "provider": "$provider",
                    "model": "$model_name",
                    "transport": "$transport_provider",
                },
                "last_success": {"$max": "$_ts"},
                "successes_7d": {"$sum": "$success_7d"},
                "successes_30d": {"$sum": "$success_30d"},
                "successes_120d": {"$sum": "$success_120d"},
            }
        },
    ]

    metrics: Dict[Tuple[str, str, str], SuccessMetrics] = {}
    try:
        cursor = collection.aggregate(pipeline)
    except Exception:
        return metrics

    for doc in cursor:
        key_doc = doc.get("_id", {})
        provider = key_doc.get("provider")
        model_id = key_doc.get("model")
        transport_provider = key_doc.get("transport") or "direct"
        if not provider or not model_id:
            continue

        metrics[(provider, model_id, transport_provider)] = SuccessMetrics(
            last_success=_ensure_utc(doc.get("last_success")),
            successes_7d=int(doc.get("successes_7d", 0)),
            successes_30d=int(doc.get("successes_30d", 0)),
            successes_120d=int(doc.get("successes_120d", 0)),
        )

    return metrics


def _load_error_metrics(
    collection,
    providers: Optional[Sequence[str]],
    now: datetime,
) -> Dict[Tuple[str, str, str], ErrorMetrics]:
    seven_ago = now - timedelta(days=7)
    thirty_ago = now - timedelta(days=30)

    if providers:
        match_stage: Dict[str, object] = {
            "provider": {"$in": list(providers)},
            "model_name": {"$exists": True},
        }
    else:
        match_stage = {
            "provider": {"$exists": True},
            "model_name": {"$exists": True},
        }
    # Same boundary as the success side: a model failing only long runs is not
    # failing the published series, so its long errors must not push it toward
    # a deprecation verdict.
    match_stage["$or"] = [
        {"profile_id": {"$exists": False}},
        {"profile_id": _PUBLISHED_PROFILE_ID},
    ]

    pipeline = [
        {"$match": match_stage},
        {
            "$addFields": {
                "_ts": {"$ifNull": ["$ts", {"$ifNull": ["$created_at", {"$ifNull": ["$timestamp", "$ts"]}]}]},
                "_kind": {"$ifNull": ["$error_kind", ""]},
                "_message": {
                    "$ifNull": [
                        "$normalized_message",
                        {"$ifNull": ["$message", {"$ifNull": ["$error", {"$ifNull": ["$traceback", ""]}]}]},
                    ]
                },
            }
        },
        {"$match": {"_ts": {"$ne": None}}},
        {
            "$project": {
                "provider": 1,
                "model_name": 1,
                "transport_provider": {"$ifNull": ["$transport_provider", "direct"]},
                "_ts": 1,
                "_message": 1,
                "_kind": 1,
                "error_7d": {"$cond": [{"$gte": ["$_ts", seven_ago]}, 1, 0]},
                "error_30d": {"$cond": [{"$gte": ["$_ts", thirty_ago]}, 1, 0]},
            }
        },
        {"$sort": {"provider": 1, "model_name": 1, "_ts": -1}},
        {
            "$group": {
                "_id": {
                    "provider": "$provider",
                    "model": "$model_name",
                    "transport": "$transport_provider",
                },
                "last_error": {"$max": "$_ts"},
                "errors_7d": {"$sum": "$error_7d"},
                "errors_30d": {"$sum": "$error_30d"},
                "messages": {"$push": {"ts": "$_ts", "message": "$_message", "kind": "$_kind"}},
            }
        },
        {
            "$project": {
                "last_error": 1,
                "errors_7d": 1,
                "errors_30d": 1,
                "messages": {"$slice": ["$messages", 20]},
            }
        },
    ]

    metrics: Dict[Tuple[str, str, str], ErrorMetrics] = {}
    try:
        cursor = collection.aggregate(pipeline)
    except Exception:
        return metrics

    for doc in cursor:
        key_doc = doc.get("_id", {})
        provider = key_doc.get("provider")
        model_id = key_doc.get("model")
        transport_provider = key_doc.get("transport") or "direct"
        if not provider or not model_id:
            continue

        last_error = _ensure_utc(doc.get("last_error"))
        errors_7d = int(doc.get("errors_7d", 0))
        errors_30d = int(doc.get("errors_30d", 0))

        recent_messages: List[ErrorMessage] = []
        hard_failures_7d = 0
        hard_failures_30d = 0

        for idx, entry in enumerate(doc.get("messages", [])):
            ts = _ensure_utc(entry.get("ts")) if isinstance(entry, dict) else None
            message_raw = ""
            message_kind = ""
            if isinstance(entry, dict):
                message_raw = entry.get("message") or ""
                message_kind = entry.get("kind") or ""
            elif isinstance(entry, str):
                message_raw = entry
            else:
                message_raw = ""

            # Prefer structured error_kind from the runner; fall back to keyword categorization.
            normalized_kind = str(message_kind).strip().lower()
            if normalized_kind:
                kind = normalized_kind
            else:
                kind = _categorize_message(message_raw)

            # Per v2 spec: only hard_model is treated as "hard failure" for deprecation signals.
            is_hard_model = kind in ("hard_model", "hard")  # "hard" is legacy fallback
            if _within(timedelta(days=7), now, ts) and is_hard_model:
                hard_failures_7d += 1
            if _within(timedelta(days=30), now, ts) and is_hard_model:
                hard_failures_30d += 1

            if idx < 10:
                recent_messages.append(ErrorMessage(timestamp=ts, message=message_raw, kind=kind))

        metrics[(provider, model_id, transport_provider)] = ErrorMetrics(
            last_error=last_error,
            errors_7d=errors_7d,
            errors_30d=errors_30d,
            hard_failures_7d=hard_failures_7d,
            hard_failures_30d=hard_failures_30d,
            recent_messages=recent_messages,
        )

    return metrics
