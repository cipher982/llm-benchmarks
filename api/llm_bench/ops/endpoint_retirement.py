"""Retire endpoints that have proved they cannot produce a measurement.

Some catalogued endpoints are not benchmarkable at all. `gemini-3-pro-image`
returns images, so every run ends with empty visible text; others answer 400
because they do not accept the request shape the profile sends. Left enabled
they are scheduled forever, spend money on every pass, and sit permanently in
the starvation check as violations nobody can act on.

Two rules govern this, both learned the hard way in this codebase.

**Never filter by name.** Two passes of pattern matching over model ids missed
`veo`, `kling`, `vidu`, `ideogram` and `parakeet`. The only reliable signal is
a real call, so retirement here is driven entirely by what the runner observed:
repeated attempts, no success ever, and a verdict from the non-retryable
capability class.

**Every terminal state needs a way back.** `max_attempts` with no resurrection
path decayed coverage to 11.7% with no single incident. A retirement is
therefore stamped with the measurement protocol version that produced it, and
`restore_stale_protocol_retirements` returns any endpoint whose verdict was
reached under an older protocol. Raise the token budget, change the profile,
and every endpoint retired under the old one is reconsidered automatically —
rather than staying dead because nothing was watching.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pymongo.database import Database

from llm_bench.ops import endpoint_discovery
from llm_bench.scheduler import policies
from llm_bench.scheduler.mongo import health_collection_name
from llm_bench.scheduler.mongo import jobs_collection_name
from llm_bench.scheduler.queue import utcnow

#: Verdicts that describe the endpoint rather than the moment. A timeout or a
#: 429 says nothing about whether the endpoint can be measured; these do.
CAPABILITY_ERROR_KINDS = {"hard_capability", "hard_model"}

#: `unknown` covers both transient noise and permanent incapability, so it only
#: counts when the message itself is a capability statement. Matched on what
#: the provider said, never on the model's name.
#:
#: "visible output text is empty" is deliberately NOT here. It is the same
#: symptom a reasoning model produces when it spends its budget thinking before
#: emitting anything -- a live dry run flagged `deepseek-r1-0528` alongside
#: three genuine image endpoints. That case has its own verdict
#: (`budget_exhausted`, terminal but protocol-dependent), and an endpoint whose
#: emptiness is not yet explained belongs in the classifier's queue rather than
#: in a retirement decided by a regex.
CAPABILITY_MESSAGE_PATTERNS = (
    r"does not support",
    r"no endpoints found",
    r"is not a valid model",
    r"is not supported",
    r"not supported for this model",
    r"no allowed providers",
    r"notfounderror",
    r"not found",
    r"badrequesterror",
    r"bad request",
    r"invalid_request_error",
    r"expected <",
    r"tokens_per_second <= 0",
    r"run measured None",
    r"not the job's endpoint",
)

#: One failure is an incident. Repeated failure with no success ever is a fact
#: about the endpoint.
MIN_ATTEMPTS = 3


def _is_capability_failure(kind: str | None, message: str | None) -> bool:
    if kind in CAPABILITY_ERROR_KINDS:
        return True
    if kind == "pin_unverified":
        return True
    text = message or ""
    if "visible output text is empty" in text.lower():
        return False
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in CAPABILITY_MESSAGE_PATTERNS)


def _verdict(db: Database, doc: dict[str, Any]) -> tuple[str | None, str | None, int]:
    """The endpoint's verdict, read from health first and its job second.

    The two disagree whenever a run failed before completions were credited per
    endpoint, so reading only one of them misses real evidence.
    """
    kind = doc.get("last_error_kind")
    message = doc.get("last_error_message")
    attempts = max(
        int(doc.get("failures_24h") or 0),
        int(doc.get("consecutive_failures") or 0),
    )
    job = db[jobs_collection_name()].find_one(
        {"provider": "openrouter", "model_id": doc.get("model_id"), "endpoint_tag": doc.get("endpoint_tag")},
        {"last_attempt_error_kind": 1, "last_attempt_error_message": 1, "attempt": 1, "status": 1},
    )
    if job:
        job_kind = job.get("last_attempt_error_kind")
        if job_kind in CAPABILITY_ERROR_KINDS:
            kind = job_kind
        else:
            kind = kind or job_kind
        message = message or job.get("last_attempt_error_message")
        attempts = max(attempts, int(job.get("attempt") or 0))
        # A dead letter has exhausted its attempts by definition; the counter
        # stops at max_attempts and understates what was actually tried.
        if job.get("status") == "dead_letter":
            attempts = max(attempts, MIN_ATTEMPTS)
    return kind, message, attempts


def retire_unmeasurable_endpoints(
    db: Database, *, now: datetime | None = None, dry_run: bool = False
) -> list[dict[str, Any]]:
    """Disable enabled endpoints that have never produced a measurement and
    whose verdict says they never will."""
    now = now or utcnow()
    retired: list[dict[str, Any]] = []
    collection = db[endpoint_discovery.endpoints_collection_name()]

    for doc in db[health_collection_name()].find(
        {"provider": "openrouter", "endpoint_tag": {"$ne": None}, "last_success_at": None},
        {
            "model_id": 1,
            "endpoint_tag": 1,
            "last_error_kind": 1,
            "last_error_message": 1,
            "failures_24h": 1,
            "consecutive_failures": 1,
        },
    ):
        kind, message, attempts = _verdict(db, doc)
        if attempts < MIN_ATTEMPTS or not _is_capability_failure(kind, message):
            continue
        entry = {
            "model_id": doc["model_id"],
            "endpoint_tag": doc["endpoint_tag"],
            "error_kind": kind,
            "attempts": attempts,
            "reason": (message or "")[:200],
        }
        retired.append(entry)
        if dry_run:
            continue
        collection.update_one(
            {"model_id": doc["model_id"], "endpoint_tag": doc["endpoint_tag"], "enabled": True},
            {
                "$set": {
                    "enabled": False,
                    "disabled_reason": (
                        f"no measurement in {attempts} attempts; provider said: {(message or kind or '')[:160]}"
                    ),
                    "disabled_at": now,
                    # The way back. A verdict reached under an older protocol is
                    # about that protocol, not about the endpoint.
                    "disabled_protocol_version": policies.MEASUREMENT_PROTOCOL_VERSION,
                }
            },
        )
        db[health_collection_name()].update_one(
            {"provider": "openrouter", "model_id": doc["model_id"], "endpoint_tag": doc["endpoint_tag"]},
            {"$set": {"enabled": False}},
        )
    return retired


def restore_stale_protocol_retirements(
    db: Database, *, now: datetime | None = None, dry_run: bool = False
) -> list[dict[str, Any]]:
    """Re-enable endpoints retired under a superseded measurement protocol.

    Without this, raising the token budget leaves every endpoint that failed
    under the old one permanently disabled — which is exactly how 419 models
    came to be holding a verdict reached against a budget that no longer
    existed.
    """
    now = now or utcnow()
    collection = db[endpoint_discovery.endpoints_collection_name()]
    query = {
        "enabled": False,
        "disabled_protocol_version": {"$lt": policies.MEASUREMENT_PROTOCOL_VERSION},
    }
    restored = [
        {"model_id": doc.get("model_id"), "endpoint_tag": doc.get("endpoint_tag")}
        for doc in collection.find(query, {"model_id": 1, "endpoint_tag": 1})
    ]
    if restored and not dry_run:
        collection.update_many(
            query,
            {
                "$set": {"enabled": True, "restored_at": now, "missing_passes": 0},
                "$unset": {"disabled_reason": "", "disabled_at": "", "disabled_protocol_version": ""},
            },
        )
        for item in restored:
            db[health_collection_name()].update_one(
                {"provider": "openrouter", "model_id": item["model_id"], "endpoint_tag": item["endpoint_tag"]},
                {"$set": {"enabled": True}},
            )
    return restored
