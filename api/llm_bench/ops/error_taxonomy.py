from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorKind(str, Enum):
    AUTH = "auth"
    BILLING = "billing"
    RATE_LIMIT = "rate_limit"
    HARD_MODEL = "hard_model"
    HARD_CAPABILITY = "hard_capability"
    TRANSIENT_PROVIDER = "transient_provider"
    NETWORK = "network"
    TIMEOUT = "timeout"
    # The model answered; the 64-token budget was spent on reasoning before it
    # emitted anything visible. Nothing is broken — the profile cannot measure
    # this model. Kept apart from UNKNOWN because they call for opposite
    # responses: an unknown error wants investigation, this wants a decision
    # about the benchmark profile, and mixing them buried the second inside
    # 279 dead letters that also included models the provider had deleted.
    BUDGET_EXHAUSTED = "budget_exhausted"
    UNKNOWN = "unknown"


_RE_ERR_CODE = re.compile(r"error code:\s*(\d{3})", re.IGNORECASE)
_RE_HTTP_STATUS = re.compile(r"\b(?:http\s*status|status(?:\s*code)?)\s*[:=]\s*(\d{3})\b", re.IGNORECASE)
# gRPC status names followed by HTTP-equivalent code (e.g. "NotFound: 404", "PermissionDenied: 403")
_RE_GRPC_STATUS = re.compile(
    r"\b(?:NotFound|PermissionDenied|Unauthenticated|ResourceExhausted|Internal|Unavailable|DeadlineExceeded)\s*:\s*(\d{3})\b",
    re.IGNORECASE,
)
_RE_REQUEST_ID = re.compile(r"\b(request[_ -]?id|activityid)\b\s*[:=]\s*['\"]?[a-z0-9-]{8,}['\"]?", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ClassifiedError:
    kind: ErrorKind
    normalized_message: str
    http_status: Optional[int] = None
    provider_error_code: Optional[str] = None

    def fingerprint(self, *, provider: str, model: str, stage: str) -> str:
        base = f"{provider}\n{model}\n{stage}\n{self.kind.value}\n{self.normalized_message}".encode("utf-8")
        return hashlib.sha256(base).hexdigest()


def _extract_http_status(message: str) -> Optional[int]:
    m = _RE_ERR_CODE.search(message) or _RE_HTTP_STATUS.search(message) or _RE_GRPC_STATUS.search(message)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def normalize_error_message(message: str) -> str:
    msg = (message or "").strip().lower()
    msg = _RE_REQUEST_ID.sub("request_id=<redacted>", msg)
    msg = re.sub(r"\s+", " ", msg)
    return msg[:2000]


def classify_error(*, message: str, exc_type: str = "") -> ClassifiedError:
    """Classify explicit provider errors that determine retry behavior.

    HTTP status is the default signal. Provider messages override it only when
    the status is overloaded, as with OpenRouter returning 403 for an exhausted
    monthly credit limit.
    """
    raw = message or ""
    http_status = _extract_http_status(raw)
    normalized = normalize_error_message(raw)

    if "key limit exceeded" in normalized:
        return ClassifiedError(kind=ErrorKind.BILLING, normalized_message=normalized, http_status=http_status)

    # Only use explicit HTTP status codes for classification
    if http_status in (401, 403):
        return ClassifiedError(kind=ErrorKind.AUTH, normalized_message=normalized, http_status=http_status)
    if http_status == 402:
        return ClassifiedError(kind=ErrorKind.BILLING, normalized_message=normalized, http_status=http_status)
    if http_status == 429:
        return ClassifiedError(kind=ErrorKind.RATE_LIMIT, normalized_message=normalized, http_status=http_status)
    if http_status and 500 <= http_status <= 599:
        return ClassifiedError(
            kind=ErrorKind.TRANSIENT_PROVIDER, normalized_message=normalized, http_status=http_status
        )
    if http_status == 404:
        return ClassifiedError(kind=ErrorKind.HARD_MODEL, normalized_message=normalized, http_status=http_status)
    if exc_type == "TimeoutError" or "timed out" in normalized or "timeout" in normalized:
        return ClassifiedError(kind=ErrorKind.TIMEOUT, normalized_message=normalized, http_status=http_status)
    if any(marker in normalized for marker in ("overloaded", "model busy", "retry later", "temporarily unavailable")):
        return ClassifiedError(
            kind=ErrorKind.TRANSIENT_PROVIDER,
            normalized_message=normalized,
            http_status=http_status,
        )

    # Together "model not available as serverless" — provider removed serverless access
    if "'code': 'model_not_available'" in raw or '"code": "model_not_available"' in raw:
        return ClassifiedError(kind=ErrorKind.HARD_MODEL, normalized_message=normalized, http_status=http_status)

    # Bedrock retires model versions and answers ResourceNotFoundException with
    # this phrase and no HTTP status to key on. As terminal as a 404 — the
    # version is gone and no retry brings it back.
    if "reached the end of its life" in normalized:
        return ClassifiedError(kind=ErrorKind.HARD_MODEL, normalized_message=normalized, http_status=http_status)

    # Together and Fireworks answer 400 for a model that exists but is only
    # reachable through a dedicated endpoint. The account cannot call it, which
    # is the same practical outcome as a 404 — and leaving it UNKNOWN means it
    # is retried on every pass forever.
    if "dedicated endpoint" in normalized:
        return ClassifiedError(kind=ErrorKind.HARD_MODEL, normalized_message=normalized, http_status=http_status)

    # Our own validator's message, not a provider's. No LLM is needed to
    # recognize a string this repository writes.
    if "token budget was exhausted" in normalized:
        return ClassifiedError(kind=ErrorKind.BUDGET_EXHAUSTED, normalized_message=normalized, http_status=http_status)

    # Everything else is UNKNOWN - LLM will classify later
    return ClassifiedError(kind=ErrorKind.UNKNOWN, normalized_message=normalized, http_status=http_status)
