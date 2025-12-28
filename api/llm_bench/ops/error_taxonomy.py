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
    UNKNOWN = "unknown"


_RE_ERR_CODE = re.compile(r"error code:\s*(\d{3})", re.IGNORECASE)
_RE_HTTP_STATUS = re.compile(r"\b(?:http\s*status|status(?:\s*code)?)\s*[:=]\s*(\d{3})\b", re.IGNORECASE)
_RE_REQUEST_ID = re.compile(r"\b(request[_ -]?id|activityid)\b\s*[:=]\s*['\"]?[a-z0-9-]{8,}['\"]?", re.IGNORECASE)


_HARD_CAPABILITY_HINTS = (
    "only supported in v1/responses",
    "only supported in the responses api",
    "not supported in the v1/chat/completions",
    "not a chat model",
    "max_output_tokens",
    # Audio/image-only models (not text chat models)
    "modality contain audio",
    "requires audio",
    "audio input",
    "audio output",
    "image generation",
    "not supported for this model",
)

_HARD_MODEL_HINTS = (
    "model_not_found",
    "does not exist",
    "no endpoints found",
    "not found",
    "unknown model",
    "unsupported model",
    "unsupported model_name",
    "identifier",
)

_AUTH_HINTS = (
    "unauthorized",
    "forbidden",
    "invalid api key",
    "authentication",
    "credentials",
    "api key",
    # AWS/cloud credential issues
    "profilenotfound",
    "config profile",  # AWS profile errors: "config profile ... could not be found"
    "nosuchprofile",
    "access denied",
    "security token",
    "expired token",
)

_BILLING_HINTS = (
    "insufficient credits",
    "overdue invoices",
    "billing",
    "payment required",
    "inference prohibited",
)

_RATE_HINTS = (
    "rate limit",
    "quota",
    "too many requests",
    "throttl",
)

_NETWORK_HINTS = (
    "timed out",
    "timeout",
    "connection error",
    "connection reset",
    "dns",
    "name or service not known",
    "temporarily unavailable",
)


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
    m = _RE_ERR_CODE.search(message) or _RE_HTTP_STATUS.search(message)
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
    raw = message or ""
    http_status = _extract_http_status(raw)
    normalized = normalize_error_message(raw)
    exc_type_norm = (exc_type or "").lower()

    # Prefer explicit status codes when present.
    if http_status in (401, 403):
        return ClassifiedError(kind=ErrorKind.AUTH, normalized_message=normalized, http_status=http_status)
    if http_status == 402:
        return ClassifiedError(kind=ErrorKind.BILLING, normalized_message=normalized, http_status=http_status)
    if http_status == 429:
        return ClassifiedError(kind=ErrorKind.RATE_LIMIT, normalized_message=normalized, http_status=http_status)
    if http_status and 500 <= http_status <= 599:
        return ClassifiedError(kind=ErrorKind.TRANSIENT_PROVIDER, normalized_message=normalized, http_status=http_status)
    if http_status == 404:
        # 404 can be model missing or aggregator routing. If it looks like capability mismatch, classify that instead.
        if any(h in normalized for h in _HARD_CAPABILITY_HINTS):
            return ClassifiedError(kind=ErrorKind.HARD_CAPABILITY, normalized_message=normalized, http_status=http_status)
        return ClassifiedError(kind=ErrorKind.HARD_MODEL, normalized_message=normalized, http_status=http_status)

    # Keyword-based fallback.
    if any(h in normalized for h in _HARD_CAPABILITY_HINTS):
        return ClassifiedError(kind=ErrorKind.HARD_CAPABILITY, normalized_message=normalized, http_status=http_status)
    if any(h in normalized for h in _BILLING_HINTS):
        return ClassifiedError(kind=ErrorKind.BILLING, normalized_message=normalized, http_status=http_status)
    if any(h in normalized for h in _AUTH_HINTS):
        return ClassifiedError(kind=ErrorKind.AUTH, normalized_message=normalized, http_status=http_status)
    if any(h in normalized for h in _RATE_HINTS):
        return ClassifiedError(kind=ErrorKind.RATE_LIMIT, normalized_message=normalized, http_status=http_status)
    if any(h in normalized for h in _NETWORK_HINTS) or "apiconnectionerror" in exc_type_norm:
        return ClassifiedError(kind=ErrorKind.NETWORK, normalized_message=normalized, http_status=http_status)

    # Hard-model-ish hints without an explicit 404.
    if any(h in normalized for h in _HARD_MODEL_HINTS):
        return ClassifiedError(kind=ErrorKind.HARD_MODEL, normalized_message=normalized, http_status=http_status)

    return ClassifiedError(kind=ErrorKind.UNKNOWN, normalized_message=normalized, http_status=http_status)

