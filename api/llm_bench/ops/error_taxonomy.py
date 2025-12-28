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
    """
    Initial error classification based ONLY on HTTP status codes.

    Returns UNKNOWN for all errors without clear HTTP status signals.
    LLM-based classification happens later via llm_error_classifier.py.
    """
    raw = message or ""
    http_status = _extract_http_status(raw)
    normalized = normalize_error_message(raw)

    # Only use explicit HTTP status codes for classification
    if http_status in (401, 403):
        return ClassifiedError(kind=ErrorKind.AUTH, normalized_message=normalized, http_status=http_status)
    if http_status == 402:
        return ClassifiedError(kind=ErrorKind.BILLING, normalized_message=normalized, http_status=http_status)
    if http_status == 429:
        return ClassifiedError(kind=ErrorKind.RATE_LIMIT, normalized_message=normalized, http_status=http_status)
    if http_status and 500 <= http_status <= 599:
        return ClassifiedError(kind=ErrorKind.TRANSIENT_PROVIDER, normalized_message=normalized, http_status=http_status)
    if http_status == 404:
        return ClassifiedError(kind=ErrorKind.HARD_MODEL, normalized_message=normalized, http_status=http_status)

    # Everything else is UNKNOWN - LLM will classify later
    return ClassifiedError(kind=ErrorKind.UNKNOWN, normalized_message=normalized, http_status=http_status)

