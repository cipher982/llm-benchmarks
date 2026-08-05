"""HTTP output module for posting benchmark results to remote ingest API."""

import logging
import os
from typing import Dict

import httpx

from llm_bench.ops.error_taxonomy import classify_error

logger = logging.getLogger(__name__)


def _error_url() -> str:
    """Sibling of INGEST_API_URL. Derived rather than separately configured so a
    runner cannot end up posting metrics while silently dropping failures."""
    override = os.getenv("INGEST_ERROR_URL")
    if override:
        return override
    base = (os.getenv("INGEST_API_URL") or "").rstrip("/")
    return f"{base}/error" if base else ""


def log_http_error(config, *, message: str, stage: str = "generate", exc_type: str = "") -> bool:
    """POST a failed attempt to the remote ingest API.

    The Bedrock runner had no way to say a benchmark failed. Every failure path
    wrote to a local log file on an EC2 instance and returned False, so three
    models failed on roughly 657 consecutive cycles over fourteen days while
    `errors_cloud` stayed empty. Nothing was wrong with the models that anyone
    could see; they simply had no recent data, and only a coverage invariant
    over the desired set ever noticed.

    Failing to report a failure is itself swallowed — deliberately. The caller
    already has a failure to return, and raising here would replace a specific
    cause with a reporting error.
    """
    url = _error_url()
    api_key = os.getenv("INGEST_API_KEY")
    if not url or not api_key:
        logger.error("Cannot report failure: INGEST_API_URL/INGEST_API_KEY not set")
        return False

    payload = {
        "provider": config.provider,
        "model_name": config.model_name,
        "ts": getattr(config, "run_ts", None),
        "stage": stage,
        "message": str(message)[:4000],
        "exc_type": exc_type,
        # Classified here because this process holds the original exception and
        # shares a repository with the taxonomy. The bridge only stores it.
        "error_kind": classify_error(message=str(message), exc_type=exc_type).kind.value,
    }

    try:
        response = httpx.post(
            url, json=payload, headers={"X-API-Key": api_key, "Content-Type": "application/json"}, timeout=10.0
        )
        response.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to report benchmark failure for {config.provider}:{config.model_name}: {exc}")
        return False


def log_http(config, metrics: Dict) -> bool:
    """
    POST benchmark result to remote ingest API.

    Args:
        config: CloudConfig instance with provider, model_name, run_ts, temperature, misc
        metrics: Dict containing benchmark metrics (gen_ts, output_tokens, generate_time, etc.)

    Returns:
        True on success, False on failure

    Environment variables:
        INGEST_API_URL: URL of the ingest API endpoint
        INGEST_API_KEY: API key for authentication
    """
    ingest_url = os.getenv("INGEST_API_URL")
    api_key = os.getenv("INGEST_API_KEY")

    if not ingest_url:
        logger.error("INGEST_API_URL not set in environment")
        return False

    if not api_key:
        logger.error("INGEST_API_KEY not set in environment")
        return False

    # Build the payload combining config and metrics
    payload = {
        **config.to_dict(),
        **metrics,
    }

    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    try:
        response = httpx.post(ingest_url, json=payload, headers=headers, timeout=10.0)
        response.raise_for_status()
        try:
            response_payload = response.json()
        except ValueError:
            response_payload = {}
        if response_payload.get("status") == "rejected":
            logger.warning(
                "Ingest rejected benchmark result for %s:%s - %s",
                config.provider,
                config.model_name,
                response_payload.get("reason", "no reason provided"),
            )
            return False
        logger.info(f"Successfully posted benchmark result for {config.provider}:{config.model_name}")
        return True
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error posting to ingest API: {e.response.status_code} - {e.response.text}")
        return False
    except httpx.RequestError as e:
        logger.error(f"Request error posting to ingest API: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error posting to ingest API: {e}")
        return False
