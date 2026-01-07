"""HTTP output module for posting benchmark results to remote ingest API."""

import logging
import os
from typing import Dict

import httpx

logger = logging.getLogger(__name__)


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
