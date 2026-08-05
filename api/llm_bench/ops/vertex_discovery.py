"""Catalogue discovery for Vertex, which the Sauron job cannot reach.

Every other provider is discovered by a Sauron job that reads a `/models`
endpoint with a bearer token. Vertex needs a Google service account, and Sauron
has no GCP credentials — only SES. The benchmark daemon does have them, because
it calls Vertex to benchmark it, so discovery for this one provider runs here.

That is a credential boundary, not a design preference. Moving GCP credentials
into Sauron would unify the code and widen what a single compromised container
can reach; running one provider's discovery where its credentials already exist
does not.

Bedrock is the remaining uncovered provider and cannot be done the same way: it
is benchmarked from an EC2 instance under an IAM role, and clifford holds no AWS
credentials for it at all. Closing that one needs a decision about where its
discovery runs, not more code here.

Writes the same `bench_discovery_runs` ledger the Sauron job does, because
`discovery_completed_recently` reads that ledger and does not care which process
produced the row — only that a completed read happened.
"""

from __future__ import annotations

import os
from datetime import datetime
from datetime import timezone
from typing import Any

import httpx

from llm_bench.scheduler.mongo import collection_name

PROVIDER = "vertex"
SOURCE_VERSION = 1

# Google's published model garden for first-party models. Regional host, because
# the global endpoint does not serve this listing.
DEFAULT_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
PUBLISHER = os.getenv("VERTEX_PUBLISHER", "google")
PAGE_SIZE = 100
# A hard stop, so a malformed nextPageToken cannot loop forever on a paid API.
MAX_PAGES = 20


def discovery_runs_collection_name() -> str:
    return collection_name("MONGODB_COLLECTION_DISCOVERY_RUNS", "bench_discovery_runs")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _access_token() -> str:
    """Mint a token from the same service account the benchmark runner uses."""
    import google.auth
    import google.auth.transport.requests

    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token


def fetch_models(*, location: str | None = None, timeout: float = 60.0) -> tuple[list[dict[str, Any]], bool]:
    """Return (models, pagination_complete).

    `pagination_complete` is reported rather than assumed. A partial read that
    looks like a complete one is what makes deprecation decisions dangerous —
    models absent because the listing stopped early are indistinguishable from
    models the provider removed.
    """
    location = location or DEFAULT_LOCATION
    url = f"https://{location}-aiplatform.googleapis.com/v1beta1/publishers/{PUBLISHER}/models"
    headers = {"Authorization": f"Bearer {_access_token()}"}

    models: list[dict[str, Any]] = []
    page_token = ""
    for _ in range(MAX_PAGES):
        params = {"pageSize": PAGE_SIZE}
        if page_token:
            params["pageToken"] = page_token
        response = httpx.get(url, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        models.extend(payload.get("publisherModels") or [])
        page_token = payload.get("nextPageToken") or ""
        if not page_token:
            return models, True
    return models, False


def _model_id(row: dict[str, Any]) -> str | None:
    """`publishers/google/models/gemini-2.5-flash` -> `gemini-2.5-flash`."""
    name = row.get("name") or ""
    return name.rsplit("/", 1)[-1] or None


def refresh_catalog(db, *, now: datetime | None = None, location: str | None = None) -> dict[str, Any]:
    """Read Vertex's model list into provider_catalog and record the run.

    A failure is recorded as a failed run rather than swallowed. An absent row
    and a failed row mean different things to the invariant that reads them, and
    only one of them is a reason to stop trusting the catalogue.
    """
    now = now or utcnow()
    started_at = now
    catalog = db.provider_catalog
    runs = db[discovery_runs_collection_name()]

    try:
        rows, pagination_complete = fetch_models(location=location)
    except Exception as exc:  # noqa: BLE001
        runs.insert_one(
            {
                "provider": PROVIDER,
                "started_at": started_at,
                "finished_at": utcnow(),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "raw_count": 0,
                "accepted_count": 0,
                "new_count": 0,
                "pagination_complete": False,
                "source_version": SOURCE_VERSION,
            }
        )
        return {"status": "failed", "error": str(exc)}

    known = {doc["model_id"] for doc in catalog.find({"provider": PROVIDER}, {"model_id": 1})}
    accepted = 0
    new_models = []
    for row in rows:
        model_id = _model_id(row)
        if not model_id:
            continue
        accepted += 1
        if model_id not in known:
            new_models.append(model_id)
        catalog.update_one(
            {"provider": PROVIDER, "model_id": model_id},
            {
                "$set": {
                    "provider": PROVIDER,
                    "model_id": model_id,
                    "name": model_id,
                    "last_seen_at": now,
                    # Whether the model is servable is deliberately not inferred
                    # from this listing. Admission decides that by calling it.
                    "raw": {"versionId": row.get("versionId"), "launchStage": row.get("launchStage")},
                },
                "$setOnInsert": {"first_seen_at": now},
            },
            upsert=True,
        )

    runs.insert_one(
        {
            "provider": PROVIDER,
            "started_at": started_at,
            "finished_at": utcnow(),
            "status": "completed",
            "error": None,
            "raw_count": len(rows),
            "accepted_count": accepted,
            "new_count": len(new_models),
            "pagination_complete": pagination_complete,
            "source_version": SOURCE_VERSION,
        }
    )
    return {
        "status": "completed",
        "raw_count": len(rows),
        "accepted_count": accepted,
        "new_models": new_models,
        "pagination_complete": pagination_complete,
    }
