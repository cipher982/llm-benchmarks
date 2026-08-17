"""Freshness is per endpoint, or the fleet lies about what it measured.

`deepinfra/bf16` and `deepinfra/turbo` are separate deployments of one model. If
they share a health record, measuring one marks the other fresh and the
scheduler stops asking for an endpoint nobody ever benchmarked. The symptom is
silence, which is the failure mode this codebase is worst at noticing.
"""

import mongomock
import pytest
from llm_bench.scheduler import health
from llm_bench.scheduler.queue import scheduled_job_id


@pytest.fixture
def db():
    return mongomock.MongoClient().db


def test_job_ids_separate_endpoints_of_one_model():
    a = scheduled_job_id("openrouter", "openai/gpt-oss-120b", "deepinfra/bf16")
    b = scheduled_job_id("openrouter", "openai/gpt-oss-120b", "deepinfra/turbo")
    assert a != b
    # Routes predating endpoint identity keep their two-part id.
    assert scheduled_job_id("openrouter", "m") == "openrouter:m"


def test_measuring_one_endpoint_does_not_refresh_its_sibling(db):
    for tag in ("deepinfra/bf16", "deepinfra/turbo"):
        health.refresh_model_health_doc(
            db,
            provider="openrouter",
            model_id="openai/gpt-oss-120b",
            endpoint_tag=tag,
            enabled=True,
            cadence_seconds=3600,
        )

    health.record_success(
        db,
        provider="openrouter",
        model_id="openai/gpt-oss-120b",
        endpoint_tag="deepinfra/bf16",
        cadence_seconds=3600,
    )

    measured = health.health_collection(db).find_one(
        health.health_filter("openrouter", "openai/gpt-oss-120b", "deepinfra/bf16")
    )
    sibling = health.health_collection(db).find_one(
        health.health_filter("openrouter", "openai/gpt-oss-120b", "deepinfra/turbo")
    )
    assert measured["last_success_at"] is not None
    assert sibling["last_success_at"] is None, "an unmeasured endpoint was marked fresh"


def test_a_model_record_and_its_endpoint_records_coexist(db):
    """The unique index must key on the tag, not reject the second endpoint."""
    health.ensure_indexes(db)
    for tag in (None, "groq", "deepinfra/fp8"):
        health.refresh_model_health_doc(
            db,
            provider="openrouter",
            model_id="m",
            endpoint_tag=tag,
            enabled=True,
            cadence_seconds=3600,
        )
    assert health.health_collection(db).count_documents({"model_id": "m"}) == 3


def test_dedupe_does_not_mistake_endpoints_for_duplicates(db):
    for tag in ("groq", "deepinfra/fp8", "novita/fp8"):
        health.refresh_model_health_doc(
            db,
            provider="openrouter",
            model_id="m",
            endpoint_tag=tag,
            enabled=True,
            cadence_seconds=3600,
        )
    removed = health.dedupe_existing_health_docs(db)
    assert removed == 0
    assert health.health_collection(db).count_documents({"model_id": "m"}) == 3
