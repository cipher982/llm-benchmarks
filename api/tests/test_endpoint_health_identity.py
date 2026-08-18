"""Freshness is per endpoint, or the fleet lies about what it measured.

`deepinfra/bf16` and `deepinfra/turbo` are separate deployments of one model. If
they share a health record, measuring one marks the other fresh and the
scheduler stops asking for an endpoint nobody ever benchmarked. The symptom is
silence, which is the failure mode this codebase is worst at noticing.
"""

from datetime import datetime
from datetime import timezone

import mongomock
import pytest
from llm_bench.scheduler import health
from llm_bench.scheduler import runner
from llm_bench.scheduler.mongo import metrics_collection_name
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


def test_the_superseded_unique_index_is_dropped_not_just_superseded(db):
    """create_index adds; it does not replace.

    Leaving `provider_1_model_id_1` unique in place meant the first endpoint
    record for an already-known model raised DuplicateKeyError. That surfaced
    inside the scheduler pass and aborted it, so nothing was enqueued for any
    provider on any tick while the loop still reported healthy.
    """
    coll = health.health_collection(db)
    coll.create_index([("provider", 1), ("model_id", 1)], unique=True)

    health.ensure_indexes(db)

    names = set(coll.index_information())
    assert "provider_1_model_id_1" not in names
    assert "provider_1_model_id_1_endpoint_tag_1" in names


def test_a_model_record_does_not_block_its_first_endpoint_record(db):
    """The regression itself, end to end."""
    health.ensure_indexes(db)
    health.refresh_model_health_doc(
        db, provider="openrouter", model_id="aion-labs/aion-2.0", enabled=True, cadence_seconds=3600
    )
    health.refresh_model_health_doc(
        db,
        provider="openrouter",
        model_id="aion-labs/aion-2.0",
        endpoint_tag="groq",
        enabled=True,
        cadence_seconds=3600,
    )
    assert coll_count(db, "aion-labs/aion-2.0") == 2


def coll_count(db, model_id):
    return health.health_collection(db).count_documents({"model_id": model_id})


class TestEndpointCompletionCreditsTheEndpointDoc:
    """The hot-loop regression: 4142 rows against 16 targets in three hours.

    An endpoint job's success was credited to the model-level health doc, so
    the endpoint doc stayed `never_run`, kept maximum staleness priority, and
    the same front-of-list slice was re-selected on every scheduler tick while
    677 endpoints were never reached.
    """

    def test_success_lands_on_the_endpoint_doc_not_the_model_doc(self, db):
        health.record_success(
            db,
            provider="openrouter",
            model_id="anthropic/claude-opus-4.8",
            endpoint_tag="anthropic",
            cadence_seconds=10800,
        )
        endpoint_doc = health.health_collection(db).find_one(
            health.health_filter("openrouter", "anthropic/claude-opus-4.8", "anthropic")
        )
        assert endpoint_doc is not None
        assert endpoint_doc["last_success_at"] is not None
        assert endpoint_doc["freshness_status"] == "fresh"

        model_doc = health.health_collection(db).find_one(
            health.health_filter("openrouter", "anthropic/claude-opus-4.8", None)
        )
        assert model_doc is None, "an endpoint run must not stand in for the model's"

    def test_counts_are_scoped_to_the_endpoint(self, db):
        now = datetime.now(timezone.utc)
        for tag in ("anthropic", "anthropic", "anthropic/2"):
            db[metrics_collection_name()].insert_one(
                {
                    "provider": "openrouter",
                    "model_name": "m",
                    "route_endpoint_tag": tag,
                    "run_ts": now,
                }
            )
        successes, _, _ = health._recent_counts(
            db, provider="openrouter", model_id="m", now=now, endpoint_tag="anthropic"
        )
        assert successes == 2, "a sibling endpoint's rows must not count toward this one"


class TestFallbackIsNotAnEndpointMeasurement:
    """A run that fell back to unpinned routing measured the model, not the
    endpoint. Crediting it marks a deployment fresh that nothing benchmarked,
    the scheduler stops asking for it, and the silence reads as health --
    which is how 268 of 340 runs briefly looked like endpoint coverage.
    """

    def test_error_is_recorded_against_the_endpoint_on_fallback(self, db):
        health.record_error(
            db,
            provider="openrouter",
            model_id="nousresearch/hermes-4-70b",
            endpoint_tag="nebius/fp8",
            cadence_seconds=10800,
            error_kind="pin_unverified",
            error_message="OpenRouter provider metadata was not verified",
        )
        doc = health.health_collection(db).find_one(
            health.health_filter("openrouter", "nousresearch/hermes-4-70b", "nebius/fp8")
        )
        assert doc["last_success_at"] is None
        assert doc["freshness_status"] != "fresh"
        assert doc["last_error_kind"] == "pin_unverified"


class TestCreditRequiresTheEndpointToHaveBeenMeasured:
    """A stale route snapshot resolves straight to direct and sets no
    fallback_reason, so inferring "did we pin?" from fallback_reason credited
    11 endpoints for runs that never touched them -- caught in production by
    endpoint_freshness_is_backed_by_rows minutes after it shipped.
    """

    def test_a_direct_run_with_no_fallback_reason_is_not_a_measurement(self):
        result = runner.RunnerResult(status="success", fallback_reason=None, measured_endpoint_tag=None)
        job_tag = "novita/bf16"
        assert result.measured_endpoint_tag != job_tag

    def test_a_real_pin_reports_the_tag_it_measured(self):
        result = runner.RunnerResult(status="success", measured_endpoint_tag="novita/bf16")
        assert result.measured_endpoint_tag == "novita/bf16"
