"""An endpoint that cannot be measured should stop being scheduled — and must
be able to come back when the measurement changes.

`gemini-3-pro-image` returns images, so every run ends with empty visible text.
Left enabled it is scheduled forever, spends money every pass, and sits in the
starvation check as a violation nobody can act on. But retiring it on a name
pattern is how two earlier passes missed veo, kling, vidu, ideogram and
parakeet, and retiring it permanently is how coverage once decayed to 11.7%.
"""

from datetime import datetime
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import endpoint_discovery
from llm_bench.ops import endpoint_retirement
from llm_bench.scheduler import policies
from llm_bench.scheduler.mongo import health_collection_name
from llm_bench.scheduler.mongo import jobs_collection_name

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db():
    return mongomock.MongoClient().db


def _endpoint(db, model_id, tag, enabled=True, **extra):
    db[endpoint_discovery.endpoints_collection_name()].insert_one(
        {"model_id": model_id, "endpoint_tag": tag, "enabled": enabled, **extra}
    )


def _health(db, model_id, tag, *, kind=None, message=None, failures=0, last_success_at=None):
    db[health_collection_name()].insert_one(
        {
            "provider": "openrouter",
            "model_id": model_id,
            "endpoint_tag": tag,
            "last_success_at": last_success_at,
            "last_error_kind": kind,
            "last_error_message": message,
            "failures_24h": failures,
        }
    )


def _enabled(db, model_id, tag):
    doc = db[endpoint_discovery.endpoints_collection_name()].find_one({"model_id": model_id, "endpoint_tag": tag})
    return doc["enabled"]


class TestRetirement:
    def test_an_endpoint_classified_incapable_is_retired(self, db):
        _endpoint(db, "google/gemini-3-pro-image", "google-ai-studio/global")
        _health(
            db,
            "google/gemini-3-pro-image",
            "google-ai-studio/global",
            kind="hard_capability",
            message="model returns images, not text",
            failures=5,
        )
        retired = endpoint_retirement.retire_unmeasurable_endpoints(db, now=NOW)
        assert len(retired) == 1
        assert _enabled(db, "google/gemini-3-pro-image", "google-ai-studio/global") is False
        health_doc = db[health_collection_name()].find_one(
            {"model_id": "google/gemini-3-pro-image", "endpoint_tag": "google-ai-studio/global"}
        )
        assert health_doc["enabled"] is False

    def test_empty_visible_text_alone_never_retires_anything(self, db):
        # A live dry run flagged deepseek-r1-0528 on this message alongside
        # three genuine image endpoints. A reasoning model that spends its
        # budget thinking produces exactly the same symptom, so the message is
        # not evidence of incapability -- it belongs to the classifier.
        _endpoint(db, "deepseek/deepseek-r1-0528", "novita/fp8")
        _health(
            db,
            "deepseek/deepseek-r1-0528",
            "novita/fp8",
            kind="unknown",
            message="visible output text is empty",
            failures=9,
        )
        assert endpoint_retirement.retire_unmeasurable_endpoints(db, now=NOW) == []
        assert _enabled(db, "deepseek/deepseek-r1-0528", "novita/fp8") is True

    def test_the_decision_is_recorded_so_it_can_be_audited_and_reversed(self, db):
        _endpoint(db, "m", "t")
        _health(db, "m", "t", kind="hard_capability", message="does not support streaming", failures=4)
        endpoint_retirement.retire_unmeasurable_endpoints(db, now=NOW)
        doc = db[endpoint_discovery.endpoints_collection_name()].find_one({"model_id": "m"})
        assert "no measurement in" in doc["disabled_reason"]
        # mongomock strips tzinfo on round-trip; the instant is what matters.
        assert doc["disabled_at"].replace(tzinfo=timezone.utc) == NOW
        assert doc["disabled_protocol_version"] == policies.MEASUREMENT_PROTOCOL_VERSION

    def test_a_timeout_is_not_a_verdict_about_the_endpoint(self, db):
        # Transient failure says nothing about whether it can be measured.
        _endpoint(db, "m", "t")
        _health(db, "m", "t", kind="timeout", message="deadline exceeded", failures=9)
        assert endpoint_retirement.retire_unmeasurable_endpoints(db, now=NOW) == []
        assert _enabled(db, "m", "t") is True

    def test_one_bad_run_is_an_incident_not_a_fact(self, db):
        _endpoint(db, "m", "t")
        _health(db, "m", "t", kind="hard_capability", message="does not support", failures=1)
        assert endpoint_retirement.retire_unmeasurable_endpoints(db, now=NOW) == []

    def test_an_endpoint_that_has_ever_succeeded_is_never_retired(self, db):
        _endpoint(db, "m", "t")
        _health(db, "m", "t", kind="hard_capability", message="does not support", failures=9, last_success_at=NOW)
        assert endpoint_retirement.retire_unmeasurable_endpoints(db, now=NOW) == []

    def test_the_verdict_is_read_from_the_job_when_health_has_none(self, db):
        # The disagreement that hid eight endpoints' real verdicts.
        _endpoint(db, "m", "t")
        _health(db, "m", "t")
        db[jobs_collection_name()].insert_one(
            {
                "_id": "openrouter:m:t",
                "provider": "openrouter",
                "model_id": "m",
                "endpoint_tag": "t",
                "status": "dead_letter",
                "last_attempt_error_kind": "unknown",
                "last_attempt_error_message": "model does not support chat completions",
                "attempt": 2,
            }
        )
        assert len(endpoint_retirement.retire_unmeasurable_endpoints(db, now=NOW)) == 1

    def test_dry_run_changes_nothing(self, db):
        _endpoint(db, "m", "t")
        _health(db, "m", "t", kind="hard_capability", message="does not support", failures=4)
        assert len(endpoint_retirement.retire_unmeasurable_endpoints(db, now=NOW, dry_run=True)) == 1
        assert _enabled(db, "m", "t") is True

    def test_retirement_is_never_decided_by_the_model_name(self, db):
        # "image" in the name, but it answers with text. Two earlier passes of
        # name matching missed veo, kling, vidu, ideogram and parakeet; this is
        # the same error in the other direction.
        _endpoint(db, "some/image-to-text-model", "t")
        _health(db, "some/image-to-text-model", "t", last_success_at=NOW)
        assert endpoint_retirement.retire_unmeasurable_endpoints(db, now=NOW) == []


class TestTheWayBack:
    def test_a_verdict_from_an_older_protocol_is_reconsidered(self, db):
        _endpoint(
            db,
            "m",
            "t",
            enabled=False,
            disabled_reason="no measurement in 3 attempts",
            disabled_protocol_version=policies.MEASUREMENT_PROTOCOL_VERSION - 1,
        )
        _health(db, "m", "t", failures=3)
        restored = endpoint_retirement.restore_stale_protocol_retirements(db, now=NOW)
        assert len(restored) == 1
        doc = db[endpoint_discovery.endpoints_collection_name()].find_one({"model_id": "m"})
        assert doc["enabled"] is True
        assert "disabled_reason" not in doc
        assert doc["missing_passes"] == 0
        health_doc = db[health_collection_name()].find_one({"model_id": "m", "endpoint_tag": "t"})
        assert health_doc["enabled"] is True

    def test_a_current_protocol_verdict_stands(self, db):
        _endpoint(
            db,
            "m",
            "t",
            enabled=False,
            disabled_protocol_version=policies.MEASUREMENT_PROTOCOL_VERSION,
        )
        assert endpoint_retirement.restore_stale_protocol_retirements(db, now=NOW) == []
        assert _enabled(db, "m", "t") is False

    def test_endpoints_disabled_for_other_reasons_are_left_alone(self, db):
        # Absent from OpenRouter listings, not a measurement verdict.
        _endpoint(db, "m", "t", enabled=False, disabled_reason="absent from 3 consecutive listings")
        assert endpoint_retirement.restore_stale_protocol_retirements(db, now=NOW) == []
        assert _enabled(db, "m", "t") is False
