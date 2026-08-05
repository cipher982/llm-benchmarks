"""Shadow samples for models the default profile cannot measure.

Nine enabled models have no data because a reasoning model spends the whole
64-token budget on hidden reasoning and emits nothing visible. Measuring them
under a larger budget is not a publication change: shadow rows go to the probe
collection and never reach the site.
"""

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import mongomock
import pytest
from llm_bench.ops import reasoning_shadow
from llm_bench.scheduler import queue
from llm_bench.scheduler import runner

NOW = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def db():
    return mongomock.MongoClient()["llm-bench"]


def _model(db, provider, model_id, **extra):
    db.models.insert_one({"provider": provider, "model_id": model_id, "enabled": True, **extra})


def _job(db, provider, model_id, kind):
    db.bench_jobs.insert_one(
        {
            "_id": f"{provider}:{model_id}",
            "provider": provider,
            "model_id": model_id,
            "status": "dead_letter",
            "last_attempt_error_kind": kind,
            "updated_at": NOW,
        }
    )


def test_only_budget_exhausted_models_qualify(db):
    _model(db, "deepinfra", "MiniMaxAI/MiniMax-M2")
    _job(db, "deepinfra", "MiniMaxAI/MiniMax-M2", "budget_exhausted")
    # A genuinely dead model must not be measured again under any profile.
    _model(db, "together", "gone/model")
    _job(db, "together", "gone/model", "hard_model")
    # Nor a healthy one.
    _model(db, "groq", "fine/model")

    assert reasoning_shadow.find_unmeasurable(db) == [("deepinfra", "MiniMaxAI/MiniMax-M2")]


def test_shadow_jobs_carry_the_profile_and_do_not_publish(db):
    _model(db, "deepinfra", "MiniMaxAI/MiniMax-M2")
    _job(db, "deepinfra", "MiniMaxAI/MiniMax-M2", "budget_exhausted")

    assert reasoning_shadow.enqueue_shadow_samples(db, now=NOW) == ["deepinfra/MiniMaxAI/MiniMax-M2"]

    job = db.bench_jobs.find_one({"job_kind": "shadow"})
    assert job["sample_role"] == "shadow"
    assert job["benchmark_profile_id"] == reasoning_shadow.PROFILE_ID
    assert job["sample_role"] in runner.NON_PUBLISHING_ROLES
    # Reasoning generations are long; the default deadline would kill them and
    # record a timeout that says nothing about the model.
    assert job["deadline_seconds"] > 120


def test_a_recent_shadow_sample_is_not_repeated(db):
    _model(db, "deepinfra", "MiniMaxAI/MiniMax-M2")
    _job(db, "deepinfra", "MiniMaxAI/MiniMax-M2", "budget_exhausted")
    db[runner.probe_metrics_collection_name()].insert_one(
        {
            "provider": "deepinfra",
            "model_name": "MiniMaxAI/MiniMax-M2",
            "benchmark_profile_id": reasoning_shadow.PROFILE_ID,
            "run_ts": NOW - timedelta(minutes=5),
        }
    )

    assert reasoning_shadow.enqueue_shadow_samples(db, now=NOW) == []


def test_the_reasoning_profile_asks_for_a_bigger_budget():
    default = runner.profile_max_tokens(runner.DEFAULT_PROFILE_ID)
    reasoning = runner.profile_max_tokens(reasoning_shadow.PROFILE_ID)
    assert default == 64
    assert reasoning > default


def test_an_unknown_profile_falls_back_to_the_default():
    """An unrecognised id must not silently become a different measurement."""
    assert runner.profile_max_tokens("cloud-typo-v9") == runner.profile_max_tokens(runner.DEFAULT_PROFILE_ID)


def test_a_shadow_job_on_an_enabled_model_is_eligible(db):
    """The claim path cancels ineligible work.

    Shadow work targets enabled models, so requiring `probing` — as probe work
    does — would have cancelled every one of these jobs the moment a worker
    picked it up.
    """
    _model(db, "deepinfra", "MiniMaxAI/MiniMax-M2")

    assert queue.is_model_eligible(db, provider="deepinfra", model_id="MiniMaxAI/MiniMax-M2", sample_role="shadow")
    assert not queue.is_model_eligible(db, provider="deepinfra", model_id="MiniMaxAI/MiniMax-M2", sample_role="probe")


def test_a_shadow_job_on_a_deprecated_model_is_not_eligible(db):
    _model(db, "together", "retired/model", deprecated=True)

    assert not queue.is_model_eligible(db, provider="together", model_id="retired/model", sample_role="shadow")
