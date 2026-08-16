import mongomock
from llm_bench.scheduler import runner
from llm_bench.scheduler import worker


def test_direct_fallback_job_cannot_resolve_to_openrouter():
    job = {
        "provider": "deepinfra",
        "model_id": "Qwen/Qwen3-32B",
        "route_snapshot": {
            "state": "active",
            "transport_provider": "openrouter",
            "route_policy": "pinned-provider",
            "route_revocation_generation": 0,
        },
    }

    fallback = worker.direct_fallback_job(job, reason="openrouter-quota-unavailable")

    assert fallback["route_fallback_reason"] == "openrouter-quota-unavailable"
    assert fallback["route_snapshot"]["state"] == "revoked"
    assert fallback["route_snapshot"]["transport_provider"] == "direct"


def test_openrouter_native_models_use_the_shared_openrouter_lane(monkeypatch):
    monkeypatch.delenv("OPENROUTER_ROUTING_ENABLED", raising=False)

    job = {"provider": "openrouter", "model_id": "qwen/qwen3-coder"}

    assert runner.job_requires_openrouter(job) is True
    assert runner.is_openrouter_native_job(job) is True


def test_dispatch_guard_marks_newer_generation_revoked():
    db = mongomock.MongoClient()["llm-bench"]
    db["bench_route_revocations"].insert_one(
        {"source_provider": "deepinfra", "source_model_id": "Qwen/Qwen3-32B", "generation": 2}
    )
    job = {
        "provider": "deepinfra",
        "model_id": "Qwen/Qwen3-32B",
        "route_snapshot": {
            "state": "active",
            "transport_provider": "openrouter",
            "route_policy": "pinned-provider",
            "route_revocation_generation": 1,
        },
    }

    guarded = worker.apply_route_revocation_guard(db, job)

    assert guarded["route_snapshot"]["state"] == "revoked"
    assert guarded["route_snapshot"]["route_revocation_generation"] == 2
    assert guarded["route_fallback_reason"] == "route-revoked-before-dispatch"
