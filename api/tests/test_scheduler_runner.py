import time

from llm_bench.scheduler import policies
from llm_bench.scheduler.runner import RunnerResult
from llm_bench.scheduler.runner import run_job_in_child


def active_route_snapshot():
    return {
        "source_provider": "deepinfra",
        "source_model_id": "Qwen/Qwen3-32B",
        "route_decision_version": "or-route-v1",
        "state": "active",
        "transport_provider": "openrouter",
        "route_policy": "pinned-endpoint",
        "route_model_id": "qwen/qwen3-32b",
        "route_provider_slug": "deepinfra",
        "route_endpoint_tag": "deepinfra/fp8",
        "route_snapshot_at": "2026-08-09T00:00:00+00:00",
    }


def test_smoke_hang_success_does_not_write_metrics():
    result = run_job_in_child(
        {
            "_id": "smoke_hang:openai:fake-hang",
            "provider": "openai",
            "model_id": "fake-hang",
            "job_kind": "smoke_hang",
            "smoke_seconds": 0,
        },
        deadline_seconds=2,
    )

    assert result == RunnerResult(status="success")


def test_smoke_hang_timeout_kills_child(monkeypatch):
    logged = []

    def fake_log_error_mongo(**kwargs):
        logged.append(kwargs)
        return "timeout"

    monkeypatch.setattr("llm_bench.scheduler.runner.log_error_mongo", fake_log_error_mongo)

    started = time.monotonic()
    result = run_job_in_child(
        {
            "_id": "smoke_hang:openai:fake-hang",
            "provider": "openai",
            "model_id": "fake-hang",
            "job_kind": "smoke_hang",
            "smoke_seconds": 10,
        },
        deadline_seconds=0.2,
    )

    assert result.status == "timeout"
    assert result.error_kind == "timeout"
    assert logged[0]["stage"] == "timeout"
    assert time.monotonic() - started < policies.LEASE_GRACE_SECONDS


def test_routed_child_timeout_marks_route_attempted(monkeypatch):
    monkeypatch.setenv("OPENROUTER_ROUTING_ENABLED", "1")
    monkeypatch.setattr("llm_bench.scheduler.runner.log_error_mongo", lambda **kwargs: "timeout")

    started = time.monotonic()
    result = run_job_in_child(
        {
            "_id": "smoke_hang:deepinfra:routed",
            "provider": "deepinfra",
            "model_id": "Qwen/Qwen3-32B",
            "job_kind": "smoke_hang",
            "smoke_seconds": 10,
            "route_snapshot": active_route_snapshot(),
        },
        deadline_seconds=0.2,
    )

    assert result.status == "timeout"
    assert result.route_attempted is True
    assert result.transport_provider == "openrouter"
    assert time.monotonic() - started < policies.LEASE_GRACE_SECONDS
