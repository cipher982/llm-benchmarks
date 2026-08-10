from llm_bench.scheduler.runner import RunnerResult
from llm_bench.scheduler.runner import run_job_in_child


def active_route_snapshot():
    return {
        "source_provider": "deepinfra",
        "source_model_id": "Qwen/Qwen3-32B",
        "route_decision_version": "or-route-v1",
        "state": "active",
        "transport_provider": "openrouter",
        "route_policy": "pinned-provider",
        "route_model_id": "qwen/qwen3-32b",
        "route_provider_slug": "deepinfra",
        "observed_provider_slug": "deepinfra",
        "provider_metadata_verified": True,
        "route_snapshot_at": "2026-08-09T00:00:00+00:00",
        "route_probe_id": "probe-1",
        "canary_id": "canary-1",
        "canary_state": "passed",
        "canary_successes": 2,
        "canary_required_successes": 2,
        "canary_cost_status": "verified",
        "canary_evidence_uri": "s3://artifacts/canary.json",
        "canary_evidence_sha256": "a" * 64,
        "canary_promotion_gate": "passed",
        "canary_tps_ci95_lower": 0.9,
        "canary_ttft_ci95_upper": 1.2,
        "canary_cost_ci95_upper": 1.05,
        "expires_at": "2099-08-10T00:00:00+00:00",
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


def test_routed_child_timeout_marks_route_attempted(monkeypatch):
    monkeypatch.setenv("OPENROUTER_ROUTING_ENABLED", "1")
    monkeypatch.setattr("llm_bench.scheduler.runner.log_error_mongo", lambda **kwargs: "timeout")

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
