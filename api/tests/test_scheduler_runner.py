from llm_bench.scheduler.runner import RunnerResult
from llm_bench.scheduler.runner import run_job_in_child


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
