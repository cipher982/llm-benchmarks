import threading
from datetime import datetime
from datetime import timezone

import mongomock
from llm_bench.scheduler import queue
from llm_bench.scheduler import worker
from llm_bench.scheduler.runner import RunnerResult


class FakeClient:
    def __getitem__(self, name):
        return object()

    def close(self):
        pass


def test_worker_survives_mongodb_error_when_claiming_job(monkeypatch):
    stop_event = threading.Event()

    def fail_claim(*args, **kwargs):
        stop_event.set()
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(worker, "mongo_env", lambda: ("unused", "test"))
    monkeypatch.setattr(worker, "mongo_client", lambda: FakeClient())
    monkeypatch.setattr(queue, "claim_next_job", fail_claim)

    worker.run_worker_loop(
        provider="openai",
        slot=0,
        cadence_seconds=1800,
        stop_event=stop_event,
        idle_sleep_seconds=0,
    )


def test_shared_openrouter_gate_bounds_concurrency(monkeypatch):
    gate = threading.BoundedSemaphore(1)
    monkeypatch.setattr(worker, "OPENROUTER_GATE", gate)

    assert worker.acquire_openrouter_slot(1) is True
    assert worker.acquire_openrouter_slot(1) is False
    gate.release()
    assert worker.acquire_openrouter_slot(1) is True
    gate.release()


def test_route_failure_persists_cooldown_without_deleting_source_lane():
    db = mongomock.MongoClient()["llm-bench"]
    job = {
        "_id": "job-1",
        "status": "running",
        "provider": "deepinfra",
        "model_id": "Qwen/Qwen3-32B",
        "route_snapshot": {
            "source_provider": "deepinfra",
            "source_model_id": "Qwen/Qwen3-32B",
            "route_snapshot_at": "2026-08-10T00:00:00+00:00",
            "state": "active",
        },
    }
    result = RunnerResult(status="success", route_attempted=True, fallback_reason="route 429")
    queue.jobs_collection(db).insert_one(job)

    worker.persist_route_cooldown(
        db,
        job=job,
        result=result,
        now=datetime(2026, 8, 10, tzinfo=timezone.utc),
    )

    saved = db["bench_route_decisions"].find_one({"source_provider": "deepinfra"})
    assert saved["state"] == "cooldown"
    assert saved["route_health_failure_reason"] == "route 429"
    assert saved["source_model_id"] == "Qwen/Qwen3-32B"
    saved_job = db[queue.jobs_collection_name()].find_one({"_id": "job-1"})
    assert saved_job["route_snapshot"]["state"] == "cooldown"
