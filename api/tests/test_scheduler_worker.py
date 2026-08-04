import threading

from llm_bench.scheduler import queue
from llm_bench.scheduler import worker


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
