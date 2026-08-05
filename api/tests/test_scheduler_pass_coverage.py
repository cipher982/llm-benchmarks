"""The per-pass cap must bound work, never exclude a model permanently.

Found in production: DeepInfra had 112 enabled models and the pass sliced the
list at 100. The same twelve fell off every time, because the order is stable.
They had run successfully months before and then stopped — no error, no dead
letter, nothing disabled — and admission adding models made it worse each week.
"""

from unittest.mock import patch

from llm_bench.scheduler import cli


def _health_doc(staleness):
    return {"freshness_status": "stale", "staleness_seconds": staleness}


def test_every_model_can_be_scheduled_even_past_the_cap():
    models = [f"model-{i:03d}" for i in range(112)]
    # The tail is stalest, which a slice of the first 100 would never reach.
    staleness = {model_id: 100 + index for index, model_id in enumerate(models)}
    enqueued = []

    class FakeHealthCollection:
        def find_one(self, query):
            return _health_doc(staleness[query["model_id"]])

    with (
        patch.object(cli, "mongo_env", return_value=("uri", "db")),
        patch.object(cli, "mongo_client") as client,
        patch.object(cli, "load_provider_models", return_value={"deepinfra": models}),
        patch.object(cli.health, "refresh_all_model_docs"),
        patch.object(cli.health, "heartbeat"),
        patch.object(cli.health, "health_collection", return_value=FakeHealthCollection()),
        patch.object(cli.queue, "enqueue_scheduled_job", side_effect=lambda *a, **k: enqueued.append(k["model_id"])),
    ):
        client.return_value.__getitem__.return_value = object()
        cli.scheduler_pass(providers="deepinfra", limit=100, cadence_seconds=900)

    assert len(enqueued) == 100, "the cap still bounds one pass"
    # The twelve the old slice dropped are the stalest, so they go first.
    assert set(models[-12:]) <= set(enqueued)


def test_the_cap_takes_the_stalest_first():
    models = ["fresh-ish", "very-stale", "middling"]
    staleness = {"fresh-ish": 10, "very-stale": 9000, "middling": 500}
    enqueued = []

    class FakeHealthCollection:
        def find_one(self, query):
            return _health_doc(staleness[query["model_id"]])

    with (
        patch.object(cli, "mongo_env", return_value=("uri", "db")),
        patch.object(cli, "mongo_client") as client,
        patch.object(cli, "load_provider_models", return_value={"groq": models}),
        patch.object(cli.health, "refresh_all_model_docs"),
        patch.object(cli.health, "heartbeat"),
        patch.object(cli.health, "health_collection", return_value=FakeHealthCollection()),
        patch.object(cli.queue, "enqueue_scheduled_job", side_effect=lambda *a, **k: enqueued.append(k["model_id"])),
    ):
        client.return_value.__getitem__.return_value = object()
        cli.scheduler_pass(providers="groq", limit=2, cadence_seconds=900)

    assert enqueued == ["very-stale", "middling"]
