from datetime import datetime

import mongomock
from llm_bench.model_lifecycle.collector import collect_lifecycle_snapshots


def test_lifecycle_metrics_keep_direct_and_routed_transports_separate(monkeypatch):
    # mongomock stores datetimes as naive UTC values.  Use the same shape for
    # aggregation cutoffs while the production collector still normalizes its
    # returned timestamps to UTC-aware values.
    now = datetime(2026, 8, 10)
    client = mongomock.MongoClient()
    db = client["llm-bench-test"]
    db.models.insert_one(
        {
            "provider": "deepinfra",
            "model_id": "Qwen/Qwen3-32B",
            "display_name": "Qwen 3 32B",
            "enabled": True,
        }
    )
    db.metrics_cloud_v2.insert_many(
        [
            {
                "provider": "deepinfra",
                "model_name": "Qwen/Qwen3-32B",
                "transport_provider": "direct",
                "gen_ts": now,
            },
            {
                "provider": "deepinfra",
                "model_name": "Qwen/Qwen3-32B",
                "transport_provider": "openrouter",
                "gen_ts": now,
            },
        ]
    )
    monkeypatch.setenv("MONGODB_URI", "mongodb://test")

    snapshots = collect_lifecycle_snapshots(now=now, client=client)

    assert [(item.provider, item.model_id, item.transport_provider) for item in snapshots] == [
        ("deepinfra", "Qwen/Qwen3-32B", "direct"),
        ("deepinfra", "Qwen/Qwen3-32B", "openrouter"),
    ]
    assert all(item.successes.successes_7d == 1 for item in snapshots)


def test_legacy_lifecycle_rows_default_to_direct(monkeypatch):
    now = datetime(2026, 8, 10)
    client = mongomock.MongoClient()
    db = client["llm-bench-test"]
    db.metrics_cloud_v2.insert_one(
        {
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "gen_ts": now,
        }
    )
    monkeypatch.setenv("MONGODB_URI", "mongodb://test")

    snapshots = collect_lifecycle_snapshots(now=now, client=client)

    assert len(snapshots) == 1
    assert snapshots[0].transport_provider == "direct"
