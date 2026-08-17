from datetime import datetime
from datetime import timezone

import mongomock
from llm_bench.ops import admission
from llm_bench.ops import openrouter_discovery

NOW = datetime(2026, 8, 16, 15, 0, tzinfo=timezone.utc)


def test_canonical_models_prefers_unsuffixed_variant():
    rows = [
        {"id": "qwen/qwen3-coder:free", "name": "free"},
        {"id": "qwen/qwen3-coder", "name": "canonical"},
        {"id": "deepseek/deepseek-v4-flash", "name": "flash"},
    ]

    result = openrouter_discovery.canonical_models(rows)

    assert [row["id"] for row in result] == ["deepseek/deepseek-v4-flash", "qwen/qwen3-coder"]
    assert result[1]["name"] == "canonical"


def test_refresh_catalog_mirrors_unique_models_and_records_run():
    db = mongomock.MongoClient()["test"]

    result = openrouter_discovery.refresh_catalog(
        db,
        now=NOW,
        fetcher=lambda: (
            [
                {"id": "qwen/qwen3-coder:free", "name": "free"},
                {"id": "qwen/qwen3-coder", "name": "Qwen3 Coder", "canonical_slug": "qwen3-coder"},
            ],
            2,
        ),
    )

    assert result["status"] == "completed"
    assert result["accepted_count"] == 1
    assert db.provider_catalog.count_documents({"provider": "openrouter"}) == 1
    row = db.provider_catalog.find_one({"provider": "openrouter"})
    assert row["model_id"] == "qwen/qwen3-coder"
    # OpenRouter's human label, not its dated machine slug. This asserted
    # "qwen3-coder" — the canonical_slug — which is how 195 of 236
    # leaderboard rows came to render as raw ids.
    assert row["name"] == "Qwen3 Coder"
    run = db.bench_discovery_runs.find_one({"provider": "openrouter"})
    assert run["pagination_complete"] is True


def test_refresh_failure_is_not_a_completed_empty_catalogue():
    db = mongomock.MongoClient()["test"]

    result = openrouter_discovery.refresh_catalog(
        db,
        now=NOW,
        fetcher=lambda: (_ for _ in ()).throw(RuntimeError("upstream down")),
    )

    assert result["status"] == "failed"
    run = db.bench_discovery_runs.find_one({"provider": "openrouter"})
    assert run["status"] == "failed"
    assert run["pagination_complete"] is False


def test_legacy_disabled_openrouter_reference_becomes_a_candidate():
    db = mongomock.MongoClient()["test"]
    db.models.insert_one(
        {
            "provider": "openrouter",
            "model_id": "qwen/qwen3-coder",
            "enabled": False,
            "deprecated": True,
        }
    )
    db.provider_catalog.insert_one({"provider": "openrouter", "model_id": "qwen/qwen3-coder", "name": "qwen3-coder"})

    registered = admission.register_candidates(db, now=NOW, limit=10)

    assert registered == ["openrouter/qwen/qwen3-coder"]
    row = db.models.find_one({"provider": "openrouter"})
    assert row["enabled"] is False
    assert row["deprecated"] is False
    assert row["status"] == admission.CANDIDATE_STATUS


def test_canonical_models_drops_openrouter_routers():
    """A router picks a different upstream every call; it cannot be benchmarked."""

    rows = [
        {"id": "openrouter/auto-beta"},
        {"id": "openrouter/free"},
        {"id": "openrouter/pareto-code"},
        {"id": "meta-llama/llama-4-maverick"},
    ]

    result = openrouter_discovery.canonical_models(rows)

    assert [row["id"] for row in result] == ["meta-llama/llama-4-maverick"]
