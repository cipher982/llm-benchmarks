from datetime import datetime
from datetime import timezone

import mongomock
from llm_bench import logging as benchmark_logging
from llm_bench.scheduler import queue
from llm_bench.scheduler.routing import DIRECT_TRANSPORT
from llm_bench.scheduler.routing import OPENROUTER_TRANSPORT
from llm_bench.scheduler.routing import ROUTE_DECISION_VERSION
from llm_bench.scheduler.routing import RouteDecision
from llm_bench.scheduler.routing import freeze_route_snapshot
from llm_bench.scheduler.routing import resolve_job_route


def active_snapshot(**overrides):
    value = {
        "source_provider": "deepinfra",
        "source_model_id": "Qwen/Qwen3-32B",
        "route_decision_version": ROUTE_DECISION_VERSION,
        "state": "active",
        "transport_provider": OPENROUTER_TRANSPORT,
        "route_policy": "pinned-provider",
        "route_model_id": "qwen/qwen3-32b",
        "route_provider_slug": "deepinfra",
        "observed_provider": "DeepInfra",
        "observed_provider_slug": "deepinfra",
        "route_snapshot_at": "2026-08-09T00:00:00+00:00",
        "route_probe_id": "probe-1",
        "provider_metadata_verified": True,
    }
    value.update(overrides)
    return value


def test_missing_or_non_active_evidence_is_direct():
    missing = RouteDecision.from_snapshot("deepinfra", "m", None)
    pending = RouteDecision.from_snapshot("deepinfra", "m", {"state": "canary_pending"})

    assert missing.transport_provider == DIRECT_TRANSPORT
    assert missing.reason == "missing-route-snapshot"
    assert pending.transport_provider == DIRECT_TRANSPORT
    assert pending.reason == "route-decision-version-mismatch"


def test_active_route_preserves_source_identity():
    decision = RouteDecision.from_snapshot("deepinfra", "Qwen/Qwen3-32B", active_snapshot())

    assert decision.transport_provider == OPENROUTER_TRANSPORT
    assert decision.transport_model_id == "qwen/qwen3-32b"
    assert decision.source_provider == "deepinfra"
    assert decision.source_model_id == "Qwen/Qwen3-32B"
    assert decision.metric_fields()["route_policy"] == "pinned-provider"


def test_provider_mismatch_and_expiry_fail_closed():
    mismatch = RouteDecision.from_snapshot(
        "deepinfra", "Qwen/Qwen3-32B", active_snapshot(observed_provider_slug="together")
    )
    expired = RouteDecision.from_snapshot(
        "deepinfra",
        "Qwen/Qwen3-32B",
        active_snapshot(expires_at="2026-08-08T00:00:00+00:00"),
        now=datetime(2026, 8, 9, tzinfo=timezone.utc),
    )

    assert mismatch.transport_provider == DIRECT_TRANSPORT
    assert mismatch.reason == "observed-provider-mismatch"
    assert expired.transport_provider == DIRECT_TRANSPORT
    assert expired.reason == "route-evidence-expired"


def test_incomplete_or_unverified_evidence_stays_direct():
    incomplete = RouteDecision.from_snapshot("deepinfra", "Qwen/Qwen3-32B", active_snapshot(route_probe_id=None))
    unverified = RouteDecision.from_snapshot(
        "deepinfra", "Qwen/Qwen3-32B", active_snapshot(provider_metadata_verified=False)
    )

    assert incomplete.reason == "incomplete-route-evidence"
    assert unverified.reason == "unverified-provider-metadata"
    assert incomplete.transport_provider == DIRECT_TRANSPORT
    assert unverified.transport_provider == DIRECT_TRANSPORT


def test_bedrock_route_is_always_direct():
    decision = RouteDecision.from_snapshot(
        "bedrock",
        "us.anthropic.claude-opus-4-7",
        active_snapshot(source_provider="bedrock", source_model_id="us.anthropic.claude-opus-4-7"),
    )

    assert decision.transport_provider == DIRECT_TRANSPORT
    assert decision.reason == "bedrock-out-of-scope"


def test_snapshot_is_copied_when_frozen_and_job_resolution_is_fail_closed():
    original = active_snapshot()
    frozen = freeze_route_snapshot("deepinfra", "Qwen/Qwen3-32B", original)
    original["state"] = "direct"
    decision = resolve_job_route({"provider": "deepinfra", "model_id": "Qwen/Qwen3-32B", "route_snapshot": frozen})

    assert frozen["state"] == "active"
    assert decision.transport_provider == OPENROUTER_TRANSPORT


def test_queue_persists_a_frozen_route_snapshot():
    db = mongomock.MongoClient()["llm-bench-test"]
    original = active_snapshot()
    queue.enqueue_manual_job(
        db,
        provider="deepinfra",
        model_id="Qwen/Qwen3-32B",
        now=datetime(2026, 8, 9, tzinfo=timezone.utc),
        route_snapshot=original,
    )
    original["state"] = "direct"

    saved = db.bench_jobs.find_one({"provider": "deepinfra"})
    assert saved["route_snapshot"]["state"] == "active"
    assert saved["route_snapshot"]["queued_at"] == "2026-08-09T00:00:00+00:00"


def test_metric_writer_keeps_source_identity_for_routed_config(monkeypatch):
    inserted = {}

    class Collection:
        def insert_one(self, data):
            inserted.update(data)

    monkeypatch.setattr(benchmark_logging, "setup_database", lambda *args: Collection())
    config = benchmark_logging.CloudConfig(
        provider="deepinfra",
        model_name="Qwen/Qwen3-32B",
        run_ts="2026-08-09 00:00:00",
        temperature=0.1,
        transport_provider="openrouter",
        transport_model_id="qwen/qwen3-32b",
    )
    metrics = {
        "gen_ts": "2026-08-09 00:00:01",
        "requested_tokens": 64,
        "output_tokens": 64,
        "generate_time": 1.0,
        "tokens_per_second": 64.0,
        "time_to_first_token": 0.1,
        "times_between_tokens": [],
        "transport_provider": "openrouter",
        "route_model_id": "qwen/qwen3-32b",
    }

    benchmark_logging.log_mongo("cloud", config, metrics, "mongodb://test", "db", "metrics")

    assert inserted["provider"] == "deepinfra"
    assert inserted["model_name"] == "Qwen/Qwen3-32B"
    assert inserted["transport_provider"] == "openrouter"
