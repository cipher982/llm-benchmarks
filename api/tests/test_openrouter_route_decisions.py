import mongomock

from scripts.openrouter_route_decisions import apply_decisions
from scripts.openrouter_route_decisions import materialize_report


def audit_row(source_key: str, *, decision: str = "keep-direct") -> dict:
    provider, model_id = source_key.split("/", 1)
    row = {
        "source_key": source_key,
        "provider": provider,
        "model_id": model_id,
        "decision": decision,
        "reason_class": "verified-pinned-route" if decision == "route-or" else "unknown",
    }
    if decision == "route-or":
        row.update(
            {
                "or_model_id": "qwen/qwen3-32b",
                "route_provider_slug": "deepinfra",
                "observed_provider": "DeepInfra",
                "evidence": {
                    "probe": {
                        "route_provider_slug": "deepinfra",
                        "observed_provider_slug": "deepinfra",
                        "provider_metadata_verified": True,
                    }
                },
            }
        )
    return row


def test_materializer_keeps_availability_candidates_fail_closed():
    report = materialize_report(
        {
            "generated_at": "2026-08-09T12:00:00+00:00",
            "rows": [audit_row("deepinfra/Qwen/Qwen3-32B", decision="route-or"), audit_row("openai/gpt-4o")],
        },
        audit_path="audit.json",
    )

    candidate, direct = report["decisions"]
    assert report["counts"] == {"candidate": 1, "direct": 1}
    assert candidate["state"] == "candidate"
    assert candidate["canary_state"] == "availability_passed"
    assert candidate["canary_successes"] == 0
    assert candidate["route_probe_id"].startswith("coverage:")
    assert direct["transport_provider"] == "direct"


def test_incomplete_route_evidence_stays_direct():
    row = audit_row("deepinfra/Qwen/Qwen3-32B", decision="route-or")
    row["evidence"]["probe"]["provider_metadata_verified"] = False
    report = materialize_report({"rows": [row]}, audit_path="audit.json")

    assert report["decisions"][0]["state"] == "direct"
    assert report["decisions"][0]["audit_reason"] == "incomplete-audit-evidence"


def test_apply_preserves_passed_canary():
    report = materialize_report(
        {"generated_at": "2026-08-09T12:00:00+00:00", "rows": [audit_row("openai/gpt-4o")]},
        audit_path="audit.json",
    )
    client = mongomock.MongoClient()
    collection = client["llm-bench"]["bench_route_decisions"]
    collection.insert_one(
        {
            "source_provider": "openai",
            "source_model_id": "gpt-4o",
            "route_snapshot_at": "2026-08-09T12:00:00+00:00",
            "canary_state": "passed",
            "canary_id": "canary-1",
        }
    )

    result = apply_decisions(report, client=client, db_name="llm-bench")

    assert result == {"inserted_or_updated": 0, "preserved_passed": 1}
    assert collection.find_one({"source_provider": "openai"})["canary_id"] == "canary-1"
