import json

import mongomock
import pytest

from scripts.openrouter_retire_unrouted import retire_unrouted
from scripts.openrouter_retire_unrouted import wave_keys


def _db():
    return mongomock.MongoClient()["llm-bench"]


def _model(provider: str, model_id: str):
    return {"provider": provider, "model_id": model_id, "enabled": True, "deprecated": False}


def _route(provider: str, model_id: str):
    return {
        "source_provider": provider,
        "source_model_id": model_id,
        "transport_provider": "openrouter",
        "state": "active",
    }


def test_retirement_requires_a_complete_wave_covering_enabled_rows():
    db = _db()
    db.models.insert_many([_model("deepinfra", "one"), _model("deepinfra", "two")])

    with pytest.raises(ValueError, match="incomplete marketplace wave"):
        retire_unrouted(
            db=db,
            wave_id="wave-1",
            passing={("deepinfra", "one")},
            covered={("deepinfra", "one")},
            apply=False,
        )


def test_active_routes_are_protected_even_when_absent_from_this_wave():
    db = _db()
    db.models.insert_many([_model("deepinfra", "one"), _model("deepinfra", "two")])
    db.bench_route_decisions.insert_one(_route("deepinfra", "one"))

    result = retire_unrouted(
        db=db,
        wave_id="wave-1",
        passing=set(),
        covered={("deepinfra", "two")},
        apply=True,
    )

    assert result["retirements"] == 1
    assert db.models.find_one({"model_id": "one"})["enabled"] is True
    retired = db.models.find_one({"model_id": "two"})
    assert retired["enabled"] is False
    assert retired["disabled_class"] == "openrouter_unrouted"


def test_native_openrouter_rows_are_not_treated_as_source_rows():
    db = _db()
    db.models.insert_many(
        [
            _model("openrouter", "native-model"),
            _model("deepinfra", "source-model"),
        ]
    )

    result = retire_unrouted(
        db=db,
        wave_id="wave-1",
        passing=set(),
        covered={("deepinfra", "source-model")},
        apply=False,
    )

    assert result["enabled_non_core"] == 1
    assert result["retirements"] == 1


def test_wave_keys_include_candidates_and_skipped_rows(tmp_path):
    report = tmp_path / "decisions.json"
    report.write_text(
        json.dumps(
            {
                "decisions": [{"source_provider": "deepinfra", "source_model_id": "one"}],
                "skipped": [{"source": "together/model/with/slashes", "reason": "no-alias"}],
            }
        ),
        encoding="utf-8",
    )

    assert wave_keys(report) == {
        ("deepinfra", "one"),
        ("together", "model/with/slashes"),
    }
