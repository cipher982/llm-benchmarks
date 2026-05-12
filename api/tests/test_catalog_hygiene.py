from llm_bench.catalog_hygiene import analyze_catalog_rows
from llm_bench.catalog_hygiene import normalize_display_identity


def test_bedrock_hygiene_preserves_real_claude_versions():
    rows = [
        {
            "provider": "bedrock",
            "model_id": "us.anthropic.claude-opus-4-5-20251101-v1:0",
            "display_name": "claude-opus-4.5",
            "enabled": True,
        },
        {
            "provider": "bedrock",
            "model_id": "us.anthropic.claude-opus-4-6-v1",
            "display_name": "claude-opus-4.6",
            "enabled": True,
        },
        {
            "provider": "bedrock",
            "model_id": "us.anthropic.claude-opus-4-7",
            "display_name": "claude-opus-4.7",
            "enabled": True,
        },
    ]

    assert analyze_catalog_rows(rows, provider="bedrock") == []


def test_bedrock_hygiene_flags_duplicate_enabled_aliases():
    rows = [
        {
            "provider": "bedrock",
            "model_id": "us.anthropic.claude-opus-4-6-v1",
            "enabled": True,
        },
        {
            "provider": "bedrock",
            "model_id": "us.anthropic.claude-opus-4-6-20251201-v1:0",
            "enabled": True,
        },
    ]

    issues = analyze_catalog_rows(rows, provider="bedrock")

    assert {issue.code for issue in issues} == {"duplicate_enabled_alias"}
    assert {issue.normalized_identity for issue in issues} == {"claude-opus-4-6"}


def test_bedrock_hygiene_flags_checkpoint_suffix_in_display_fields():
    rows = [
        {
            "provider": "bedrock",
            "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "display_name": "claude-haiku-4.5-20251001",
            "canonical_id": "claude-haiku-4-5-20251001",
            "enabled": True,
        }
    ]

    issues = analyze_catalog_rows(rows, provider="bedrock")

    assert [issue.field for issue in issues] == ["display_name", "canonical_id"]
    assert all(issue.code == "checkpoint_suffix" for issue in issues)


def test_bedrock_display_identity_strips_provider_date_but_not_version():
    assert (
        normalize_display_identity(
            "bedrock",
            "us.anthropic.claude-opus-4-5-20251101-v1:0",
        )
        == "claude-opus-4-5"
    )
