from llm_bench.discovery.cli import format_mongosh_add_command
from llm_bench.discovery.matcher import ModelMatch
from llm_bench.discovery.promotion import build_promotion_plan


def test_promotion_candidate_is_disabled_with_audit_fields():
    match = ModelMatch(
        openrouter_id="anthropic/claude-opus-4.7",
        openrouter_name="Anthropic Claude Opus 4.7",
        provider="bedrock",
        model_id="us.anthropic.claude-opus-4-7",
        confidence=0.91,
        reasoning="Available through Bedrock global endpoint.",
    )

    plan = build_promotion_plan(match, [])

    assert plan.insert_doc["enabled"] is False
    assert plan.insert_doc["promotion_status"] == "candidate"
    assert plan.insert_doc["promotion_source"] == "openrouter"
    assert plan.display_name == "claude-opus-4.7"


def test_promotion_plan_warns_on_duplicate_bedrock_alias_identity():
    match = ModelMatch(
        openrouter_id="anthropic/claude-opus-4.6",
        openrouter_name="Anthropic Claude Opus 4.6",
        provider="bedrock",
        model_id="us.anthropic.claude-opus-4-6-20251201-v1:0",
        confidence=0.9,
        reasoning="Alias for current Opus 4.6.",
    )

    plan = build_promotion_plan(
        match,
        [
            {
                "provider": "bedrock",
                "model_id": "us.anthropic.claude-opus-4-6-v1",
                "display_name": "claude-opus-4.6",
                "enabled": True,
            }
        ],
    )

    assert plan.warnings == ["enabled alias already exists for claude-opus-4-6: us.anthropic.claude-opus-4-6-v1"]


def test_formatted_command_does_not_enable_discovered_model():
    match = ModelMatch(
        openrouter_id="anthropic/claude-haiku-4.5",
        openrouter_name="Anthropic Claude Haiku 4.5",
        provider="bedrock",
        model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        confidence=0.88,
        reasoning="Bedrock model ID match.",
    )

    command = format_mongosh_add_command(match, [])

    assert '"enabled": false' in command
    assert "catalog-hygiene" in command
    assert "promotion_openrouter_id" in command
