"""
CLI interface for model discovery via OpenRouter.

Usage:
    uv run python -m api.llm_bench.discovery.cli fetch
    uv run python -m api.llm_bench.discovery.cli report [--max-matches 20]
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import List
from typing import Optional

import dotenv
import typer

from .matcher import ModelMatch
from .matcher import match_to_direct_providers
from .matcher import store_matches_in_db
from .openrouter import fetch_openrouter_models
from .openrouter import get_catalog_from_db
from .openrouter import get_our_models_from_db
from .openrouter import store_catalog_in_db
from .promotion import PromotionPlan
from .promotion import build_promotion_plan

# Setup
app = typer.Typer(help="Model discovery via OpenRouter catalog")
dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def format_mongosh_add_command(match: ModelMatch, our_models: List[dict] | None = None) -> str:
    """
    Generate a copy-paste MongoDB command to stage a promotion candidate.

    Discovery does not automatically enable models. The candidate row carries
    audit fields and must be reviewed before `enabled` is flipped to true.
    """
    plan = build_promotion_plan(match, our_models or [])
    return format_promotion_command(plan)


def format_promotion_command(plan: PromotionPlan) -> str:
    comments = [
        "# Stage as disabled promotion candidate; review before enabling.",
        f"# Provider/model: {plan.provider}/{plan.model_id}",
    ]
    if plan.provider == "bedrock":
        comments.append("# Bedrock: run catalog-hygiene before enabling this candidate.")
    comments.extend(f"# WARNING: {warning}" for warning in plan.warnings)
    doc = json.dumps(plan.insert_doc, indent=2)
    return (
        "\n".join(comments)
        + f"""
mongosh "$MONGODB_URI" --eval '
db.models.updateOne(
  {{provider: "{plan.provider}", model_id: "{plan.model_id}"}},
  {{$setOnInsert: {doc}, $set: {{promotion_reviewed_at: new Date()}}}},
  {{upsert: true}}
)'"""
    )


@app.command()
def fetch():
    """
    Fetch the latest OpenRouter catalog and store in MongoDB.

    This command:
    1. Calls OpenRouter API (free, no auth)
    2. Stores results in openrouter_catalog collection
    3. Updates first_seen_at and last_seen_at timestamps
    """
    typer.echo("Fetching OpenRouter catalog...")

    try:
        # Fetch catalog
        models = asyncio.run(fetch_openrouter_models())

        # Store in DB
        stored_count = store_catalog_in_db(models)

        typer.echo(f"✅ Stored {stored_count} models in openrouter_catalog collection")

    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def report(
    max_matches: int = typer.Option(20, "--max-matches", help="Maximum number of new models to show"),
    min_confidence: float = typer.Option(0.7, "--min-confidence", help="Minimum confidence threshold"),
    provider_filter: Optional[List[str]] = typer.Option(None, "--provider", "-p", help="Filter to specific providers"),
):
    """
    Generate discovery report showing new models to add.

    This command:
    1. Loads OpenRouter catalog from DB
    2. Compares to our models collection
    3. Uses LLM to match new models to direct providers
    4. Prints copy-paste commands to add them
    """
    typer.echo("Generating discovery report...")
    typer.echo()

    try:
        # Load catalogs
        openrouter_models = get_catalog_from_db()
        our_models = get_our_models_from_db()

        typer.echo(f"📊 OpenRouter catalog: {len(openrouter_models)} models")
        typer.echo(f"📊 Our models: {len(our_models)} models")
        typer.echo()

        # Match new models
        typer.echo("🔍 Matching new models to direct providers...")
        matches = asyncio.run(
            match_to_direct_providers(
                openrouter_models,
                our_models,
                batch_size=10,
                max_matches=max_matches,
            )
        )

        # Filter by confidence
        matches = [m for m in matches if m.confidence >= min_confidence]

        # Filter by provider if specified
        if provider_filter:
            matches = [m for m in matches if m.provider in provider_filter]

        if not matches:
            typer.echo("✅ No new models discovered (all OpenRouter models already covered)")
            return

        # Store matches in DB
        store_matches_in_db(matches)

        # Print summary
        typer.echo()
        typer.echo(f"🎉 Found {len(matches)} new models to add:")
        typer.echo()

        # Group by provider
        by_provider = {}
        for match in matches:
            by_provider.setdefault(match.provider, []).append(match)

        # Sort providers by count
        sorted_providers = sorted(by_provider.items(), key=lambda x: len(x[1]), reverse=True)

        # Print per-provider
        for provider, provider_matches in sorted_providers:
            typer.echo(f"=== {provider.upper()} ({len(provider_matches)} models) ===")
            typer.echo()

            for i, match in enumerate(provider_matches, 1):
                typer.echo(f"{i}. {match.openrouter_name}")
                typer.echo(f"   OpenRouter ID: {match.openrouter_id}")
                typer.echo(f"   → {match.provider}/{match.model_id}")
                typer.echo(f"   Confidence: {match.confidence:.2f}")
                typer.echo(f"   Reasoning: {match.reasoning}")
                typer.echo()
                plan = build_promotion_plan(match, our_models)
                if plan.warnings:
                    typer.echo("   Warnings:")
                    for warning in plan.warnings:
                        typer.echo(f"   - {warning}")
                    typer.echo()

                typer.echo("   Stage candidate with:")
                typer.echo()
                for line in format_promotion_command(plan).split("\n"):
                    typer.echo(f"   {line}")
                typer.echo()
                typer.echo()

    except Exception as e:
        logger.exception("Report generation failed")
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def stats():
    """
    Show statistics about the OpenRouter catalog.

    Useful for understanding the catalog size and match coverage.
    """
    try:
        openrouter_models = get_catalog_from_db()
        our_models = get_our_models_from_db()

        typer.echo("📊 Catalog Statistics")
        typer.echo()
        typer.echo(f"OpenRouter models: {len(openrouter_models)}")
        typer.echo(f"Our enabled models: {len([m for m in our_models if m.get('enabled')])}")
        typer.echo(f"Our deprecated models: {len([m for m in our_models if m.get('deprecated')])}")
        typer.echo(f"Our disabled models: {len([m for m in our_models if not m.get('enabled')])}")
        typer.echo()

        # Count matched models
        matched_count = len([m for m in openrouter_models if m.get("matched_provider")])
        typer.echo(f"Matched models: {matched_count}")
        typer.echo(f"Unmatched models: {len(openrouter_models) - matched_count}")
        typer.echo()

        # Breakdown by provider (our models)
        our_by_provider = {}
        for model in our_models:
            if model.get("enabled"):
                provider = model.get("provider", "unknown")
                our_by_provider[provider] = our_by_provider.get(provider, 0) + 1

        typer.echo("Our models by provider:")
        for provider in sorted(our_by_provider.keys()):
            typer.echo(f"  {provider}: {our_by_provider[provider]}")

    except Exception as e:
        logger.exception("Stats generation failed")
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
