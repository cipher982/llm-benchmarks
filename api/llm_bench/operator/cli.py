"""
CLI interface for AI Operator.

Usage:
    uv run python -m api.llm_bench.operator.cli analyze [--dry-run] [--write]
    uv run python -m api.llm_bench.operator.cli pending
    uv run python -m api.llm_bench.operator.cli execute [--dry-run]
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import dotenv
import typer

from .actions import execute_decisions
from .engine import OperatorDecision, generate_decisions
from .io import load_pending_decisions, load_snapshots, store_decisions


# Setup
app = typer.Typer(help="AI Operator for model lifecycle management")
dotenv.load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def format_decision_summary(decisions: List[OperatorDecision]) -> dict:
    """Format decisions into summary statistics."""
    by_action = {"disable": [], "monitor": [], "ignore": []}

    for decision in decisions:
        by_action[decision.action].append(decision)

    # Sort each category by confidence (highest first)
    for action in by_action:
        by_action[action].sort(key=lambda d: d.confidence, reverse=True)

    return {
        "total": len(decisions),
        "disable": len(by_action["disable"]),
        "monitor": len(by_action["monitor"]),
        "ignore": len(by_action["ignore"]),
        "by_action": by_action,
        "high_confidence_disable": [
            d for d in by_action["disable"] if d.confidence >= 0.90
        ]
    }


def print_decision_table(decisions: List[OperatorDecision]) -> None:
    """Print decisions in a formatted table."""
    if not decisions:
        typer.echo("No decisions to display")
        return

    # Headers
    headers = ["ACTION", "CONF", "PROVIDER", "MODEL_ID", "REASONING"]
    col_widths = [len(h) for h in headers]

    # Build table data
    rows = []
    for decision in decisions:
        row = [
            decision.action.upper(),
            f"{decision.confidence:.2f}",
            decision.provider,
            decision.model_id,
            decision.reasoning[:80] + "..." if len(decision.reasoning) > 80 else decision.reasoning
        ]
        rows.append(row)

        # Update column widths
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    # Print header
    header_row = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    typer.echo(header_row)
    typer.echo("  ".join("-" * w for w in col_widths))

    # Print rows
    for row in rows:
        typer.echo("  ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)))


@app.command()
def analyze(
    provider: List[str] = typer.Option(None, "--provider", "-p", help="Filter to specific providers"),
    dry_run: bool = typer.Option(True, help="Run without writing to database (default: True)"),
    write: bool = typer.Option(False, help="Write decisions to database"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save decisions to JSON file"),
    json_output: bool = typer.Option(False, "--json", help="Output JSON instead of table"),
) -> None:
    """
    Analyze model health and generate AI operator decisions.

    This command loads lifecycle snapshots, uses LLM reasoning to make decisions,
    and optionally writes them to the database.
    """
    typer.echo("Loading lifecycle snapshots...")
    snapshots = load_snapshots(provider_filter=provider or None)

    if not snapshots:
        typer.echo("No models found matching filters")
        raise typer.Exit(0)

    typer.echo(f"Analyzing {len(snapshots)} models with LLM reasoning...")

    # Generate decisions (async)
    decisions = asyncio.run(generate_decisions(snapshots))

    # Format summary
    summary = format_decision_summary(decisions)

    if json_output:
        # Output as JSON
        output_data = {
            "summary": {
                "total": summary["total"],
                "disable": summary["disable"],
                "monitor": summary["monitor"],
                "ignore": summary["ignore"],
                "high_confidence_disable": len(summary["high_confidence_disable"])
            },
            "decisions": [
                {
                    "provider": d.provider,
                    "model_id": d.model_id,
                    "action": d.action,
                    "confidence": d.confidence,
                    "reasoning": d.reasoning,
                    "suggested_by": d.suggested_by,
                    "suggested_at": d.suggested_at.isoformat()
                }
                for d in decisions
            ]
        }
        typer.echo(json.dumps(output_data, indent=2))
    else:
        # Print summary
        typer.echo("\n=== Analysis Summary ===")
        typer.echo(f"Total models analyzed: {summary['total']}")
        typer.echo(f"  Disable: {summary['disable']} (high confidence: {len(summary['high_confidence_disable'])})")
        typer.echo(f"  Monitor: {summary['monitor']}")
        typer.echo(f"  Ignore: {summary['ignore']}")

        # Print high-confidence disable decisions
        if summary["high_confidence_disable"]:
            typer.echo("\n=== High-Confidence Disable Recommendations ===")
            print_decision_table(summary["high_confidence_disable"])

        # Print monitor decisions
        if summary["by_action"]["monitor"]:
            typer.echo("\n=== Monitor Recommendations ===")
            print_decision_table(summary["by_action"]["monitor"][:10])  # Top 10

    # Save to file if requested
    if output:
        output_data = [
            {
                "provider": d.provider,
                "model_id": d.model_id,
                "action": d.action,
                "confidence": d.confidence,
                "reasoning": d.reasoning,
                "suggested_by": d.suggested_by,
                "suggested_at": d.suggested_at.isoformat()
            }
            for d in decisions
        ]
        output.write_text(json.dumps(output_data, indent=2))
        typer.echo(f"\nDecisions saved to {output}")

    # Write to database if requested
    if write and not dry_run:
        typer.echo("\nWriting decisions to model_status collection...")
        count = store_decisions(decisions)
        typer.echo(f"Updated {count} documents in model_status")
    elif write and dry_run:
        typer.echo("\n⚠️  Cannot write with --dry-run. Use --no-dry-run --write to persist.")
    elif dry_run:
        typer.echo("\n[DRY-RUN] No changes written to database. Use --no-dry-run --write to persist.")


@app.command()
def pending(
    json_output: bool = typer.Option(False, "--json", help="Output JSON instead of table"),
) -> None:
    """
    Show pending operator decisions awaiting human review.
    """
    typer.echo("Loading pending decisions from model_status collection...")
    docs = load_pending_decisions()

    if not docs:
        typer.echo("No pending decisions found")
        raise typer.Exit(0)

    # Convert to OperatorDecision objects for formatting
    decisions = []
    for doc in docs:
        op_dec = doc.get("operator_decision", {})
        decision = OperatorDecision(
            provider=doc["provider"],
            model_id=doc["model_id"],
            action=op_dec.get("action", "unknown"),
            confidence=op_dec.get("confidence", 0.0),
            reasoning=op_dec.get("reasoning", ""),
            suggested_at=op_dec.get("suggested_at", datetime.now()),
            suggested_by=op_dec.get("suggested_by", "unknown"),
            status=op_dec.get("status", "pending")
        )
        decisions.append(decision)

    if json_output:
        output_data = [
            {
                "provider": d.provider,
                "model_id": d.model_id,
                "action": d.action,
                "confidence": d.confidence,
                "reasoning": d.reasoning,
                "suggested_by": d.suggested_by,
                "suggested_at": d.suggested_at.isoformat()
            }
            for d in decisions
        ]
        typer.echo(json.dumps(output_data, indent=2))
    else:
        typer.echo(f"\n=== {len(decisions)} Pending Decisions ===\n")
        print_decision_table(decisions)


@app.command()
def execute(
    dry_run: bool = typer.Option(True, help="Run without executing actions (default: True)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """
    Execute pending auto-executable decisions.

    Only high-confidence (≥0.95) disable actions are auto-executed.
    """
    typer.echo("Loading pending decisions...")
    docs = load_pending_decisions()

    if not docs:
        typer.echo("No pending decisions found")
        raise typer.Exit(0)

    # Convert to OperatorDecision objects
    decisions = []
    for doc in docs:
        op_dec = doc.get("operator_decision", {})
        decision = OperatorDecision(
            provider=doc["provider"],
            model_id=doc["model_id"],
            action=op_dec.get("action", "unknown"),
            confidence=op_dec.get("confidence", 0.0),
            reasoning=op_dec.get("reasoning", ""),
            suggested_at=op_dec.get("suggested_at", datetime.now()),
            suggested_by=op_dec.get("suggested_by", "unknown"),
            status=op_dec.get("status", "pending")
        )
        decisions.append(decision)

    # Load snapshots for signal-based auto-exec checks
    typer.echo("Loading snapshots for signal verification...")
    snapshots = load_snapshots(provider_filter=None)

    # Execute decisions (with signal-based filtering)
    typer.echo("Checking auto-execution eligibility...")
    stats = execute_decisions(
        decisions,
        snapshots=snapshots,
        auto_execute=True,
        dry_run=True  # First do a dry run to see what would execute
    )

    auto_executable_count = stats["executed"]
    if auto_executable_count == 0:
        typer.echo(f"Found {len(decisions)} pending decisions, but none are auto-executable")
        typer.echo("(Requires: hard errors 48+ hrs, no recent success, high confidence)")
        raise typer.Exit(0)

    typer.echo(f"\n{auto_executable_count} decisions eligible for auto-execution")

    if not dry_run and not yes:
        confirm = typer.confirm("\nExecute these decisions?")
        if not confirm:
            typer.echo("Cancelled")
            raise typer.Exit(0)

    # Execute decisions for real
    typer.echo("\nExecuting decisions...")
    stats = execute_decisions(
        decisions,
        snapshots=snapshots,
        auto_execute=True,
        dry_run=dry_run
    )

    typer.echo(f"\nExecution complete:")
    typer.echo(f"  Executed: {stats['executed']}")
    typer.echo(f"  Skipped: {stats['skipped']}")
    typer.echo(f"  Failed: {stats['failed']}")

    if dry_run:
        typer.echo("\n[DRY-RUN] No changes made. Use --no-dry-run to execute.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
