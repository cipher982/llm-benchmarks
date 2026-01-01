#!/usr/bin/env python3
"""
Daily LLM Benchmark Health Check

Queries MongoDB for recent errors, sends to OpenAI for analysis,
and emails a summary. Designed to run daily via systemd timer.

Usage:
    # Set environment variables
    export MONGODB_URI="mongodb://..."
    export OPENAI_API_KEY="sk-..."
    export NOTIFY_EMAIL="david010@gmail.com"

    python daily-health-check.py
    python daily-health-check.py --days 7  # Look back further
    python daily-health-check.py --dry-run  # Print but don't email
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import dotenv
import httpx
from pymongo import MongoClient

# Load environment variables
dotenv.load_dotenv()


# --- Configuration ---

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "llm-bench")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NOTIFY_EMAIL = os.getenv("NOTIFY_EMAIL", "david010@gmail.com")

# Collections
ERRORS_COLLECTION = os.getenv("MONGODB_COLLECTION_ERRORS", "errors_cloud")
ROLLUPS_COLLECTION = os.getenv("MONGODB_COLLECTION_ERROR_ROLLUPS", "error_rollups")
METRICS_COLLECTION = os.getenv("MONGODB_COLLECTION_CLOUD", "metrics_cloud_v2")


# --- Data Collection ---

@dataclass
class HealthData:
    """Aggregated health metrics for the report."""
    period_hours: int
    total_errors: int
    total_successes: int
    errors_by_kind: dict[str, int]
    top_failing_models: list[dict[str, Any]]
    new_error_fingerprints: list[dict[str, Any]]
    models_with_only_errors: list[dict[str, Any]]  # No successes in period
    provider_summary: dict[str, dict[str, int]]


def collect_health_data(client: MongoClient, hours: int = 24) -> HealthData:
    """Query MongoDB and aggregate health metrics."""
    db = client[MONGODB_DB]
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)

    # 1. Count errors by kind in period
    errors_pipeline = [
        {"$match": {"ts": {"$gte": since}}},
        {"$group": {
            "_id": "$error_kind",
            "count": {"$sum": 1}
        }}
    ]
    errors_by_kind = {}
    for doc in db[ERRORS_COLLECTION].aggregate(errors_pipeline):
        kind = doc["_id"] or "unknown"
        errors_by_kind[kind] = doc["count"]

    total_errors = sum(errors_by_kind.values())

    # 2. Count successes in period
    success_count = db[METRICS_COLLECTION].count_documents({
        "$or": [
            {"gen_ts": {"$gte": since}},
            {"run_ts": {"$gte": since}}
        ]
    })

    # 3. Top failing models (by error count)
    top_failing_pipeline = [
        {"$match": {"ts": {"$gte": since}}},
        {"$sort": {"ts": -1}},  # Sort first so $last gets the most recent message
        {"$group": {
            "_id": {"provider": "$provider", "model": "$model_name"},
            "error_count": {"$sum": 1},
            "kinds": {"$addToSet": "$error_kind"},
            "last_error": {"$max": "$ts"},
            "sample_message": {"$last": "$normalized_message"}
        }},
        {"$sort": {"error_count": -1}},
        {"$limit": 15}
    ]
    top_failing = []
    for doc in db[ERRORS_COLLECTION].aggregate(top_failing_pipeline):
        top_failing.append({
            "provider": doc["_id"]["provider"],
            "model": doc["_id"]["model"],
            "error_count": doc["error_count"],
            "error_kinds": doc["kinds"],
            "last_error": doc["last_error"].isoformat() if doc["last_error"] else None,
            "sample_message": (doc.get("sample_message") or "")[:200]
        })

    # 4. New error fingerprints (first_seen in period)
    new_fingerprints_pipeline = [
        {"$match": {"first_seen": {"$gte": since}}},
        {"$sort": {"first_seen": -1}},
        {"$limit": 20},
        {"$project": {
            "_id": 0,
            "provider": 1,
            "model_name": 1,
            "error_kind": 1,
            "count": 1,
            "first_seen": 1,
            "sample_messages": {"$slice": ["$sample_messages", 2]}
        }}
    ]
    new_fingerprints = list(db[ROLLUPS_COLLECTION].aggregate(new_fingerprints_pipeline))
    for fp in new_fingerprints:
        if fp.get("first_seen"):
            fp["first_seen"] = fp["first_seen"].isoformat()

    # 5. Models with errors but no successes in period (potential dead models)
    # Get all models that had errors
    models_with_errors = db[ERRORS_COLLECTION].distinct(
        "model_name",
        {"ts": {"$gte": since}}
    )

    # Get all models that had successes
    models_with_success = set()
    for field in ["gen_ts", "run_ts"]:
        models_with_success.update(
            db[METRICS_COLLECTION].distinct(
                "model_name",
                {field: {"$gte": since}}
            )
        )

    # Find models with only errors
    error_only_models = []
    for model in models_with_errors:
        if model not in models_with_success:
            # Get error details for this model
            sample = db[ERRORS_COLLECTION].find_one(
                {"model_name": model, "ts": {"$gte": since}},
                {"provider": 1, "error_kind": 1, "normalized_message": 1}
            )
            if sample:
                error_only_models.append({
                    "provider": sample.get("provider"),
                    "model": model,
                    "error_kind": sample.get("error_kind"),
                    "sample": (sample.get("normalized_message") or "")[:150]
                })

    # 6. Provider summary
    provider_pipeline = [
        {"$match": {"ts": {"$gte": since}}},
        {"$group": {
            "_id": "$provider",
            "errors": {"$sum": 1}
        }}
    ]
    provider_errors = {doc["_id"]: doc["errors"] for doc in db[ERRORS_COLLECTION].aggregate(provider_pipeline)}

    provider_success_pipeline = [
        {"$match": {"$or": [{"gen_ts": {"$gte": since}}, {"run_ts": {"$gte": since}}]}},
        {"$group": {
            "_id": "$provider",
            "successes": {"$sum": 1}
        }}
    ]
    provider_successes = {doc["_id"]: doc["successes"] for doc in db[METRICS_COLLECTION].aggregate(provider_success_pipeline)}

    all_providers = set(provider_errors.keys()) | set(provider_successes.keys())
    provider_summary = {}
    for p in all_providers:
        provider_summary[p] = {
            "errors": provider_errors.get(p, 0),
            "successes": provider_successes.get(p, 0)
        }

    return HealthData(
        period_hours=hours,
        total_errors=total_errors,
        total_successes=success_count,
        errors_by_kind=errors_by_kind,
        top_failing_models=top_failing,
        new_error_fingerprints=new_fingerprints,
        models_with_only_errors=error_only_models,
        provider_summary=provider_summary
    )


def format_health_data(data: HealthData) -> str:
    """Format health data as a summary string for the LLM."""
    lines = [
        f"## LLM Benchmark Health Report",
        f"Period: Last {data.period_hours} hours",
        f"Total: {data.total_successes} successes, {data.total_errors} errors",
        "",
        "### Errors by Kind",
    ]

    for kind, count in sorted(data.errors_by_kind.items(), key=lambda x: -x[1]):
        lines.append(f"  {kind}: {count}")

    lines.append("")
    lines.append("### Provider Summary")
    for provider, stats in sorted(data.provider_summary.items()):
        rate = stats["errors"] / (stats["errors"] + stats["successes"]) * 100 if (stats["errors"] + stats["successes"]) > 0 else 0
        lines.append(f"  {provider}: {stats['successes']} ok, {stats['errors']} err ({rate:.1f}% error rate)")

    if data.models_with_only_errors:
        lines.append("")
        lines.append("### Models with ONLY Errors (no successes in period)")
        for m in data.models_with_only_errors[:10]:
            lines.append(f"  {m['provider']}:{m['model']} - {m['error_kind']} - {m['sample'][:80]}")

    if data.new_error_fingerprints:
        lines.append("")
        lines.append("### New Error Patterns (first seen in period)")
        for fp in data.new_error_fingerprints[:10]:
            samples = fp.get("sample_messages", [])
            sample_str = samples[0][:80] if samples else "no message"
            lines.append(f"  {fp.get('provider')}:{fp.get('model_name')} - {fp.get('error_kind')} ({fp.get('count')} times)")
            lines.append(f"    → {sample_str}")

    if data.top_failing_models:
        lines.append("")
        lines.append("### Top Failing Models")
        for m in data.top_failing_models[:10]:
            lines.append(f"  {m['provider']}:{m['model']} - {m['error_count']} errors - kinds: {m['error_kinds']}")

    return "\n".join(lines)


# --- OpenAI Analysis ---

SYSTEM_PROMPT = """You are a DevOps engineer monitoring an LLM benchmarking service. Your job is to analyze daily health reports and provide actionable insights.

## CONTEXT
This service benchmarks multiple LLM providers (OpenAI, Anthropic, Bedrock, Vertex, etc.) every few hours. It tracks successes and errors, classifying errors by type:
- auth: API key / authentication issues (401/403)
- billing: Payment / quota issues (402)
- rate_limit: Rate limiting (429)
- hard_model: Model not found / deprecated (404, "does not exist")
- hard_capability: Wrong endpoint or capability mismatch ("use responses API", "not a chat model")
- transient_provider: Server errors (5xx)
- network: Timeouts, connection issues
- unknown: Unclassified

## YOUR TASK
Analyze the health report and provide:
1. Overall assessment: Is the system healthy? Any concerns?
2. Models likely deprecated: Models with hard_model errors and no successes
3. Code fixes needed: Models with hard_capability errors (these need code updates, not disabling)
4. Provider issues: Any provider-wide problems (auth, billing, rate limits)?
5. Recommended actions: What should the human operator do?

## GUIDELINES
- Be concise but specific
- Distinguish between model-level issues and provider-level issues
- hard_capability errors mean OUR code needs updating (new API version, endpoint change)
- hard_model errors mean the model is likely deprecated/removed by the provider
- Transient errors are usually fine unless they're persistent
- If everything looks normal, say so briefly"""

USER_PROMPT_TEMPLATE = """Here is today's LLM benchmark health report:

{health_data}

Please analyze this report and provide your assessment. What needs attention? What actions should I take?"""


def analyze_with_openai(health_summary: str) -> Optional[str]:
    """Send health data to OpenAI for analysis."""
    if not OPENAI_API_KEY:
        return None

    request_body = {
        "model": "gpt-4o-mini",  # Fast and cheap for daily summaries
        "max_output_tokens": 2000,  # Cap response size
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(health_data=health_summary)}
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "health_assessment",
                "description": "Analysis of LLM benchmark health",
                "schema": {
                    "type": "object",
                    "properties": {
                        "overall_status": {
                            "type": "string",
                            "enum": ["healthy", "warning", "critical"],
                            "description": "Overall system health"
                        },
                        "summary": {
                            "type": "string",
                            "description": "2-3 sentence executive summary"
                        },
                        "models_likely_deprecated": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of 'provider:model' that appear deprecated"
                        },
                        "code_fixes_needed": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of 'provider:model' needing code updates"
                        },
                        "provider_issues": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Provider-wide issues (e.g., 'openai: auth errors')"
                        },
                        "recommended_actions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific actions the operator should take"
                        }
                    },
                    "required": ["overall_status", "summary", "models_likely_deprecated", "code_fixes_needed", "provider_issues", "recommended_actions"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    }

    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=request_body
            )
            response.raise_for_status()

            result = response.json()
            # Extract the message content from the Responses API format
            for output in result.get("output", []):
                if output.get("type") == "message":
                    for content in output.get("content", []):
                        if content.get("type") == "output_text":
                            return content.get("text")

            return None
    except Exception as e:
        print(f"OpenAI API error: {e}", file=sys.stderr)
        return None


# --- Email ---

def format_email_body(health_data: HealthData, ai_analysis: Optional[dict], operator_results: Optional[dict] = None, llm_usage: Optional[dict] = None) -> str:
    """Format the full email body."""
    lines = []

    # Operator section (if available) - goes FIRST
    if operator_results:
        auto_executed = operator_results.get("auto_executed", [])
        manual_review = operator_results.get("manual_review", [])

        # Auto-executed actions
        if auto_executed:
            lines.append("=" * 60)
            lines.append("AUTO-EXECUTED ACTIONS")
            lines.append("=" * 60)
            lines.append("")
            lines.append(f"Disabled {len(auto_executed)} models (high confidence ≥0.95):")
            lines.append("")

            for i, decision in enumerate(auto_executed, 1):
                lines.append(f"{i}. {decision.provider}/{decision.model_id}")
                lines.append(f"   Reason: {decision.reasoning}")
                lines.append(f"   Confidence: {decision.confidence:.2f}")
                if decision.executed_at:
                    lines.append(f"   Executed: {decision.executed_at.strftime('%Y-%m-%d %H:%M UTC')}")
                lines.append("")

        # Manual review suggestions
        if manual_review:
            lines.append("=" * 60)
            lines.append("SUGGESTIONS FOR REVIEW")
            lines.append("=" * 60)
            lines.append("")
            lines.append(f"{len(manual_review)} models flagged for monitoring:")
            lines.append("")

            # Group by action
            by_action = {"disable": [], "monitor": []}
            for d in manual_review:
                by_action.get(d.action, []).append(d)

            # Show disable suggestions first (sorted by confidence)
            if by_action["disable"]:
                lines.append("DISABLE CANDIDATES:")
                for i, decision in enumerate(sorted(by_action["disable"], key=lambda d: d.confidence, reverse=True)[:10], 1):
                    lines.append(f"{i}. {decision.provider}/{decision.model_id}")
                    lines.append(f"   Action: Disable model")
                    lines.append(f"   Reason: {decision.reasoning}")
                    lines.append(f"   Confidence: {decision.confidence:.2f}")
                    lines.append("")
                    lines.append("   Approve with:")
                    lines.append(f'   mongosh "$MONGODB_URI" --eval \'')
                    lines.append(f'   db.models.updateOne(')
                    lines.append(f'     {{provider: "{decision.provider}", model_id: "{decision.model_id}"}},')
                    lines.append(f'     {{$set: {{enabled: false, disabled_reason: "Operator: {decision.reasoning[:80]}"}},')
                    lines.append(f'      disabled_at: new Date()}}')
                    lines.append(f'   )\'')
                    lines.append("")

            # Show monitor suggestions
            if by_action["monitor"]:
                lines.append("MONITOR CANDIDATES:")
                for i, decision in enumerate(sorted(by_action["monitor"], key=lambda d: d.confidence, reverse=True)[:5], 1):
                    lines.append(f"{i}. {decision.provider}/{decision.model_id}")
                    lines.append(f"   Reason: {decision.reasoning}")
                    lines.append(f"   Confidence: {decision.confidence:.2f}")
                    lines.append("")

        lines.append("")

    # AI analysis section (if available)
    if ai_analysis:
        status_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "🚨"}.get(ai_analysis.get("overall_status"), "❓")
        lines.append(f"{status_emoji} Status: {ai_analysis.get('overall_status', 'unknown').upper()}")
        lines.append("")
        lines.append(ai_analysis.get("summary", ""))
        lines.append("")

        if ai_analysis.get("models_likely_deprecated"):
            lines.append("📦 Models Likely Deprecated:")
            for m in ai_analysis["models_likely_deprecated"]:
                lines.append(f"  • {m}")
            lines.append("")

        if ai_analysis.get("code_fixes_needed"):
            lines.append("🔧 Code Fixes Needed:")
            for m in ai_analysis["code_fixes_needed"]:
                lines.append(f"  • {m}")
            lines.append("")

        if ai_analysis.get("provider_issues"):
            lines.append("🏢 Provider Issues:")
            for issue in ai_analysis["provider_issues"]:
                lines.append(f"  • {issue}")
            lines.append("")

        if ai_analysis.get("recommended_actions"):
            lines.append("📋 Recommended Actions:")
            for action in ai_analysis["recommended_actions"]:
                lines.append(f"  • {action}")
            lines.append("")
    else:
        lines.append("⚠️ AI analysis unavailable (check OPENAI_API_KEY)")
        lines.append("")

    # Raw stats
    lines.append("─" * 50)
    lines.append("RAW METRICS")
    lines.append("─" * 50)
    lines.append(f"Period: {health_data.period_hours}h")
    lines.append(f"Successes: {health_data.total_successes}")
    lines.append(f"Errors: {health_data.total_errors}")
    lines.append("")

    lines.append("Errors by Kind:")
    for kind, count in sorted(health_data.errors_by_kind.items(), key=lambda x: -x[1]):
        lines.append(f"  {kind}: {count}")
    lines.append("")

    lines.append("Provider Summary:")
    for provider, stats in sorted(health_data.provider_summary.items()):
        total = stats["errors"] + stats["successes"]
        rate = stats["errors"] / total * 100 if total > 0 else 0
        lines.append(f"  {provider}: {stats['successes']} ok / {stats['errors']} err ({rate:.0f}%)")

    if health_data.models_with_only_errors:
        lines.append("")
        lines.append("Models with ONLY Errors:")
        for m in health_data.models_with_only_errors[:5]:
            lines.append(f"  {m['provider']}:{m['model']} ({m['error_kind']})")

    # LLM Usage section
    if llm_usage and llm_usage.get("api_calls", 0) > 0:
        lines.append("")
        lines.append("─" * 50)
        lines.append("LLM CLASSIFICATION USAGE")
        lines.append("─" * 50)
        lines.append(f"Model: {llm_usage.get('model', 'N/A')}")
        lines.append(f"API Calls: {llm_usage.get('api_calls', 0)}")
        lines.append(f"Input Tokens: {llm_usage.get('input_tokens', 0):,}")
        lines.append(f"Output Tokens: {llm_usage.get('output_tokens', 0):,}")
        reasoning = llm_usage.get('reasoning_tokens', 0)
        if reasoning > 0:
            lines.append(f"Reasoning Tokens: {reasoning:,}")
        lines.append(f"Total Tokens: {llm_usage.get('total_tokens', 0):,}")
        cost = llm_usage.get('cost_estimate_usd', 0)
        lines.append(f"Est. Cost: ${cost:.4f}")
        rollups_classified = llm_usage.get('rollups_classified', 0)
        if rollups_classified:
            lines.append(f"Rollups Classified: {rollups_classified}")

    return "\n".join(lines)


def send_email(subject: str, body: str, dry_run: bool = False) -> bool:
    """Send email via msmtp."""
    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN - Would send email:")
        print(f"To: {NOTIFY_EMAIL}")
        print(f"Subject: {subject}")
        print(f"{'='*60}")
        print(body)
        print(f"{'='*60}\n")
        return True

    message = f"Subject: {subject}\n\n{body}"

    try:
        result = subprocess.run(
            ["msmtp", NOTIFY_EMAIL],
            input=message.encode(),
            capture_output=True,
            timeout=30
        )
        if result.returncode != 0:
            print(f"msmtp error: {result.stderr.decode()}", file=sys.stderr)
            return False
        return True
    except FileNotFoundError:
        print("msmtp not installed - cannot send email", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Email error: {e}", file=sys.stderr)
        return False


# --- Main ---

async def classify_errors_async():
    """Run LLM error classification on unclassified rollups."""
    # Import here to avoid circular dependencies
    try:
        # Add parent directory to path to import from api module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from api.llm_bench.ops.llm_error_classifier import classify_unclassified_rollups

        print("Classifying unclassified errors...")
        results = await classify_unclassified_rollups(batch_size=50, max_rollups=200)
        if results.get("updated", 0) > 0:
            print(f"  Classified {results['updated']} new error patterns")
        else:
            print("  No new errors to classify")
        return results
    except Exception as e:
        print(f"Warning: Error classification failed: {e}", file=sys.stderr)
        return None


async def run_operator_async(provider_filter=None):
    """Run AI operator analysis and return decisions."""
    try:
        # Add parent directory to path to import from api module
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from api.llm_bench.operator.engine import generate_decisions
        from api.llm_bench.operator.io import load_snapshots, store_decisions
        from api.llm_bench.operator.actions import execute_decisions, should_auto_execute

        print("Running AI operator analysis...")

        # Load lifecycle snapshots
        snapshots = load_snapshots(provider_filter=provider_filter)
        if not snapshots:
            print("  No models found for operator analysis")
            return None

        print(f"  Analyzing {len(snapshots)} models...")

        # Generate decisions
        decisions = await generate_decisions(snapshots)

        # Store all decisions in model_status
        stored_count = store_decisions(decisions)
        print(f"  Stored {stored_count} decisions in model_status")

        # Separate auto-executable from manual review
        auto_exec = [d for d in decisions if should_auto_execute(d)]
        manual_review = [d for d in decisions if not should_auto_execute(d) and d.action != "ignore"]

        print(f"  Auto-executable: {len(auto_exec)}")
        print(f"  Manual review: {len(manual_review)}")

        # Execute auto-executable decisions
        execution_stats = None
        if auto_exec:
            print(f"  Executing {len(auto_exec)} high-confidence decisions...")
            execution_stats = execute_decisions(auto_exec, auto_execute=True, dry_run=False)
            print(f"    Executed: {execution_stats['executed']}, Failed: {execution_stats['failed']}")

            # Update in-memory decisions with executed_at timestamps from DB
            # This ensures the email shows correct execution times
            from pymongo import MongoClient
            uri = os.getenv("MONGODB_URI")
            if uri:
                client = MongoClient(uri)
                try:
                    db_name = os.getenv("MONGODB_DB", "llm-bench")
                    collection = client[db_name]["model_status"]

                    for decision in auto_exec:
                        # Read back executed_at from DB
                        doc = collection.find_one(
                            {"provider": decision.provider, "model_id": decision.model_id},
                            {"operator_decision.executed_at": 1}
                        )
                        if doc and doc.get("operator_decision", {}).get("executed_at"):
                            decision.executed_at = doc["operator_decision"]["executed_at"]
                finally:
                    client.close()

        return {
            "total_analyzed": len(snapshots),
            "total_decisions": len(decisions),
            "auto_executed": auto_exec,
            "manual_review": manual_review,
            "execution_stats": execution_stats
        }

    except Exception as e:
        print(f"Warning: Operator analysis failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Daily LLM benchmark health check")
    parser.add_argument("--days", type=float, default=1, help="Look back this many days (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="Print report but don't email")
    parser.add_argument("--skip-classification", action="store_true", help="Skip LLM error classification step")
    parser.add_argument("--skip-operator", action="store_true", help="Skip AI operator analysis step")
    parser.add_argument("--operator-provider", type=str, action="append", help="Filter operator to specific providers (can repeat)")
    args = parser.parse_args()

    hours = max(1, int(args.days * 24))  # Minimum 1 hour to avoid empty queries

    # Validate environment
    if not MONGODB_URI:
        print("ERROR: MONGODB_URI not set", file=sys.stderr)
        sys.exit(1)

    # Step 1: Classify unclassified errors (unless skipped)
    classification_results = None
    if not args.skip_classification:
        classification_results = asyncio.run(classify_errors_async())

    # Step 2: Run AI operator analysis (unless skipped)
    operator_results = None
    if not args.skip_operator:
        operator_results = asyncio.run(run_operator_async(provider_filter=args.operator_provider))

    # Step 3: Collect data
    print(f"Collecting health data for last {hours} hours...")
    client = MongoClient(MONGODB_URI)
    try:
        health_data = collect_health_data(client, hours=hours)
    finally:
        client.close()

    print(f"Found {health_data.total_successes} successes, {health_data.total_errors} errors")

    # Format for AI
    health_summary = format_health_data(health_data)

    # AI analysis
    ai_analysis = None
    if OPENAI_API_KEY:
        print("Sending to OpenAI for analysis...")
        ai_response = analyze_with_openai(health_summary)
        if ai_response:
            try:
                ai_analysis = json.loads(ai_response)
                print(f"AI assessment: {ai_analysis.get('overall_status', 'unknown')}")
            except json.JSONDecodeError:
                print(f"Failed to parse AI response: {ai_response[:200]}", file=sys.stderr)
    else:
        print("OPENAI_API_KEY not set - skipping AI analysis")

    # Determine email subject tag
    operator_tag = ""
    if operator_results:
        auto_exec_count = len(operator_results.get("auto_executed", []))
        manual_count = len(operator_results.get("manual_review", []))
        if auto_exec_count > 0:
            operator_tag = f"[OPERATED] {auto_exec_count} auto-actions, {manual_count} suggestions - "

    if ai_analysis:
        status = ai_analysis.get("overall_status", "unknown")
        if status == "critical":
            tag = "[CRITICAL]"
        elif status == "warning":
            tag = "[URGENT]"
        else:
            tag = "[INFO]"
    else:
        # No AI, use error count heuristic
        if health_data.total_errors > 100:
            tag = "[URGENT]"
        else:
            tag = "[INFO]"

    # Format and send email
    subject = f"{operator_tag}{tag} LLM Benchmarks Daily Health - {datetime.now().strftime('%Y-%m-%d')}"

    # Extract LLM usage from classification results
    llm_usage = None
    if classification_results:
        llm_usage = classification_results.get("llm_usage")
        if llm_usage:
            llm_usage["rollups_classified"] = classification_results.get("updated", 0)

    body = format_email_body(health_data, ai_analysis, operator_results=operator_results, llm_usage=llm_usage)

    if send_email(subject, body, dry_run=args.dry_run):
        print(f"Email sent: {subject}")
    else:
        print("Failed to send email", file=sys.stderr)
        # Print to stdout as fallback
        print("\n" + body)
        sys.exit(1)


if __name__ == "__main__":
    main()
