"""
Core AI Operator engine using LLM reasoning for model lifecycle decisions.

Falls back to classifier.py if LLM calls fail.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import httpx

from ..model_lifecycle.classifier import LifecycleStatus, classify_snapshot
from ..model_lifecycle.collector import LifecycleSnapshot


logger = logging.getLogger(__name__)


# Configuration
OPENAI_MODEL = "gpt-5.2"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


@dataclass(slots=True)
class OperatorDecision:
    """Decision made by the AI operator."""
    provider: str
    model_id: str
    action: str  # disable, monitor, ignore
    confidence: float
    reasoning: str
    suggested_at: datetime
    suggested_by: str
    status: str = "pending"  # pending, approved, rejected, auto_executed
    executed_at: Optional[datetime] = None


@dataclass(slots=True)
class Situation:
    """Grouped models sharing similar error patterns."""
    situation_id: str
    provider: str
    error_kind: str  # hard, auth, rate, other
    model_count: int
    model_ids: List[str]
    snapshots: List[LifecycleSnapshot]
    oldest_error: Optional[datetime]
    newest_error: Optional[datetime]
    sample_messages: List[str]


def format_snapshot_context(snapshot: LifecycleSnapshot) -> str:
    """Format snapshot data for LLM context."""
    lines = [
        f"Provider: {snapshot.provider}",
        f"Model ID: {snapshot.model_id}",
        f"Display Name: {snapshot.display_name}",
        "",
        "=== Success Metrics ===",
    ]

    if snapshot.successes.last_success:
        lines.append(f"Last Success: {snapshot.successes.last_success.isoformat()}")
    else:
        lines.append("Last Success: Never")

    lines.extend([
        f"Successes (7d): {snapshot.successes.successes_7d}",
        f"Successes (30d): {snapshot.successes.successes_30d}",
        f"Successes (120d): {snapshot.successes.successes_120d}",
        "",
        "=== Error Metrics ===",
    ])

    if snapshot.errors.last_error:
        lines.append(f"Last Error: {snapshot.errors.last_error.isoformat()}")
    else:
        lines.append("Last Error: None")

    lines.extend([
        f"Errors (7d): {snapshot.errors.errors_7d}",
        f"Errors (30d): {snapshot.errors.errors_30d}",
        f"Hard Failures (7d): {snapshot.errors.hard_failures_7d}",
        f"Hard Failures (30d): {snapshot.errors.hard_failures_30d}",
        "",
        "=== Recent Error Messages ===",
    ])

    if snapshot.errors.recent_messages:
        for i, msg in enumerate(snapshot.errors.recent_messages[:5], 1):
            ts = msg.timestamp.isoformat() if msg.timestamp else "Unknown"
            lines.append(f"{i}. [{msg.kind}] ({ts}): {msg.message[:200]}")
    else:
        lines.append("No recent errors")

    lines.extend([
        "",
        "=== Metadata ===",
        f"Enabled: {snapshot.metadata.enabled}",
        f"Deprecated: {snapshot.metadata.deprecated}",
    ])

    if snapshot.metadata.deprecation_date:
        lines.append(f"Deprecation Date: {snapshot.metadata.deprecation_date.isoformat()}")

    if snapshot.metadata.successor_model:
        lines.append(f"Successor Model: {snapshot.metadata.successor_model}")

    lines.append(f"Catalog State: {snapshot.catalog_state.value}")

    return "\n".join(lines)


DECISION_SYSTEM_PROMPT = """You are analyzing LLM model health for a benchmarking service.

Based on the provided signals, determine the appropriate action:

- **disable**: Model is permanently broken and should be disabled
  - Examples: 404 "model not found" for 48+ hours, deprecated by provider, authentication permanently fails
  - High confidence required (≥0.90)

- **monitor**: Concerning signals but not conclusive enough to disable
  - Examples: High error rate but might recover, intermittent failures, unclear error messages
  - Medium confidence (0.60-0.89)

- **ignore**: Model is healthy or issue is temporary
  - Examples: Recent successes, errors are rate limits or transient issues, low error rate
  - Any confidence level

Decision criteria:
1. Time matters: 404 errors for 48+ hours → disable; 404 for 6 hours → monitor
2. Error types matter: "model not found" is permanent; "rate limit" is temporary
3. Success pattern matters: Recent successes override old errors
4. Provider signals matter: Already marked deprecated → disable
5. Hard failures count more than soft failures

Return ONLY valid JSON (no markdown, no extra text):
{
  "action": "disable|monitor|ignore",
  "confidence": 0.95,
  "reasoning": "Clear explanation of decision in 1-2 sentences"
}
"""


def fast_pass(
    snapshots: List[LifecycleSnapshot],
    now: datetime
) -> tuple[List[OperatorDecision], List[LifecycleSnapshot]]:
    """
    Deterministic fast-pass: handle obvious cases without LLM.

    Rules:
    - Already disabled → ignore
    - Marked deprecated in metadata → disable (confidence 0.95)
    - Recent success (last 24h) + no hard failures 7d → ignore
    - No signals (never succeeded, no errors) → monitor

    Returns:
        (passthrough_decisions, needs_analysis)
    """
    passthrough = []
    needs_analysis = []

    for snapshot in snapshots:
        # Already disabled
        if not snapshot.metadata.enabled:
            passthrough.append(OperatorDecision(
                provider=snapshot.provider,
                model_id=snapshot.model_id,
                action="ignore",
                confidence=1.0,
                reasoning="Already disabled in database",
                suggested_at=now,
                suggested_by="fast-pass",
            ))
            continue

        # Marked deprecated in metadata
        if snapshot.metadata.deprecated:
            passthrough.append(OperatorDecision(
                provider=snapshot.provider,
                model_id=snapshot.model_id,
                action="disable",
                confidence=0.95,
                reasoning=f"Marked deprecated in metadata{f': {snapshot.metadata.deprecation_reason}' if snapshot.metadata.deprecation_reason else ''}",
                suggested_at=now,
                suggested_by="fast-pass",
            ))
            continue

        # Recent success (last 24h) + no hard failures
        success_age = snapshot.successes.age_days(now)
        if success_age is not None and success_age < 1.0 and snapshot.errors.hard_failures_7d == 0:
            passthrough.append(OperatorDecision(
                provider=snapshot.provider,
                model_id=snapshot.model_id,
                action="ignore",
                confidence=0.90,
                reasoning="Recent success (last 24h) with no hard failures",
                suggested_at=now,
                suggested_by="fast-pass",
            ))
            continue

        # No signals at all
        if snapshot.successes.last_success is None and snapshot.errors.last_error is None:
            passthrough.append(OperatorDecision(
                provider=snapshot.provider,
                model_id=snapshot.model_id,
                action="monitor",
                confidence=0.80,
                reasoning="No signals - never succeeded or failed",
                suggested_at=now,
                suggested_by="fast-pass",
            ))
            continue

        # Needs LLM analysis
        needs_analysis.append(snapshot)

    return passthrough, needs_analysis


def _normalize_message(message: str) -> str:
    """Normalize error message for grouping."""
    # Take first 100 chars, lowercase, strip special chars
    normalized = message[:100].lower()
    # Keep only alphanumeric and spaces
    normalized = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in normalized)
    # Collapse whitespace
    normalized = ' '.join(normalized.split())
    return normalized


def build_situations(
    snapshots: List[LifecycleSnapshot],
    now: datetime
) -> List[Situation]:
    """
    Group models by provider + error pattern.

    Grouping key: {provider}|{error_kind}|{normalized_msg_prefix}

    Returns:
        List of Situation objects
    """
    from collections import defaultdict

    # Group snapshots by situation
    situations_map: dict[str, List[LifecycleSnapshot]] = defaultdict(list)

    for snapshot in snapshots:
        # Determine error kind from most recent error
        error_kind = "other"
        if snapshot.errors.recent_messages:
            error_kind = snapshot.errors.recent_messages[0].kind

        # Normalize first error message for grouping
        msg_prefix = ""
        if snapshot.errors.recent_messages:
            msg_prefix = _normalize_message(snapshot.errors.recent_messages[0].message)

        # Create situation key
        situation_id = f"{snapshot.provider}|{error_kind}|{msg_prefix}"
        situations_map[situation_id].append(snapshot)

    # Convert to Situation objects
    situations = []
    for situation_id, group_snapshots in situations_map.items():
        provider, error_kind, _ = situation_id.split('|', 2)

        # Find oldest and newest errors
        oldest_error = None
        newest_error = None
        for snapshot in group_snapshots:
            if snapshot.errors.last_error:
                if oldest_error is None or snapshot.errors.last_error < oldest_error:
                    oldest_error = snapshot.errors.last_error
                if newest_error is None or snapshot.errors.last_error > newest_error:
                    newest_error = snapshot.errors.last_error

        # Collect sample messages (2-3 representative, truncated)
        sample_messages = []
        seen_messages = set()
        for snapshot in group_snapshots:
            for msg in snapshot.errors.recent_messages[:3]:
                truncated = msg.message[:200]
                if truncated not in seen_messages:
                    sample_messages.append(truncated)
                    seen_messages.add(truncated)
                if len(sample_messages) >= 3:
                    break
            if len(sample_messages) >= 3:
                break

        situations.append(Situation(
            situation_id=situation_id,
            provider=provider,
            error_kind=error_kind,
            model_count=len(group_snapshots),
            model_ids=[s.model_id for s in group_snapshots],
            snapshots=group_snapshots,
            oldest_error=oldest_error,
            newest_error=newest_error,
            sample_messages=sample_messages,
        ))

    return situations


async def call_llm_for_decision(context: str) -> dict:
    """
    Call OpenAI LLM to reason about model health.

    Returns parsed JSON dict with action, confidence, reasoning.
    Raises exception on API failure.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    prompt = f"""Analyze this model's health signals and determine the appropriate action:

{context}

Respond with JSON only (no markdown):
{{"action": "disable|monitor|ignore", "confidence": 0.0-1.0, "reasoning": "..."}}"""

    request_body = {
        "model": OPENAI_MODEL,
        "temperature": 0,
        "max_completion_tokens": 500,
        "messages": [
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json=request_body
        )
        response.raise_for_status()
        result = response.json()

        text = result["choices"][0]["message"]["content"]

        # Parse JSON response (handle markdown code fences)
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        return json.loads(text)


async def call_llm_for_batch_decisions(situations: List[Situation]) -> dict[str, dict]:
    """
    Make a SINGLE LLM call for all situations.

    Returns:
        Dict mapping situation_id → {action, confidence, reasoning}
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    if not situations:
        return {}

    # Build prompt with all situations
    situation_summaries = []
    for sit in situations:
        # Calculate error age
        error_age_str = "Unknown"
        if sit.oldest_error and sit.newest_error:
            oldest_days = (datetime.now(timezone.utc) - sit.oldest_error).days
            newest_days = (datetime.now(timezone.utc) - sit.newest_error).days
            if oldest_days == newest_days:
                error_age_str = f"{oldest_days} days"
            else:
                error_age_str = f"{newest_days}-{oldest_days} days"

        # Format sample messages
        sample_msgs_str = "\n".join(f"  - {msg}" for msg in sit.sample_messages[:3])

        situation_summaries.append(f"""
[situation_id: {sit.situation_id}]
Provider: {sit.provider}
Error type: {sit.error_kind}
Models affected: {sit.model_count}
Error age: {error_age_str}
Sample errors:
{sample_msgs_str}
""")

    prompt = f"""You're analyzing LLM model health. Here are situations requiring decisions:

{''.join(situation_summaries)}

For each situation, determine action (disable/monitor/ignore).

Decision criteria:
- disable: Model permanently broken (404 "not found" 48+ hours, deprecated, auth permanently fails)
- monitor: Concerning but not conclusive (high error rate might recover, intermittent failures)
- ignore: Healthy or temporary issue (recent successes, rate limits, transient errors)

Return JSON array ONLY (no markdown):
[{{"situation_id": "...", "action": "disable|monitor|ignore", "confidence": 0.0-1.0, "reasoning": "1-2 sentence explanation"}}]
"""

    request_body = {
        "model": OPENAI_MODEL,
        "temperature": 0,
        "max_completion_tokens": 2000,
        "messages": [
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json=request_body
        )
        response.raise_for_status()
        result = response.json()

        text = result["choices"][0]["message"]["content"]

        # Parse JSON response (handle markdown code fences)
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        decisions_list = json.loads(text)

        # Convert to dict keyed by situation_id
        decisions_dict = {}
        for decision in decisions_list:
            situation_id = decision["situation_id"]
            decisions_dict[situation_id] = {
                "action": decision["action"],
                "confidence": float(decision["confidence"]),
                "reasoning": decision["reasoning"]
            }

        # Log token usage
        usage = result.get("usage", {})
        logger.info(
            f"Batch LLM call: {len(situations)} situations, "
            f"{usage.get('prompt_tokens', 0)} prompt + {usage.get('completion_tokens', 0)} completion = "
            f"{usage.get('total_tokens', 0)} total tokens"
        )

        return decisions_dict


def expand_situation_decisions(
    situations: List[Situation],
    situation_decisions: dict[str, dict],
    now: datetime
) -> List[OperatorDecision]:
    """
    Expand situation-level decisions to per-model decisions.

    Applies deterministic overrides:
    - If model had recent success (7d), downgrade disable → monitor
    """
    decisions = []

    for situation in situations:
        # Get the situation-level decision
        sit_decision = situation_decisions.get(situation.situation_id)
        if not sit_decision:
            logger.warning(f"No decision for situation {situation.situation_id}, using fallback")
            # Fallback to monitor
            sit_decision = {
                "action": "monitor",
                "confidence": 0.50,
                "reasoning": "LLM did not provide decision for this situation"
            }

        # Apply to each model in the situation
        for snapshot in situation.snapshots:
            action = sit_decision["action"]
            confidence = sit_decision["confidence"]
            reasoning = sit_decision["reasoning"]

            # Deterministic override: recent success downgrades disable → monitor
            if action == "disable":
                success_age = snapshot.successes.age_days(now)
                if success_age is not None and success_age <= 7.0:
                    action = "monitor"
                    confidence = max(0.70, confidence - 0.15)  # Lower confidence
                    reasoning = f"{reasoning} [Override: Recent success within 7d, monitoring instead]"

            decisions.append(OperatorDecision(
                provider=snapshot.provider,
                model_id=snapshot.model_id,
                action=action,
                confidence=confidence,
                reasoning=reasoning,
                suggested_at=now,
                suggested_by=f"ai-operator-batch-{OPENAI_MODEL}",
            ))

    return decisions


def convert_classifier_to_operator_decision(
    snapshot: LifecycleSnapshot,
    now: datetime,
    suggested_by: str = "classifier-fallback"
) -> OperatorDecision:
    """Convert classifier.py decision to OperatorDecision format."""
    decision = classify_snapshot(snapshot, now=now)

    # Map LifecycleStatus to operator actions
    action_map = {
        LifecycleStatus.LIKELY_DEPRECATED: "disable",
        LifecycleStatus.DEPRECATED: "disable",
        LifecycleStatus.DISABLED: "ignore",  # Already disabled
        LifecycleStatus.FAILING: "monitor",
        LifecycleStatus.STALE: "monitor",
        LifecycleStatus.NEVER_SUCCEEDED: "monitor",
        LifecycleStatus.MONITOR: "monitor",
        LifecycleStatus.ACTIVE: "ignore",
    }

    action = action_map.get(decision.status, "monitor")

    # Map confidence
    confidence_map = {"high": 0.90, "medium": 0.70, "low": 0.50}
    confidence = confidence_map.get(decision.confidence, 0.70)

    # Build reasoning from decision reasons
    reasoning = " ".join(decision.reasons) if decision.reasons else "Deterministic classifier decision"

    return OperatorDecision(
        provider=snapshot.provider,
        model_id=snapshot.model_id,
        action=action,
        confidence=confidence,
        reasoning=reasoning,
        suggested_at=now,
        suggested_by=suggested_by,
    )


async def generate_decision_for_snapshot(
    snapshot: LifecycleSnapshot,
    now: datetime
) -> OperatorDecision:
    """
    Generate a single decision using LLM reasoning.
    Falls back to classifier.py on failure.
    """
    try:
        context = format_snapshot_context(snapshot)
        result = await call_llm_for_decision(context)

        return OperatorDecision(
            provider=snapshot.provider,
            model_id=snapshot.model_id,
            action=result["action"],
            confidence=float(result["confidence"]),
            reasoning=result["reasoning"],
            suggested_at=now,
            suggested_by=f"ai-operator-{OPENAI_MODEL}",
        )
    except Exception as e:
        logger.warning(
            f"LLM call failed for {snapshot.provider}/{snapshot.model_id}, "
            f"using fallback: {e}"
        )
        return convert_classifier_to_operator_decision(snapshot, now)


async def generate_decisions(
    snapshots: List[LifecycleSnapshot],
    *,
    now: Optional[datetime] = None
) -> List[OperatorDecision]:
    """
    Use LLM to reason about model health signals and suggest actions.

    Falls back to classifier.py if LLM unavailable or fails.

    Args:
        snapshots: List of lifecycle snapshots to analyze
        now: Current time (defaults to utcnow)

    Returns:
        List of operator decisions with reasoning
    """
    now = now or datetime.now(timezone.utc)

    # Process all snapshots concurrently (with reasonable limit)
    decisions = []
    llm_failures = 0

    # Batch processing to avoid overwhelming the API
    batch_size = 10
    for i in range(0, len(snapshots), batch_size):
        batch = snapshots[i:i + batch_size]

        # Process batch concurrently
        tasks = [generate_decision_for_snapshot(snap, now) for snap in batch]
        batch_decisions = await asyncio.gather(*tasks, return_exceptions=True)

        for decision in batch_decisions:
            if isinstance(decision, Exception):
                logger.error(f"Unexpected error in batch: {decision}")
                llm_failures += 1
            else:
                decisions.append(decision)
                if decision.suggested_by.startswith("classifier-fallback"):
                    llm_failures += 1

    if llm_failures > 0:
        logger.info(
            f"Used fallback classifier for {llm_failures}/{len(snapshots)} models "
            f"({llm_failures/len(snapshots)*100:.1f}%)"
        )

    return decisions
