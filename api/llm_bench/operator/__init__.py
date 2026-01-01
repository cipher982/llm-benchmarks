"""
AI Operator - LLM-driven model lifecycle management.

This module uses LLM reasoning to make decisions about model health,
with fallback to deterministic classifier.py when LLM calls fail.
"""

from .engine import generate_decisions, OperatorDecision

__all__ = ["generate_decisions", "OperatorDecision"]
