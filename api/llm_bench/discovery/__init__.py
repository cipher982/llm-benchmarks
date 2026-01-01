"""
Discovery module for finding new models via OpenRouter catalog.

This module uses OpenRouter as a discovery layer to identify new models
that should be added to our direct provider benchmarks.
"""

from .openrouter import fetch_openrouter_models, store_catalog_in_db
from .matcher import match_to_direct_providers
from .cli import main

__all__ = [
    "fetch_openrouter_models",
    "store_catalog_in_db",
    "match_to_direct_providers",
    "main",
]
