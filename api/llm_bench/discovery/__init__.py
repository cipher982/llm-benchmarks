"""
Discovery module for finding new models via OpenRouter catalog.

This module uses OpenRouter as a discovery layer to identify new models
that should be added to our direct provider benchmarks.
"""

from .cli import main
from .matcher import match_to_direct_providers
from .openrouter import fetch_openrouter_models
from .openrouter import fetch_openrouter_providers
from .openrouter import store_catalog_in_db
from .openrouter import store_providers_in_db

__all__ = [
    "fetch_openrouter_models",
    "fetch_openrouter_providers",
    "store_catalog_in_db",
    "store_providers_in_db",
    "match_to_direct_providers",
    "main",
]
