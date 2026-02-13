"""
Shared utilities for LELA.
"""

from lela.utils.spans import filter_spans
from lela.utils.extensions import (
    ensure_candidates_extension,
    ensure_resolved_entity_extension,
    ensure_context_extension,
)

__all__ = [
    "filter_spans",
    "ensure_candidates_extension",
    "ensure_resolved_entity_extension",
    "ensure_context_extension",
]
