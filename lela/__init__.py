"""
LELA package.

This package provides a configurable entity linking system
using spaCy's component architecture for NER, candidate generation, reranking,
and entity disambiguation.
"""

__version__ = "0.2.0"

# Import spacy_components to register factories with spaCy
from lela import spacy_components  # noqa: F401

from .pipeline import Lela  # noqa: E402
from ._types import (  # noqa: E402
    Candidate,
    Document,
    Entity,
    Mention,
    ProgressCallback,
    ResolvedMention,
)

__all__ = [
    "Lela",
    "Candidate",
    "Document",
    "Entity",
    "Mention",
    "ProgressCallback",
    "ResolvedMention",
]
