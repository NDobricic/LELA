"""Knowledge base adapters."""

from .jsonl import JSONLKnowledgeBase  # noqa: F401
from .yago_downloader import yago_kb  # noqa: F401  (registers "yago" factory)
