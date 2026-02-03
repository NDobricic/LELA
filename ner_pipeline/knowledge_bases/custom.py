import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from rapidfuzz import process

from ner_pipeline.registry import knowledge_bases
from ner_pipeline.types import Entity
from ner_pipeline.knowledge_bases.base import KnowledgeBase

logger = logging.getLogger(__name__)


@knowledge_bases.register("custom")
class CustomJSONLKnowledgeBase:
    """
    Loads entities from a JSONL file.

    Supports two formats:
    - Full format: {"id": "...", "title": "...", "description": "..."}
    - Simple format: {"title": "...", "description": "..."} (id defaults to title)

    The 'id' field is optional - if not provided, 'title' is used as the ID.
    """

    def __init__(self, path: str, cache_dir: Optional[str] = None):
        self.source_path = path
        self.entities: Dict[str, Entity] = {}  # Indexed by ID
        self.titles: List[str] = []

        # Try loading from cache
        if cache_dir and self._load_from_cache(cache_dir):
            return

        self._parse_jsonl(path)

        # Save to cache
        if cache_dir:
            self._save_to_cache(cache_dir)

    @property
    def identity_hash(self) -> str:
        """Content-based hash for cache invalidation (uses path + mtime + size)."""
        stat = os.stat(self.source_path)
        raw = f"kb:{self.source_path}:{stat.st_mtime}:{stat.st_size}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _cache_path(self, cache_dir: str) -> Path:
        """Return the cache file path for this KB."""
        kb_dir = Path(cache_dir) / "kb"
        kb_dir.mkdir(parents=True, exist_ok=True)
        return kb_dir / f"{self.identity_hash}.pkl"

    def _load_from_cache(self, cache_dir: str) -> bool:
        """Try to load parsed KB data from cache. Returns True on success."""
        try:
            cache_file = self._cache_path(cache_dir)
            if not cache_file.exists():
                return False
            with cache_file.open("rb") as f:
                entities, titles = pickle.load(f)
            self.entities = entities
            self.titles = titles
            logger.info(
                f"Loaded {len(self.entities)} entities from cache ({cache_file.name})"
            )
            return True
        except Exception:
            logger.warning("KB cache load failed, will rebuild from JSONL", exc_info=True)
            return False

    def _save_to_cache(self, cache_dir: str) -> None:
        """Save parsed KB data to cache."""
        try:
            cache_file = self._cache_path(cache_dir)
            with cache_file.open("wb") as f:
                pickle.dump((self.entities, self.titles), f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved KB cache ({cache_file.name})")
        except Exception:
            logger.warning("Failed to save KB cache", exc_info=True)

    def _parse_jsonl(self, path: str) -> None:
        """Parse the JSONL file and populate entities and titles."""
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Handle YAGO N-Triples trailing "\t."
                if line.endswith("\t."):
                    line = line[:-2].rstrip()
                item = json.loads(line)
                # ID is optional - fall back to title if not provided
                entity_id = item.get("id") or item.get("title")
                entity_title = item.get("title") or item.get("id")
                if not entity_id or not entity_title:
                    logger.warning(f"Skipping entity without id or title: {item}")
                    continue
                entity = Entity(
                    id=entity_id,
                    title=entity_title,
                    description=item.get("description"),
                    metadata={k: v for k, v in item.items() if k not in {"id", "title", "description"}},
                )
                self.entities[entity.id] = entity
                self.titles.append(entity.title)
        logger.info(f"Loaded {len(self.entities)} entities from {path}")

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def search(self, query: str, top_k: int = 10) -> List[Entity]:
        """Fuzzy search entities by title."""
        if not self.titles:
            return []
        results = process.extract(query, self.titles, limit=top_k)
        hits: List[Entity] = []
        for _, _, idx in results:
            title = self.titles[idx]
            # Find entity with this title
            for entity in self.entities.values():
                if entity.title == title:
                    hits.append(entity)
                    break
        return hits

    def all_entities(self) -> Iterable[Entity]:
        return self.entities.values()

