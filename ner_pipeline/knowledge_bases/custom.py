import json
import logging
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

    def __init__(self, path: str):
        self.entities: Dict[str, Entity] = {}  # Indexed by ID
        self.titles: List[str] = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
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

