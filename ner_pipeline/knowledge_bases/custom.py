import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from rapidfuzz import process

from ner_pipeline.registry import knowledge_bases
from ner_pipeline.types import Entity
from ner_pipeline.knowledge_bases.base import KnowledgeBase


@knowledge_bases.register("custom")
class CustomJSONLKnowledgeBase:
    """Loads entities from a JSONL file with fields: id, title, description."""

    def __init__(self, path: str):
        self.entities: Dict[str, Entity] = {}  # Indexed by ID
        self.entities_by_title: Dict[str, Entity] = {}  # Indexed by title
        self.titles: List[str] = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                entity = Entity(
                    id=item["id"],
                    title=item.get("title") or item["id"],
                    description=item.get("description"),
                    metadata={k: v for k, v in item.items() if k not in {"id", "title", "description"}},
                )
                self.entities[entity.id] = entity
                self.entities_by_title[entity.title] = entity
                self.titles.append(entity.title)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID or title (checks ID first, then title)."""
        entity = self.entities.get(entity_id)
        if entity is None:
            entity = self.entities_by_title.get(entity_id)
        return entity

    def search(self, query: str, top_k: int = 10) -> List[Entity]:
        results = process.extract(query, self.titles, limit=top_k)
        hits: List[Entity] = []
        for _, _, idx in results:
            title = self.titles[idx]
            ent = next(e for e in self.entities.values() if e.title == title)
            hits.append(ent)
        return hits

    def all_entities(self) -> Iterable[Entity]:
        return self.entities.values()

