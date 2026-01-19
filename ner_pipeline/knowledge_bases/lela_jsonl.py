"""
LELA-format JSONL knowledge base.

Loads entities from JSONL files with the LELA format:
{"title": "Entity Name", "description": "Entity description..."}
"""

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from rapidfuzz import process

from ner_pipeline.registry import knowledge_bases
from ner_pipeline.types import Entity

logger = logging.getLogger(__name__)


@knowledge_bases.register("lela_jsonl")
class LELAJSONLKnowledgeBase:
    """
    Knowledge base loader for LELA-format JSONL files.

    LELA format uses 'title' as the entity ID and includes a 'description'.
    This differs from the standard 'custom' KB which uses a separate 'id' field.

    Config options:
        path: Path to the JSONL file
        title_field: Field name for entity title/ID (default: "title")
        description_field: Field name for description (default: "description")
    """

    def __init__(
        self,
        path: str,
        title_field: str = "title",
        description_field: str = "description",
    ):
        self.path = Path(path)
        self.title_field = title_field
        self.description_field = description_field

        self.entities: Dict[str, Entity] = {}
        self.titles: List[str] = []

        self._load()

    def _load(self) -> None:
        """Load entities from JSONL file."""
        logger.info(f"Loading LELA JSONL knowledge base from {self.path}")

        with self.path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue

                title = item.get(self.title_field)
                if not title:
                    logger.warning(f"Skipping line {line_num}: missing '{self.title_field}'")
                    continue

                description = item.get(self.description_field, "")

                # Use title as entity ID (LELA convention)
                entity = Entity(
                    id=str(title),
                    title=str(title),
                    description=description if description else None,
                    metadata={
                        k: v for k, v in item.items()
                        if k not in {self.title_field, self.description_field}
                    },
                )

                self.entities[entity.id] = entity
                self.titles.append(entity.title)

        logger.info(f"Loaded {len(self.entities)} entities from LELA JSONL")

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID (which is the title in LELA format)."""
        return self.entities.get(entity_id)

    def search(self, query: str, top_k: int = 10) -> List[Entity]:
        """Fuzzy search entities by title."""
        if not self.titles:
            return []

        results = process.extract(query, self.titles, limit=top_k)
        hits: List[Entity] = []
        for title, score, idx in results:
            entity = self.entities.get(title)
            if entity:
                hits.append(entity)
        return hits

    def all_entities(self) -> Iterable[Entity]:
        """Iterate over all entities."""
        return self.entities.values()

    def get_descriptions_dict(self) -> Dict[str, str]:
        """
        Get a dictionary mapping entity titles to descriptions.

        Useful for LELA-style candidate generation.
        """
        return {
            entity.title: entity.description or ""
            for entity in self.entities.values()
        }
