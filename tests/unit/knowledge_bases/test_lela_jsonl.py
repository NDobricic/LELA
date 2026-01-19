"""Unit tests for LELAJSONLKnowledgeBase."""

import json
import os
import tempfile

import pytest

from ner_pipeline.knowledge_bases.lela_jsonl import LELAJSONLKnowledgeBase
from ner_pipeline.types import Entity


class TestLELAJSONLKnowledgeBase:
    """Tests for LELAJSONLKnowledgeBase class."""

    @pytest.fixture
    def lela_kb_data(self) -> list[dict]:
        """LELA format: title as ID, with description."""
        return [
            {"title": "Barack Obama", "description": "44th US President"},
            {"title": "Joe Biden", "description": "46th US President"},
            {"title": "United States", "description": "Country in North America"},
            {"title": "New York", "description": "City in the United States"},
        ]

    @pytest.fixture
    def temp_lela_kb_file(self, lela_kb_data: list[dict]) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in lela_kb_data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def kb(self, temp_lela_kb_file: str) -> LELAJSONLKnowledgeBase:
        return LELAJSONLKnowledgeBase(path=temp_lela_kb_file)

    def test_load_entities_from_file(self, kb: LELAJSONLKnowledgeBase):
        entities = list(kb.all_entities())
        assert len(entities) == 4

    def test_entity_id_equals_title(self, kb: LELAJSONLKnowledgeBase):
        """In LELA format, title IS the entity ID."""
        entity = kb.get_entity("Barack Obama")
        assert entity is not None
        assert entity.id == "Barack Obama"
        assert entity.title == "Barack Obama"

    def test_get_entity_by_title(self, kb: LELAJSONLKnowledgeBase):
        entity = kb.get_entity("Joe Biden")
        assert entity is not None
        assert entity.title == "Joe Biden"
        assert entity.description == "46th US President"

    def test_get_nonexistent_entity(self, kb: LELAJSONLKnowledgeBase):
        entity = kb.get_entity("Donald Trump")
        assert entity is None

    def test_search_finds_matches(self, kb: LELAJSONLKnowledgeBase):
        results = kb.search("Obama", top_k=5)
        assert len(results) > 0
        titles = [e.title for e in results]
        assert "Barack Obama" in titles

    def test_search_top_k_limit(self, kb: LELAJSONLKnowledgeBase):
        results = kb.search("United", top_k=2)
        assert len(results) <= 2

    def test_all_entities_returns_iterable(self, kb: LELAJSONLKnowledgeBase):
        entities = kb.all_entities()
        assert hasattr(entities, "__iter__")
        entity_list = list(entities)
        assert all(isinstance(e, Entity) for e in entity_list)

    def test_get_descriptions_dict(self, kb: LELAJSONLKnowledgeBase):
        """Test the LELA-specific method for getting descriptions."""
        desc_dict = kb.get_descriptions_dict()
        assert isinstance(desc_dict, dict)
        assert len(desc_dict) == 4
        assert desc_dict["Barack Obama"] == "44th US President"
        assert desc_dict["Joe Biden"] == "46th US President"

    def test_metadata_preserved(self):
        """Test that extra fields are stored in metadata."""
        data = [
            {
                "title": "Test Entity",
                "description": "Test description",
                "extra_field": "extra_value",
                "count": 42,
            }
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        try:
            kb = LELAJSONLKnowledgeBase(path=path)
            entity = kb.get_entity("Test Entity")
            assert entity.metadata["extra_field"] == "extra_value"
            assert entity.metadata["count"] == 42
        finally:
            os.unlink(path)

    def test_custom_field_names(self):
        """Test using custom field names."""
        data = [
            {"name": "Custom Entity", "text": "Custom description"}
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        try:
            kb = LELAJSONLKnowledgeBase(
                path=path,
                title_field="name",
                description_field="text",
            )
            entity = kb.get_entity("Custom Entity")
            assert entity is not None
            assert entity.title == "Custom Entity"
            assert entity.description == "Custom description"
        finally:
            os.unlink(path)

    def test_missing_description(self):
        """Test handling of missing description field."""
        data = [{"title": "Entity Without Description"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        try:
            kb = LELAJSONLKnowledgeBase(path=path)
            entity = kb.get_entity("Entity Without Description")
            assert entity is not None
            assert entity.description is None
        finally:
            os.unlink(path)

    def test_empty_file(self):
        """Test loading an empty JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            kb = LELAJSONLKnowledgeBase(path=path)
            entities = list(kb.all_entities())
            assert len(entities) == 0
        finally:
            os.unlink(path)

    def test_skips_lines_without_title(self):
        """Test that lines missing the title field are skipped."""
        data = [
            {"title": "Valid Entity", "description": "Valid"},
            {"description": "Missing title - should be skipped"},
            {"title": "Another Valid", "description": "Also valid"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        try:
            kb = LELAJSONLKnowledgeBase(path=path)
            entities = list(kb.all_entities())
            assert len(entities) == 2
        finally:
            os.unlink(path)

    def test_skips_invalid_json_lines(self):
        """Test that invalid JSON lines are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"title": "Valid", "description": "Good"}\n')
            f.write('invalid json line\n')
            f.write('{"title": "Also Valid", "description": "Also good"}\n')
            path = f.name
        try:
            kb = LELAJSONLKnowledgeBase(path=path)
            entities = list(kb.all_entities())
            assert len(entities) == 2
        finally:
            os.unlink(path)

    def test_skips_empty_lines(self):
        """Test that empty lines are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"title": "Entity1", "description": "Desc1"}\n')
            f.write('\n')
            f.write('   \n')
            f.write('{"title": "Entity2", "description": "Desc2"}\n')
            path = f.name
        try:
            kb = LELAJSONLKnowledgeBase(path=path)
            entities = list(kb.all_entities())
            assert len(entities) == 2
        finally:
            os.unlink(path)

    def test_search_case_insensitive(self, kb: LELAJSONLKnowledgeBase):
        """Test that search is case-insensitive."""
        results = kb.search("OBAMA", top_k=5)
        assert len(results) > 0

    def test_descriptions_dict_empty_description(self):
        """Test get_descriptions_dict with missing descriptions."""
        data = [
            {"title": "With Desc", "description": "Has description"},
            {"title": "Without Desc"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        try:
            kb = LELAJSONLKnowledgeBase(path=path)
            desc_dict = kb.get_descriptions_dict()
            assert desc_dict["With Desc"] == "Has description"
            assert desc_dict["Without Desc"] == ""  # Empty string for missing
        finally:
            os.unlink(path)

    def test_titles_list_maintained(self, kb: LELAJSONLKnowledgeBase):
        """Test that the titles list is properly maintained."""
        assert len(kb.titles) == 4
        assert "Barack Obama" in kb.titles
        assert "Joe Biden" in kb.titles

    def test_unicode_support(self):
        """Test handling of Unicode characters."""
        data = [
            {"title": "日本", "description": "Japan in Japanese"},
            {"title": "François Hollande", "description": "French politician"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            path = f.name
        try:
            kb = LELAJSONLKnowledgeBase(path=path)
            entity = kb.get_entity("日本")
            assert entity is not None
            assert entity.description == "Japan in Japanese"

            entity2 = kb.get_entity("François Hollande")
            assert entity2 is not None
        finally:
            os.unlink(path)
