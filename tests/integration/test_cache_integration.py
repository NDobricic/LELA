"""Integration tests for the caching system."""

import json
import os
import tempfile
import time

import pytest

from lela import Lela
from lela._types import Document


@pytest.mark.integration
class TestPipelineCacheDirWiring:
    """Test that cache_dir is properly wired through the pipeline."""

    @pytest.fixture
    def config_dict(self, temp_jsonl_kb: str, temp_cache_dir: str) -> dict:
        return {
            "loader": {"name": "text"},
            "ner": {"name": "regex", "params": {"min_len": 3}},
            "candidate_generator": {"name": "fuzzy", "params": {"top_k": 5}},
            "disambiguator": {"name": "first"},
            "knowledge_base": {"name": "jsonl", "params": {"path": temp_jsonl_kb}},
            "cache_dir": temp_cache_dir,
        }

    def test_pipeline_passes_cache_dir_to_kb(
        self, config_dict: dict, temp_cache_dir: str
    ):
        """KB receives cache_dir and creates kb/ cache subdirectory."""
        lela = Lela(config_dict)

        assert lela.kb is not None
        assert hasattr(lela.kb, "source_path")
        assert hasattr(lela.kb, "identity_hash")

        # KB cache should exist
        kb_cache_dir = os.path.join(temp_cache_dir, "kb")
        assert os.path.isdir(kb_cache_dir)
        cache_files = os.listdir(kb_cache_dir)
        assert len(cache_files) == 1
        assert cache_files[0].endswith(".pkl")

    def test_pipeline_kb_cache_used_on_second_init(
        self, config_dict: dict, temp_cache_dir: str
    ):
        """Second pipeline init uses KB cache (produces identical entities)."""
        lela1 = Lela(config_dict)
        entities1 = {e.id: e.title for e in lela1.kb.all_entities()}

        lela2 = Lela(config_dict)
        entities2 = {e.id: e.title for e in lela2.kb.all_entities()}

        assert entities1 == entities2

    def test_pipeline_produces_same_results_with_cache(
        self, config_dict: dict
    ):
        """Pipeline produces identical results on cold and warm runs."""
        lela1 = Lela(config_dict)
        doc = Document(id="test", text="Barack Obama visited New York City yesterday.")
        result1 = lela1.process_document(doc)

        lela2 = Lela(config_dict)
        result2 = lela2.process_document(doc)

        assert result1["text"] == result2["text"]
        assert len(result1["entities"]) == len(result2["entities"])
        for e1, e2 in zip(result1["entities"], result2["entities"]):
            assert e1["text"] == e2["text"]
            assert e1["start"] == e2["start"]
            assert e1["end"] == e2["end"]
            assert e1["entity_id"] == e2["entity_id"]


@pytest.mark.integration
class TestKBCacheInvalidationIntegration:
    """Test KB cache invalidation through the full pipeline."""

    def test_kb_change_invalidates_cache(self, temp_cache_dir: str):
        """Modifying the KB file causes cache rebuild with new data."""
        kb_data = [
            {"id": "Q1", "title": "Barack Obama", "description": "44th US President"},
            {"id": "Q2", "title": "Joe Biden", "description": "46th US President"},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for item in kb_data:
                f.write(json.dumps(item) + "\n")
            kb_path = f.name

        try:
            config_dict = {
                "loader": {"name": "text"},
                "ner": {"name": "regex", "params": {"min_len": 3}},
                "candidate_generator": {"name": "fuzzy", "params": {"top_k": 5}},
                "knowledge_base": {"name": "jsonl", "params": {"path": kb_path}},
                "cache_dir": temp_cache_dir,
            }

            # First run
            lela1 = Lela(config_dict)
            assert len(list(lela1.kb.all_entities())) == 2

            # Modify file
            time.sleep(0.05)
            with open(kb_path, "a") as f:
                f.write(
                    json.dumps({"id": "Q3", "title": "New Entity", "description": "Added"})
                    + "\n"
                )

            # Second run - should see new entity
            lela2 = Lela(config_dict)
            assert len(list(lela2.kb.all_entities())) == 3
            assert lela2.kb.get_entity("Q3") is not None
        finally:
            os.unlink(kb_path)

    def test_clean_cache_rebuilds(self, temp_cache_dir: str, temp_jsonl_kb: str):
        """Deleting the cache directory forces rebuild."""
        import shutil
        from lela.knowledge_bases.jsonl import clear_kb_cache

        config_dict = {
            "loader": {"name": "text"},
            "ner": {"name": "regex", "params": {"min_len": 3}},
            "candidate_generator": {"name": "fuzzy", "params": {"top_k": 5}},
            "knowledge_base": {"name": "jsonl", "params": {"path": temp_jsonl_kb}},
            "cache_dir": temp_cache_dir,
        }

        # Build cache
        lela1 = Lela(config_dict)
        count1 = len(list(lela1.kb.all_entities()))

        # Delete cache contents and clear in-memory cache so rebuild is triggered
        kb_cache = os.path.join(temp_cache_dir, "kb")
        if os.path.isdir(kb_cache):
            shutil.rmtree(kb_cache)
        clear_kb_cache()

        # Rebuild
        lela2 = Lela(config_dict)
        count2 = len(list(lela2.kb.all_entities()))

        assert count1 == count2
        # Cache should be recreated
        assert os.path.isdir(kb_cache)
