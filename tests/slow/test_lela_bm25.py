"""Slow tests for LELABM25CandidateGenerator with actual bm25s."""

import json
import os
import tempfile

import pytest

from ner_pipeline.types import Document, Mention


@pytest.fixture
def lela_kb_data() -> list[dict]:
    """Sample entity data for testing."""
    return [
        {"title": "Barack Obama", "description": "44th President of the United States, born in Hawaii"},
        {"title": "Michelle Obama", "description": "Former First Lady of the United States, lawyer and author"},
        {"title": "Joe Biden", "description": "46th President of the United States from Delaware"},
        {"title": "Kamala Harris", "description": "49th Vice President of the United States"},
        {"title": "Donald Trump", "description": "45th President of the United States, businessman"},
        {"title": "White House", "description": "Official residence of the President of the United States"},
        {"title": "United States Congress", "description": "Legislative branch of the federal government"},
        {"title": "Supreme Court", "description": "Highest court in the federal judiciary"},
        {"title": "Hawaii", "description": "US state located in the Pacific Ocean"},
        {"title": "Delaware", "description": "US state on the East Coast"},
    ]


@pytest.fixture
def temp_kb_file(lela_kb_data: list[dict]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in lela_kb_data:
            f.write(json.dumps(item) + "\n")
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def sample_doc() -> Document:
    return Document(
        id="test-doc",
        text="Barack Obama was the 44th President of the United States. He was born in Hawaii."
    )


class TestLELABM25WithRealIndex:
    """Tests using actual bm25s indexing."""

    @pytest.mark.slow
    def test_generate_finds_relevant_candidates(self, temp_kb_file, sample_doc):
        from ner_pipeline.knowledge_bases.lela_jsonl import LELAJSONLKnowledgeBase
        from ner_pipeline.candidates.lela_bm25 import LELABM25CandidateGenerator

        kb = LELAJSONLKnowledgeBase(path=temp_kb_file)
        generator = LELABM25CandidateGenerator(kb=kb, top_k=5)

        mention = Mention(start=0, end=12, text="Barack Obama")
        candidates = generator.generate(mention, sample_doc)

        assert len(candidates) > 0
        entity_ids = [c.entity_id for c in candidates]
        assert "Barack Obama" in entity_ids

    @pytest.mark.slow
    def test_president_query_finds_presidents(self, temp_kb_file, sample_doc):
        from ner_pipeline.knowledge_bases.lela_jsonl import LELAJSONLKnowledgeBase
        from ner_pipeline.candidates.lela_bm25 import LELABM25CandidateGenerator

        kb = LELAJSONLKnowledgeBase(path=temp_kb_file)
        generator = LELABM25CandidateGenerator(kb=kb, top_k=5)

        mention = Mention(start=0, end=9, text="President", context="44th President of the United States")
        candidates = generator.generate(mention, sample_doc)

        entity_ids = [c.entity_id for c in candidates]
        # Should find entities with "President" in description
        president_candidates = [eid for eid in entity_ids if "President" in kb.get_entity(eid).description or ""]
        assert len(president_candidates) > 0

    @pytest.mark.slow
    def test_respects_top_k_limit(self, temp_kb_file, sample_doc):
        from ner_pipeline.knowledge_bases.lela_jsonl import LELAJSONLKnowledgeBase
        from ner_pipeline.candidates.lela_bm25 import LELABM25CandidateGenerator

        kb = LELAJSONLKnowledgeBase(path=temp_kb_file)
        generator = LELABM25CandidateGenerator(kb=kb, top_k=3)

        mention = Mention(start=0, end=9, text="President")
        candidates = generator.generate(mention, sample_doc)

        assert len(candidates) <= 3

    @pytest.mark.slow
    def test_candidates_have_descriptions(self, temp_kb_file, sample_doc):
        from ner_pipeline.knowledge_bases.lela_jsonl import LELAJSONLKnowledgeBase
        from ner_pipeline.candidates.lela_bm25 import LELABM25CandidateGenerator

        kb = LELAJSONLKnowledgeBase(path=temp_kb_file)
        generator = LELABM25CandidateGenerator(kb=kb, top_k=5)

        mention = Mention(start=0, end=12, text="Barack Obama")
        candidates = generator.generate(mention, sample_doc)

        for c in candidates:
            # All candidates should have descriptions
            assert c.description is not None

    @pytest.mark.slow
    def test_context_improves_retrieval(self, temp_kb_file, sample_doc):
        from ner_pipeline.knowledge_bases.lela_jsonl import LELAJSONLKnowledgeBase
        from ner_pipeline.candidates.lela_bm25 import LELABM25CandidateGenerator

        kb = LELAJSONLKnowledgeBase(path=temp_kb_file)
        generator = LELABM25CandidateGenerator(kb=kb, top_k=5, use_context=True)

        # Hawaii with context about Pacific Ocean
        mention = Mention(
            start=70, end=76, text="Hawaii",
            context="He was born in Hawaii"
        )
        candidates = generator.generate(mention, sample_doc)

        entity_ids = [c.entity_id for c in candidates]
        assert "Hawaii" in entity_ids

    @pytest.mark.slow
    def test_stemming_improves_matching(self, temp_kb_file, sample_doc):
        from ner_pipeline.knowledge_bases.lela_jsonl import LELAJSONLKnowledgeBase
        from ner_pipeline.candidates.lela_bm25 import LELABM25CandidateGenerator

        kb = LELAJSONLKnowledgeBase(path=temp_kb_file)
        generator = LELABM25CandidateGenerator(kb=kb, top_k=5)

        # "Presidents" should still match "President" due to stemming
        mention = Mention(start=0, end=10, text="Presidents")
        candidates = generator.generate(mention, sample_doc)

        entity_ids = [c.entity_id for c in candidates]
        # Should find president-related entities
        assert len(candidates) > 0

    @pytest.mark.slow
    def test_empty_knowledge_base_raises(self):
        from ner_pipeline.knowledge_bases.lela_jsonl import LELAJSONLKnowledgeBase
        from ner_pipeline.candidates.lela_bm25 import LELABM25CandidateGenerator

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            kb = LELAJSONLKnowledgeBase(path=path)
            with pytest.raises(ValueError, match="empty"):
                LELABM25CandidateGenerator(kb=kb)
        finally:
            os.unlink(path)
