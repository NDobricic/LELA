"""Unit tests for LELABM25CandidateGenerator."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from ner_pipeline.types import Candidate, Document, Entity, Mention
from ner_pipeline.knowledge_bases.lela_jsonl import LELAJSONLKnowledgeBase


class TestLELABM25CandidateGenerator:
    """Tests for LELABM25CandidateGenerator class."""

    @pytest.fixture
    def lela_kb_data(self) -> list[dict]:
        return [
            {"title": "Barack Obama", "description": "44th US President from Illinois"},
            {"title": "Michelle Obama", "description": "Former First Lady and author"},
            {"title": "Joe Biden", "description": "46th US President from Delaware"},
            {"title": "United States", "description": "Country in North America"},
            {"title": "White House", "description": "Official residence of the US President"},
        ]

    @pytest.fixture
    def temp_kb_file(self, lela_kb_data: list[dict]) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in lela_kb_data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def kb(self, temp_kb_file: str) -> LELAJSONLKnowledgeBase:
        return LELAJSONLKnowledgeBase(path=temp_kb_file)

    @pytest.fixture
    def sample_doc(self) -> Document:
        return Document(
            id="test-doc",
            text="Barack Obama was the 44th President of the United States."
        )

    def test_requires_knowledge_base(self):
        # Import here to allow mocking
        with patch("ner_pipeline.candidates.lela_bm25._get_bm25s"):
            with patch("ner_pipeline.candidates.lela_bm25._get_stemmer"):
                from ner_pipeline.candidates.lela_bm25 import LELABM25CandidateGenerator
                with pytest.raises(ValueError, match="requires a knowledge base"):
                    LELABM25CandidateGenerator(kb=None)

    @patch("ner_pipeline.candidates.lela_bm25._get_stemmer")
    @patch("ner_pipeline.candidates.lela_bm25._get_bm25s")
    def test_generate_returns_candidates(
        self, mock_bm25s, mock_stemmer, kb, sample_doc
    ):
        # Setup mocks
        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        # Setup retrieve response
        mock_results = MagicMock()
        mock_results.documents = [[
            {"title": "Barack Obama", "description": "44th US President"},
            {"title": "Michelle Obama", "description": "Former First Lady"},
        ]]
        mock_results.scores = [[0.9, 0.7]]
        mock_retriever.retrieve.return_value = mock_results

        mock_bm25s.return_value.tokenize.return_value = [["obama"]]

        from ner_pipeline.candidates.lela_bm25 import LELABM25CandidateGenerator
        generator = LELABM25CandidateGenerator(kb=kb, top_k=5)

        mention = Mention(start=0, end=12, text="Barack Obama")
        candidates = generator.generate(mention, sample_doc)

        assert len(candidates) == 2
        assert all(isinstance(c, Candidate) for c in candidates)

    @patch("ner_pipeline.candidates.lela_bm25._get_stemmer")
    @patch("ner_pipeline.candidates.lela_bm25._get_bm25s")
    def test_candidates_have_entity_ids(
        self, mock_bm25s, mock_stemmer, kb, sample_doc
    ):
        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        mock_results = MagicMock()
        mock_results.documents = [[
            {"title": "Barack Obama", "description": "44th US President"},
        ]]
        mock_results.scores = [[0.9]]
        mock_retriever.retrieve.return_value = mock_results
        mock_bm25s.return_value.tokenize.return_value = [["obama"]]

        from ner_pipeline.candidates.lela_bm25 import LELABM25CandidateGenerator
        generator = LELABM25CandidateGenerator(kb=kb, top_k=5)

        mention = Mention(start=0, end=12, text="Barack Obama")
        candidates = generator.generate(mention, sample_doc)

        assert candidates[0].entity_id == "Barack Obama"

    @patch("ner_pipeline.candidates.lela_bm25._get_stemmer")
    @patch("ner_pipeline.candidates.lela_bm25._get_bm25s")
    def test_candidates_have_descriptions(
        self, mock_bm25s, mock_stemmer, kb, sample_doc
    ):
        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        mock_results = MagicMock()
        mock_results.documents = [[
            {"title": "Barack Obama", "description": "44th US President"},
        ]]
        mock_results.scores = [[0.9]]
        mock_retriever.retrieve.return_value = mock_results
        mock_bm25s.return_value.tokenize.return_value = [["obama"]]

        from ner_pipeline.candidates.lela_bm25 import LELABM25CandidateGenerator
        generator = LELABM25CandidateGenerator(kb=kb, top_k=5)

        mention = Mention(start=0, end=12, text="Barack Obama")
        candidates = generator.generate(mention, sample_doc)

        assert candidates[0].description == "44th US President"

    @patch("ner_pipeline.candidates.lela_bm25._get_stemmer")
    @patch("ner_pipeline.candidates.lela_bm25._get_bm25s")
    def test_empty_tokenization_returns_empty(
        self, mock_bm25s, mock_stemmer, kb, sample_doc
    ):
        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        # Empty tokenization
        mock_bm25s.return_value.tokenize.return_value = [[]]

        from ner_pipeline.candidates.lela_bm25 import LELABM25CandidateGenerator
        generator = LELABM25CandidateGenerator(kb=kb, top_k=5)

        mention = Mention(start=0, end=3, text="...")
        candidates = generator.generate(mention, sample_doc)

        assert candidates == []

    @patch("ner_pipeline.candidates.lela_bm25._get_stemmer")
    @patch("ner_pipeline.candidates.lela_bm25._get_bm25s")
    def test_use_context_includes_context_in_query(
        self, mock_bm25s, mock_stemmer, kb, sample_doc
    ):
        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        mock_results = MagicMock()
        mock_results.documents = [[]]
        mock_results.scores = [[]]
        mock_retriever.retrieve.return_value = mock_results

        tokenize_calls = []
        def capture_tokenize(texts, **kwargs):
            tokenize_calls.append(texts)
            return [[]]
        mock_bm25s.return_value.tokenize.side_effect = capture_tokenize

        from ner_pipeline.candidates.lela_bm25 import LELABM25CandidateGenerator
        generator = LELABM25CandidateGenerator(kb=kb, top_k=5, use_context=True)

        mention = Mention(
            start=0, end=12, text="Barack Obama",
            context="was the 44th President"
        )
        generator.generate(mention, sample_doc)

        # Check that context was included in the query
        assert len(tokenize_calls) > 0
        query = tokenize_calls[0][0]
        assert "Barack Obama" in query
        assert "44th President" in query

    @patch("ner_pipeline.candidates.lela_bm25._get_stemmer")
    @patch("ner_pipeline.candidates.lela_bm25._get_bm25s")
    def test_use_context_false_excludes_context(
        self, mock_bm25s, mock_stemmer, kb, sample_doc
    ):
        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        mock_results = MagicMock()
        mock_results.documents = [[]]
        mock_results.scores = [[]]
        mock_retriever.retrieve.return_value = mock_results

        tokenize_calls = []
        def capture_tokenize(texts, **kwargs):
            tokenize_calls.append(texts)
            return [[]]
        mock_bm25s.return_value.tokenize.side_effect = capture_tokenize

        from ner_pipeline.candidates.lela_bm25 import LELABM25CandidateGenerator
        generator = LELABM25CandidateGenerator(kb=kb, top_k=5, use_context=False)

        mention = Mention(
            start=0, end=12, text="Barack Obama",
            context="was the 44th President"
        )
        generator.generate(mention, sample_doc)

        # Context should not be in query when use_context=False
        assert len(tokenize_calls) > 0
        query = tokenize_calls[0][0]
        assert query == "Barack Obama"

    @patch("ner_pipeline.candidates.lela_bm25._get_stemmer")
    @patch("ner_pipeline.candidates.lela_bm25._get_bm25s")
    def test_respects_top_k(self, mock_bm25s, mock_stemmer, kb, sample_doc):
        mock_stemmer_instance = MagicMock()
        mock_stemmer.return_value.Stemmer.return_value = mock_stemmer_instance

        mock_tokenizer = MagicMock()
        mock_bm25s.return_value.tokenization.Tokenizer.return_value = mock_tokenizer
        mock_tokenizer.tokenize.return_value = [("tokens",)]

        mock_retriever = MagicMock()
        mock_bm25s.return_value.BM25.return_value = mock_retriever

        mock_bm25s.return_value.tokenize.return_value = [["test"]]

        from ner_pipeline.candidates.lela_bm25 import LELABM25CandidateGenerator
        generator = LELABM25CandidateGenerator(kb=kb, top_k=3)

        # Check that top_k is stored
        assert generator.top_k == 3
