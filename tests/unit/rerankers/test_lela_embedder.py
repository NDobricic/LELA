"""Unit tests for LELAEmbedderReranker."""

from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from ner_pipeline.types import Candidate, Document, Mention


class TestLELAEmbedderReranker:
    """Tests for LELAEmbedderReranker class."""

    @pytest.fixture
    def sample_candidates(self) -> list[Candidate]:
        return [
            Candidate(entity_id="E1", score=0.9, description="Description 1"),
            Candidate(entity_id="E2", score=0.8, description="Description 2"),
            Candidate(entity_id="E3", score=0.7, description="Description 3"),
            Candidate(entity_id="E4", score=0.6, description="Description 4"),
            Candidate(entity_id="E5", score=0.5, description="Description 5"),
        ]

    @pytest.fixture
    def sample_mention(self) -> Mention:
        return Mention(start=0, end=5, text="Obama")

    @pytest.fixture
    def sample_doc(self) -> Document:
        return Document(
            id="test-doc",
            text="Obama was the president."
        )

    @patch("ner_pipeline.rerankers.lela_embedder.embedder_pool")
    def test_rerank_returns_candidates(
        self, mock_pool, sample_candidates, sample_mention, sample_doc
    ):
        # Mock embeddings
        mock_pool.embed.return_value = [
            [0.1, 0.2, 0.3],  # query
            [0.2, 0.3, 0.4],  # E1
            [0.3, 0.4, 0.5],  # E2
            [0.4, 0.5, 0.6],  # E3
            [0.5, 0.6, 0.7],  # E4
            [0.6, 0.7, 0.8],  # E5
        ]

        from ner_pipeline.rerankers.lela_embedder import LELAEmbedderReranker
        reranker = LELAEmbedderReranker(top_k=3)

        result = reranker.rerank(sample_mention, sample_candidates, sample_doc)

        assert len(result) == 3
        assert all(isinstance(c, Candidate) for c in result)

    @patch("ner_pipeline.rerankers.lela_embedder.embedder_pool")
    def test_rerank_respects_top_k(
        self, mock_pool, sample_candidates, sample_mention, sample_doc
    ):
        mock_pool.embed.return_value = [[0.1, 0.2, 0.3]] * 6

        from ner_pipeline.rerankers.lela_embedder import LELAEmbedderReranker
        reranker = LELAEmbedderReranker(top_k=2)

        result = reranker.rerank(sample_mention, sample_candidates, sample_doc)

        assert len(result) == 2

    @patch("ner_pipeline.rerankers.lela_embedder.embedder_pool")
    def test_rerank_returns_all_if_fewer_than_top_k(
        self, mock_pool, sample_mention, sample_doc
    ):
        candidates = [
            Candidate(entity_id="E1", score=0.9, description="Desc 1"),
        ]

        from ner_pipeline.rerankers.lela_embedder import LELAEmbedderReranker
        reranker = LELAEmbedderReranker(top_k=5)

        result = reranker.rerank(sample_mention, candidates, sample_doc)

        # Should return all candidates without calling embeddings
        assert len(result) == 1

    @patch("ner_pipeline.rerankers.lela_embedder.embedder_pool")
    def test_rerank_empty_candidates(
        self, mock_pool, sample_mention, sample_doc
    ):
        from ner_pipeline.rerankers.lela_embedder import LELAEmbedderReranker
        reranker = LELAEmbedderReranker(top_k=3)

        result = reranker.rerank(sample_mention, [], sample_doc)

        assert result == []

    @patch("ner_pipeline.rerankers.lela_embedder.embedder_pool")
    def test_rerank_sorts_by_similarity(
        self, mock_pool, sample_candidates, sample_mention, sample_doc
    ):
        # Create embeddings where E3 is most similar to query
        query_emb = [1.0, 0.0, 0.0]
        mock_pool.embed.return_value = [
            query_emb,
            [0.1, 0.9, 0.0],  # E1 - low similarity
            [0.2, 0.8, 0.0],  # E2
            [0.9, 0.1, 0.0],  # E3 - high similarity
            [0.3, 0.7, 0.0],  # E4
            [0.4, 0.6, 0.0],  # E5
        ]

        from ner_pipeline.rerankers.lela_embedder import LELAEmbedderReranker
        reranker = LELAEmbedderReranker(top_k=3)

        result = reranker.rerank(sample_mention, sample_candidates, sample_doc)

        # E3 should be first (highest similarity)
        assert result[0].entity_id == "E3"

    @patch("ner_pipeline.rerankers.lela_embedder.embedder_pool")
    def test_rerank_updates_scores(
        self, mock_pool, sample_candidates, sample_mention, sample_doc
    ):
        mock_pool.embed.return_value = [
            [1.0, 0.0, 0.0],  # query
            [0.8, 0.2, 0.0],  # E1
            [0.7, 0.3, 0.0],  # E2
            [0.6, 0.4, 0.0],  # E3
            [0.5, 0.5, 0.0],  # E4
            [0.4, 0.6, 0.0],  # E5
        ]

        from ner_pipeline.rerankers.lela_embedder import LELAEmbedderReranker
        reranker = LELAEmbedderReranker(top_k=3)

        result = reranker.rerank(sample_mention, sample_candidates, sample_doc)

        # Scores should be updated (not original scores)
        # All scores should be between -1 and 1 (cosine similarity)
        for c in result:
            assert c.score is not None
            assert -1.0 <= c.score <= 1.0

    @patch("ner_pipeline.rerankers.lela_embedder.embedder_pool")
    def test_query_includes_marked_mention(
        self, mock_pool, sample_candidates, sample_mention, sample_doc
    ):
        embed_calls = []
        def capture_embed(texts, **kwargs):
            embed_calls.append(texts)
            return [[0.1, 0.2, 0.3]] * len(texts)
        mock_pool.embed.side_effect = capture_embed

        from ner_pipeline.rerankers.lela_embedder import LELAEmbedderReranker
        reranker = LELAEmbedderReranker(top_k=3)

        reranker.rerank(sample_mention, sample_candidates, sample_doc)

        # First text in embed call is the query
        query_text = embed_calls[0][0]
        assert "[Obama]" in query_text  # Mention should be marked
        assert "Instruct:" in query_text

    @patch("ner_pipeline.rerankers.lela_embedder.embedder_pool")
    def test_candidates_formatted_for_embedding(
        self, mock_pool, sample_mention, sample_doc
    ):
        candidates = [
            Candidate(entity_id="Entity A", score=0.9, description="Description A"),
            Candidate(entity_id="Entity B", score=0.8, description="Description B"),
            Candidate(entity_id="Entity C", score=0.7, description=None),  # No description
        ]

        embed_calls = []
        def capture_embed(texts, **kwargs):
            embed_calls.append(texts)
            return [[0.1, 0.2, 0.3]] * len(texts)
        mock_pool.embed.side_effect = capture_embed

        from ner_pipeline.rerankers.lela_embedder import LELAEmbedderReranker
        reranker = LELAEmbedderReranker(top_k=2)

        reranker.rerank(sample_mention, candidates, sample_doc)

        # Check candidate formatting
        texts = embed_calls[0]
        # Query + 3 candidates = 4 texts
        assert len(texts) == 4
        assert "Entity A: Description A" in texts[1]
        assert "Entity B: Description B" in texts[2]
        assert "Entity C" in texts[3]  # No description

    @patch("ner_pipeline.rerankers.lela_embedder.embedder_pool")
    def test_preserves_descriptions(
        self, mock_pool, sample_candidates, sample_mention, sample_doc
    ):
        mock_pool.embed.return_value = [[0.1, 0.2, 0.3]] * 6

        from ner_pipeline.rerankers.lela_embedder import LELAEmbedderReranker
        reranker = LELAEmbedderReranker(top_k=3)

        result = reranker.rerank(sample_mention, sample_candidates, sample_doc)

        # Descriptions should be preserved
        for c in result:
            original = next(o for o in sample_candidates if o.entity_id == c.entity_id)
            assert c.description == original.description

    def test_initialization_with_custom_params(self):
        from ner_pipeline.rerankers.lela_embedder import LELAEmbedderReranker
        reranker = LELAEmbedderReranker(
            model_name="custom-model",
            top_k=5,
            base_url="http://custom-host",
            port=9000,
        )

        assert reranker.model_name == "custom-model"
        assert reranker.top_k == 5
        assert reranker.base_url == "http://custom-host"
        assert reranker.port == 9000
