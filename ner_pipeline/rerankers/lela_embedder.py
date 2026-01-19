"""
LELA-style embedding reranker using cosine similarity.

Reranks candidates by computing cosine similarity between the
mention (with context) and candidate descriptions using an
OpenAI-compatible embedding API.
"""

import logging
from typing import List

import numpy as np

from ner_pipeline.registry import rerankers
from ner_pipeline.types import Candidate, Document, Mention
from ner_pipeline.lela.config import (
    RERANKER_TOP_K,
    DEFAULT_EMBEDDER_MODEL,
    RERANKER_TASK,
    SPAN_OPEN,
    SPAN_CLOSE,
)
from ner_pipeline.lela.llm_pool import embedder_pool

logger = logging.getLogger(__name__)


@rerankers.register("lela_embedder")
class LELAEmbedderReranker:
    """
    Embedding-based reranker using cosine similarity.

    Uses an OpenAI-compatible embedding API to encode the mention
    in context and all candidate descriptions, then reranks by
    cosine similarity.

    Config options:
        model_name: Embedding model name (default: Qwen/Qwen3-Embedding-4B)
        top_k: Number of candidates to keep after reranking (default: 10)
        base_url: Base URL for embedding service (default: http://localhost)
        port: Port for embedding service (default: 8000)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDER_MODEL,
        top_k: int = RERANKER_TOP_K,
        base_url: str = "http://localhost",
        port: int = 8000,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.base_url = base_url
        self.port = port

        logger.info(f"LELA embedder reranker initialized: {model_name}")

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using the embedding service."""
        return embedder_pool.embed(
            texts,
            model_name=self.model_name,
            base_url=self.base_url,
            port=self.port,
        )

    def _format_query(self, mention: Mention, doc: Document) -> str:
        """
        Format query with marked mention in context.

        Marks the mention with brackets in the full document text
        for better context understanding.
        """
        text = doc.text
        start, end = mention.start, mention.end

        # Mark mention in text
        marked_text = f"{text[:start]}{SPAN_OPEN}{text[start:end]}{SPAN_CLOSE}{text[end:]}"

        return f"Instruct: {RERANKER_TASK}\nQuery: {marked_text}"

    def _format_candidate(self, candidate: Candidate) -> str:
        """Format candidate for embedding."""
        if candidate.description:
            return f"{candidate.entity_id}: {candidate.description}"
        return candidate.entity_id

    def rerank(
        self,
        mention: Mention,
        candidates: List[Candidate],
        doc: Document,
    ) -> List[Candidate]:
        """
        Rerank candidates by cosine similarity with the mention.

        Args:
            mention: The mention being linked
            candidates: List of candidates to rerank
            doc: The source document

        Returns:
            Reranked list of candidates (top_k)
        """
        if not candidates:
            return candidates

        if len(candidates) <= self.top_k:
            return candidates

        # Format query and candidates
        query_text = self._format_query(mention, doc)
        candidate_texts = [self._format_candidate(c) for c in candidates]

        # Embed all texts in one batch
        all_texts = [query_text] + candidate_texts
        embeddings = self._embed_texts(all_texts)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-9)

        # Compute cosine similarities (query vs all candidates)
        query_embedding = embeddings[0:1]
        candidate_embeddings = embeddings[1:]
        similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()

        # Sort by similarity and take top_k
        scored_candidates = list(zip(candidates, similarities))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored_candidates[: self.top_k]

        # Return reranked candidates with updated scores
        result = [
            Candidate(
                entity_id=c.entity_id,
                score=float(score),
                description=c.description,
            )
            for c, score in top_candidates
        ]

        logger.debug(
            f"Reranked {len(candidates)} candidates to {len(result)} for '{mention.text}'"
        )
        return result
