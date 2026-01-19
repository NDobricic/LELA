"""
LELA-style dense candidate generator using FAISS and OpenAI-compatible embeddings.

This uses an OpenAI-compatible embedding API (e.g., vLLM serving Qwen3-Embedding)
for encoding and FAISS for fast similarity search.
"""

import logging
from typing import List, Optional

import numpy as np

from ner_pipeline.registry import candidate_generators
from ner_pipeline.types import Candidate, Document, Mention
from ner_pipeline.knowledge_bases.base import KnowledgeBase
from ner_pipeline.lela.config import (
    CANDIDATES_TOP_K,
    DEFAULT_EMBEDDER_MODEL,
    RETRIEVER_TASK,
)
from ner_pipeline.lela.llm_pool import embedder_pool

logger = logging.getLogger(__name__)

# Lazy import for faiss
_faiss = None


def _get_faiss():
    """Lazy import of faiss."""
    global _faiss
    if _faiss is None:
        try:
            import faiss
            _faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu package required for lela_dense. "
                "Install with: pip install faiss-cpu"
            )
    return _faiss


@candidate_generators.register("lela_dense")
class LELADenseCandidateGenerator:
    """
    Dense retrieval candidate generator using OpenAI-compatible embeddings.

    Uses an external embedding service (e.g., vLLM) accessed via OpenAI API,
    and FAISS for fast nearest neighbor search.

    Config options:
        model_name: Embedding model name (default: Qwen/Qwen3-Embedding-4B)
        top_k: Number of candidates to retrieve (default: 64)
        base_url: Base URL for embedding service (default: http://localhost)
        port: Port for embedding service (default: 8000)
        use_context: Whether to include context in query
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        model_name: str = DEFAULT_EMBEDDER_MODEL,
        top_k: int = CANDIDATES_TOP_K,
        base_url: str = "http://localhost",
        port: int = 8000,
        use_context: bool = True,
    ):
        if kb is None:
            raise ValueError("LELA dense retrieval requires a knowledge base.")

        self.kb = kb
        self.model_name = model_name
        self.top_k = top_k
        self.base_url = base_url
        self.port = port
        self.use_context = use_context

        faiss = _get_faiss()

        # Get entities
        self.entities = list(kb.all_entities())
        if not self.entities:
            raise ValueError("Knowledge base is empty.")

        # Build entity text corpus for embedding
        self.entity_texts = [
            f"{e.title} {e.description or ''}" for e in self.entities
        ]

        logger.info(f"Building dense index over {len(self.entities)} entities")
        logger.info(f"Using embedding model: {model_name} at {base_url}:{port}")

        # Embed all entities
        embeddings = self._embed_texts(self.entity_texts)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-9)

        # Build FAISS index (inner product = cosine after normalization)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        logger.info(f"Dense index built: {self.index.ntotal} vectors, dim={dim}")

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using the embedding service."""
        return embedder_pool.embed(
            texts,
            model_name=self.model_name,
            base_url=self.base_url,
            port=self.port,
        )

    @staticmethod
    def _format_query(mention_text: str, context: Optional[str] = None) -> str:
        """Format query with task instruction for embedding model."""
        query = mention_text
        if context:
            query = f"{mention_text}: {context}"
        return f"Instruct: {RETRIEVER_TASK}\nQuery: {query}"

    def generate(self, mention: Mention, doc: Document) -> List[Candidate]:
        """
        Generate candidates using dense retrieval.

        Args:
            mention: The mention to find candidates for
            doc: The source document

        Returns:
            List of Candidate objects
        """
        # Build query
        context = mention.context if self.use_context else None
        query_text = self._format_query(mention.text, context)

        # Embed query
        query_embedding = self._embed_texts([query_text])
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Normalize
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        # Search
        k = min(self.top_k, len(self.entities))
        scores, indices = self.index.search(query_embedding, k)

        # Build candidates
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            entity = self.entities[int(idx)]
            candidates.append(
                Candidate(
                    entity_id=entity.id,
                    score=float(score),
                    description=entity.description,
                )
            )

        logger.debug(f"Dense-retrieved {len(candidates)} candidates for '{mention.text}'")
        return candidates
