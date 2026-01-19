"""
LELA-style BM25 candidate generator using bm25s with Stemmer.

This uses the bm25s library for fast BM25 retrieval with optional
stemming support via PyStemmer.
"""

import logging
from typing import List, Optional

from ner_pipeline.registry import candidate_generators
from ner_pipeline.types import Candidate, Document, Mention
from ner_pipeline.knowledge_bases.base import KnowledgeBase
from ner_pipeline.lela.config import CANDIDATES_TOP_K

logger = logging.getLogger(__name__)

# Lazy imports
_bm25s = None
_Stemmer = None


def _get_bm25s():
    """Lazy import of bm25s."""
    global _bm25s
    if _bm25s is None:
        try:
            import bm25s
            _bm25s = bm25s
        except ImportError:
            raise ImportError(
                "bm25s package required for lela_bm25. "
                "Install with: pip install 'bm25s[full]'"
            )
    return _bm25s


def _get_stemmer():
    """Lazy import of Stemmer."""
    global _Stemmer
    if _Stemmer is None:
        try:
            import Stemmer
            _Stemmer = Stemmer
        except ImportError:
            raise ImportError(
                "PyStemmer package required for lela_bm25. "
                "Install with: pip install PyStemmer"
            )
    return _Stemmer


@candidate_generators.register("lela_bm25")
class LELABM25CandidateGenerator:
    """
    BM25 candidate generator using bm25s library.

    Uses Stemmer tokenization for better matching and the numba
    backend for fast retrieval.

    Config options:
        top_k: Number of candidates to retrieve (default: 64)
        use_context: Whether to include mention context in query
        stemmer_language: Language for stemmer (default: "english")
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        top_k: int = CANDIDATES_TOP_K,
        use_context: bool = True,
        stemmer_language: str = "english",
    ):
        if kb is None:
            raise ValueError("LELA BM25 requires a knowledge base.")

        self.kb = kb
        self.top_k = top_k
        self.use_context = use_context
        self.stemmer_language = stemmer_language

        bm25s = _get_bm25s()
        Stemmer = _get_stemmer()

        # Get entities and build corpus
        self.entities = list(kb.all_entities())

        if not self.entities:
            raise ValueError("Knowledge base is empty.")

        # Build corpus: title + description for each entity
        self.corpus_records = []
        corpus_texts = []
        for entity in self.entities:
            record = {
                "title": entity.title,
                "description": entity.description or "",
            }
            self.corpus_records.append(record)
            corpus_texts.append(f"{entity.title} {entity.description or ''}")

        logger.info(f"Building BM25 index over {len(corpus_texts)} entities")

        # Create stemmer and tokenize corpus
        self.stemmer = Stemmer.Stemmer(stemmer_language)
        self.tokenizer = bm25s.tokenization.Tokenizer(stemmer=self.stemmer)
        corpus_tokens = self.tokenizer.tokenize(corpus_texts, return_as="tuple")

        # Build BM25 index
        self.retriever = bm25s.BM25(corpus=self.corpus_records, backend="numba")
        self.retriever.index(corpus_tokens)

        logger.info("BM25 index built successfully")

    def generate(self, mention: Mention, doc: Document) -> List[Candidate]:
        """
        Generate candidates for a mention using BM25 retrieval.

        Args:
            mention: The mention to find candidates for
            doc: The source document

        Returns:
            List of Candidate objects
        """
        bm25s = _get_bm25s()

        # Build query text
        if self.use_context and mention.context:
            query_text = f"{mention.text} {mention.context}"
        else:
            query_text = mention.text

        # Tokenize query
        query_tokens = bm25s.tokenize(
            [query_text],
            stemmer=self.stemmer,
            return_ids=False,
        )

        # Handle empty tokenization
        if not query_tokens[0]:
            logger.debug(f"Empty tokenized query for mention '{mention.text}'")
            return []

        # Retrieve candidates
        results = self.retriever.retrieve(query_tokens, k=min(self.top_k, len(self.entities)))
        candidates_docs = results.documents[0]
        scores = results.scores[0] if hasattr(results, 'scores') else [None] * len(candidates_docs)

        candidates = []
        for i, record in enumerate(candidates_docs):
            score = float(scores[i]) if scores[i] is not None else None
            candidates.append(
                Candidate(
                    entity_id=record["title"],
                    score=score,
                    description=record["description"],
                )
            )

        logger.debug(f"Retrieved {len(candidates)} candidates for '{mention.text}'")
        return candidates
