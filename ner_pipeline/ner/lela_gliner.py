"""
LELA-style GLiNER NER.

Uses the GLiNER library for zero-shot named entity recognition with
configurable labels and thresholds.
"""

import logging
from typing import List, Optional

from ner_pipeline.registry import ner_models
from ner_pipeline.types import Mention
from ner_pipeline.context import extract_context
from ner_pipeline.lela.config import (
    DEFAULT_GLINER_MODEL,
    NER_LABELS,
)

logger = logging.getLogger(__name__)

# Lazy import for GLiNER
_GLiNER = None


def _get_gliner():
    """Lazy import of GLiNER."""
    global _GLiNER
    if _GLiNER is None:
        try:
            from gliner import GLiNER
            _GLiNER = GLiNER
        except ImportError:
            raise ImportError(
                "gliner package required for lela_gliner. "
                "Install with: pip install gliner"
            )
    return _GLiNER


@ner_models.register("lela_gliner")
class LELAGLiNERNER:
    """
    Zero-shot GLiNER NER with LELA defaults.

    Uses the GLiNER library directly for reliable model loading
    and inference.

    Config options:
        model_name: GLiNER model to use (default: numind/NuNER_Zero-span)
        labels: List of entity labels to extract
        threshold: Confidence threshold for predictions (default: 0.5)
        context_mode: How to extract context ("sentence" or "window")
    """

    def __init__(
        self,
        model_name: str = DEFAULT_GLINER_MODEL,
        labels: Optional[List[str]] = None,
        threshold: float = 0.5,
        context_mode: str = "sentence",
    ):
        self.model_name = model_name
        self.labels = labels if labels is not None else list(NER_LABELS)
        self.threshold = threshold
        self.context_mode = context_mode

        GLiNER = _get_gliner()

        logger.info(f"Loading LELA GLiNER model: {model_name}")
        self.model = GLiNER.from_pretrained(model_name)
        logger.info(f"LELA GLiNER loaded with labels: {self.labels}")

    def extract(self, text: str) -> List[Mention]:
        """
        Extract entity mentions from text.

        Args:
            text: Input text

        Returns:
            List of Mention objects
        """
        if not text or not text.strip():
            return []

        predictions = self.model.predict_entities(
            text,
            labels=self.labels,
            threshold=self.threshold,
        )

        mentions = []
        for pred in predictions:
            start = pred["start"]
            end = pred["end"]

            context = extract_context(text, start, end, mode=self.context_mode)

            mentions.append(
                Mention(
                    start=start,
                    end=end,
                    text=pred["text"],
                    label=pred.get("label"),
                    context=context,
                )
            )

        logger.debug(f"Extracted {len(mentions)} mentions from text")
        return mentions
