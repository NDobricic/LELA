"""
LELA-style vLLM disambiguator.

Uses vLLM for fast batched inference to select the best entity
from a list of candidates based on context.
"""

import logging
import re
from collections import Counter
from typing import List, Optional, Tuple

from ner_pipeline.registry import disambiguators
from ner_pipeline.types import Candidate, Document, Entity, Mention
from ner_pipeline.knowledge_bases.base import KnowledgeBase
from ner_pipeline.lela.config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_TENSOR_PARALLEL_SIZE,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_GENERATION_CONFIG,
    NOT_AN_ENTITY,
    SPAN_OPEN,
    SPAN_CLOSE,
)
from ner_pipeline.lela.prompts import (
    create_disambiguation_messages,
    DEFAULT_SYSTEM_PROMPT,
)
from ner_pipeline.lela.llm_pool import get_vllm_instance

logger = logging.getLogger(__name__)

# Lazy imports
_vllm = None
_SamplingParams = None


def _get_vllm():
    """Lazy import of vllm."""
    global _vllm, _SamplingParams
    if _vllm is None:
        try:
            import vllm
            from vllm import SamplingParams
            _vllm = vllm
            _SamplingParams = SamplingParams
        except ImportError:
            raise ImportError(
                "vllm package required for lela_vllm disambiguator. "
                "Install with: pip install vllm"
            )
    return _vllm, _SamplingParams


@disambiguators.register("lela_vllm")
class LELAvLLMDisambiguator:
    """
    vLLM-based entity disambiguator.

    Uses vLLM for efficient batched LLM inference to select the
    correct entity from candidates based on context.

    Config options:
        model_name: LLM model to use (default: Qwen/Qwen3-8B)
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_model_len: Maximum sequence length
        add_none_candidate: Whether to include "None" option
        add_descriptions: Whether to include entity descriptions in prompt
        disable_thinking: Disable thinking mode for supported models
        system_prompt: Custom system prompt for disambiguation
        self_consistency_k: Number of samples for self-consistency voting
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        model_name: str = DEFAULT_LLM_MODEL,
        tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
        max_model_len: Optional[int] = DEFAULT_MAX_MODEL_LEN,
        add_none_candidate: bool = False,
        add_descriptions: bool = True,
        disable_thinking: bool = False,
        system_prompt: Optional[str] = None,
        generation_config: Optional[dict] = None,
        self_consistency_k: int = 1,
    ):
        if kb is None:
            raise ValueError("LELA vLLM disambiguator requires a knowledge base.")

        self.kb = kb
        self.model_name = model_name
        self.add_none_candidate = add_none_candidate
        self.add_descriptions = add_descriptions
        self.disable_thinking = disable_thinking
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.self_consistency_k = self_consistency_k

        # Get vLLM and SamplingParams
        vllm, SamplingParams = _get_vllm()

        # Get or create LLM instance
        self.llm = get_vllm_instance(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
        )

        # Set up sampling parameters
        gen_config = generation_config or DEFAULT_GENERATION_CONFIG
        sampling_config = {**gen_config, "n": self_consistency_k}
        self.sampling_params = SamplingParams(**sampling_config)

        logger.info(f"LELA vLLM disambiguator initialized: {model_name}")

    @staticmethod
    def _parse_output(output: str) -> int:
        """Parse LLM output to extract answer index."""
        match = re.search(r'"?answer"?:\s*(\d+)', output)
        if match:
            return int(match.group(1))
        logger.debug(f"Unexpected output format: {output}")
        return 0

    def _apply_self_consistency(self, outputs: list) -> int:
        """Apply self-consistency voting over multiple outputs."""
        if self.self_consistency_k == 1:
            return self._parse_output(outputs[0].text)
        answers = [self._parse_output(o.text) for o in outputs]
        return Counter(answers).most_common(1)[0][0]

    def _mark_mention(self, text: str, start: int, end: int) -> str:
        """Mark mention in text with brackets."""
        return f"{text[:start]}{SPAN_OPEN}{text[start:end]}{SPAN_CLOSE}{text[end:]}"

    def disambiguate(
        self,
        mention: Mention,
        candidates: List[Candidate],
        doc: Document,
    ) -> Optional[Entity]:
        """
        Disambiguate a mention to select the best entity.

        Args:
            mention: The mention to disambiguate
            candidates: List of candidate entities
            doc: The source document

        Returns:
            The selected Entity, or None if no match
        """
        if not candidates:
            return None

        # If only one candidate and no none option, return it directly
        if len(candidates) == 1 and not self.add_none_candidate:
            return self.kb.get_entity(candidates[0].entity_id)

        # Format candidate tuples (entity_id, description)
        candidate_tuples: List[Tuple[str, str]] = [
            (c.entity_id, c.description or "")
            for c in candidates
        ]

        # Mark mention in text
        marked_text = self._mark_mention(doc.text, mention.start, mention.end)

        # Create messages for LLM
        messages = create_disambiguation_messages(
            marked_text=marked_text,
            candidates=candidate_tuples,
            system_prompt=self.system_prompt,
            add_none_candidate=self.add_none_candidate,
            add_descriptions=self.add_descriptions,
        )

        # Get chat template kwargs
        chat_kwargs = {}
        if self.disable_thinking:
            chat_kwargs["enable_thinking"] = False

        try:
            responses = self.llm.chat(
                [messages],
                sampling_params=self.sampling_params,
                use_tqdm=False,
                chat_template_kwargs=chat_kwargs if chat_kwargs else {},
            )
            response = responses[0] if responses else None
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return None

        if response is None:
            return None

        try:
            answer = self._apply_self_consistency(response.outputs)

            # Answer 0 means "None" if add_none_candidate is True
            # Otherwise, valid answers are 1-indexed
            if answer == 0:
                return None

            if 0 < answer <= len(candidates):
                selected = candidates[answer - 1]
                return self.kb.get_entity(selected.entity_id)
            else:
                logger.debug(f"Answer {answer} out of range for {len(candidates)} candidates")
                return None

        except Exception as e:
            logger.error(f"Error processing LLM response: {e}")
            return None
