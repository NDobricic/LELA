"""Entity disambiguators."""

import importlib.util
import logging

from .popularity import PopularityDisambiguator  # noqa: F401
from .first import FirstCandidateDisambiguator  # noqa: F401
from .llm import LLMDisambiguator  # noqa: F401

# Only import lela_vllm if vllm is installed and importable
if importlib.util.find_spec("vllm") is not None:
    try:
        from .lela_vllm import LELAvLLMDisambiguator  # noqa: F401
    except Exception as e:
        logging.getLogger(__name__).debug(f"lela_vllm disambiguator unavailable: {e}")

