"""
Singleton pool for managing expensive LLM and embedder instances.

This module provides lazy initialization and reuse of:
- OpenAI-compatible embedding clients
- vLLM instances for text generation
"""

import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Disable vLLM V1 engine and configure multiprocessing to work from worker threads
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

logger = logging.getLogger(__name__)

# Placeholder for OpenAI client - imported lazily
_openai_module = None


def _get_openai():
    """Lazy import of openai module."""
    global _openai_module
    if _openai_module is None:
        try:
            import openai
            _openai_module = openai
        except ImportError:
            raise ImportError(
                "openai package required for LELA embedder. "
                "Install with: pip install openai"
            )
    return _openai_module


@dataclass
class EmbedderConfig:
    """Configuration for an embedding client."""
    model_name: str
    base_url: str
    port: int
    api_key: str = "EMPTY"


class EmbedderPool:
    """Pool of OpenAI-compatible embedding clients."""

    _instance: Optional["EmbedderPool"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._clients: Dict[str, Any] = {}
        return cls._instance

    def get_client(
        self,
        model_name: str,
        base_url: str = "http://localhost",
        port: int = 8000,
        api_key: str = "EMPTY",
    ):
        """Get or create an embedding client."""
        key = f"{base_url}:{port}"

        if key not in self._clients:
            openai = _get_openai()
            api_base = f"{base_url}:{port}/v1"
            self._clients[key] = openai.OpenAI(
                api_key=api_key,
                base_url=api_base,
            )
            logger.info(f"Created embedding client for {api_base}")

        return self._clients[key]

    def embed(
        self,
        texts: List[str],
        model_name: str,
        base_url: str = "http://localhost",
        port: int = 8000,
    ) -> List[List[float]]:
        """Embed texts using the specified model."""
        client = self.get_client(model_name, base_url, port)
        response = client.embeddings.create(input=texts, model=model_name)
        return [data.embedding for data in response.data]


# Placeholder for vLLM - imported lazily
_vllm_module = None
_vllm_instances: Dict[str, Any] = {}


def _get_vllm():
    """Lazy import of vllm module."""
    global _vllm_module
    if _vllm_module is None:
        try:
            import vllm
            _vllm_module = vllm
        except ImportError:
            raise ImportError(
                "vllm package required for LELA vLLM disambiguator. "
                "Install with: pip install vllm"
            )
    return _vllm_module


def get_vllm_instance(
    model_name: str,
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
    **kwargs,
):
    """
    Get or create a vLLM LLM instance.

    Args:
        model_name: HuggingFace model ID
        tensor_parallel_size: Number of GPUs for tensor parallelism
        max_model_len: Maximum sequence length
        **kwargs: Additional vLLM arguments

    Returns:
        vLLM LLM instance
    """
    key = f"{model_name}:tp{tensor_parallel_size}"

    if key not in _vllm_instances:
        vllm = _get_vllm()

        llm_kwargs = {
            "model": model_name,
            "tensor_parallel_size": tensor_parallel_size,
            "enforce_eager": True,  # Disable CUDA graphs to avoid multiprocessing issues
            **kwargs,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        logger.info(f"Loading vLLM model: {model_name}")
        _vllm_instances[key] = vllm.LLM(**llm_kwargs)
        logger.info(f"vLLM model loaded: {model_name}")

    return _vllm_instances[key]


def clear_vllm_instances():
    """Clear all cached vLLM instances (for cleanup)."""
    global _vllm_instances
    _vllm_instances.clear()


# Global singleton instances
embedder_pool = EmbedderPool()
