"""
Singleton pool for managing expensive LLM and embedder instances.

This module provides lazy initialization and reuse of:
- SentenceTransformer instances for embeddings
- vLLM instances for text generation
"""

import logging
import os
from typing import Dict, List, Optional, Any

# Disable vLLM V1 engine and configure multiprocessing to work from worker threads
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

logger = logging.getLogger(__name__)


# Placeholder for SentenceTransformers - imported lazily
_sentence_transformers_module = None
_sentence_transformer_instances: Dict[str, Any] = {}


def _get_sentence_transformers():
    """Lazy import of sentence_transformers module."""
    global _sentence_transformers_module
    if _sentence_transformers_module is None:
        try:
            import sentence_transformers
            _sentence_transformers_module = sentence_transformers
        except ImportError:
            raise ImportError(
                "sentence-transformers package required for embedding. "
                "Install with: pip install sentence-transformers"
            )
    return _sentence_transformers_module


def get_sentence_transformer_instance(
    model_name: str,
    device: Optional[str] = None,
):
    """
    Get or create a SentenceTransformer instance.

    Args:
        model_name: HuggingFace model ID (e.g., 'Qwen/Qwen3-Embedding-4B')
        device: Device to load model on ('cuda', 'cpu', or None for auto)

    Returns:
        SentenceTransformer instance
    """
    key = f"{model_name}:{device or 'auto'}"

    if key not in _sentence_transformer_instances:
        sentence_transformers = _get_sentence_transformers()

        logger.info(f"Loading SentenceTransformer model: {model_name}")

        import torch
        model_kwargs = {"torch_dtype": torch.float16}

        model = sentence_transformers.SentenceTransformer(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
            trust_remote_code=True,
        )

        _sentence_transformer_instances[key] = model
        logger.info(f"SentenceTransformer model loaded: {model_name}")

    return _sentence_transformer_instances[key]


def clear_sentence_transformer_instances(force: bool = False):
    """
    Clear all cached SentenceTransformer instances.

    Args:
        force: If True, actually delete instances and free GPU memory.
               If False (default), do nothing - instances should be reused.

    Note: SentenceTransformer instances are expensive to create and should be
    reused across pipeline runs. Only use force=True when shutting down.
    """
    global _sentence_transformer_instances

    if not force:
        return

    for key in list(_sentence_transformer_instances.keys()):
        try:
            logger.info(f"Shutting down SentenceTransformer instance: {key}")
            del _sentence_transformer_instances[key]
        except Exception as e:
            logger.warning(f"Error cleaning up SentenceTransformer instance {key}: {e}")

    _sentence_transformer_instances.clear()

    import gc
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Error clearing CUDA cache: {e}")


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
            "dtype": "half",  # float16 for P100 (compute capability 6.0)
            "max_model_len": max_model_len or 4096,  # Qwen3-4B supports up to 32K
            "gpu_memory_utilization": 0.9,
            "trust_remote_code": True,  # Required for Qwen models to load tokenizer/chat template
            **kwargs,
        }

        logger.info(f"Loading vLLM model: {model_name}")
        _vllm_instances[key] = vllm.LLM(**llm_kwargs)
        logger.info(f"vLLM model loaded: {model_name}")

    return _vllm_instances[key]


def clear_vllm_instances(force: bool = False):
    """
    Clear all cached vLLM instances.

    Args:
        force: If True, actually delete instances and free GPU memory.
               If False (default), do nothing - instances should be reused.

    Note: vLLM instances are expensive to create and should be reused across
    pipeline runs. Only use force=True when shutting down the application.
    """
    global _vllm_instances

    if not force:
        # Don't clear - instances should be reused between runs
        return

    # Actually clean up vLLM instances
    for key, instance in list(_vllm_instances.items()):
        try:
            logger.info(f"Shutting down vLLM instance: {key}")
            del instance
        except Exception as e:
            logger.warning(f"Error cleaning up vLLM instance {key}: {e}")

    _vllm_instances.clear()

    # Force garbage collection and clear CUDA cache
    import gc
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"Error clearing CUDA cache: {e}")
