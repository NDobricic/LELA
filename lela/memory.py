"""
System resource detection for LELA.

Provides utilities to detect available GPU and system memory.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SystemResources:
    """Available system resources."""

    gpu_available: bool
    gpu_name: Optional[str]
    gpu_vram_total_gb: float
    gpu_vram_free_gb: float
    ram_total_gb: float
    ram_available_gb: float


def get_system_resources() -> SystemResources:
    """Detect available system resources (GPU VRAM and system RAM)."""
    import psutil

    # Get RAM info
    ram = psutil.virtual_memory()
    ram_total_gb = ram.total / (1024**3)
    ram_available_gb = ram.available / (1024**3)

    # Get GPU info
    gpu_available = False
    gpu_name = None
    gpu_vram_total_gb = 0.0
    gpu_vram_free_gb = 0.0

    try:
        import torch

        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_vram_total_gb = props.total_memory / (1024**3)

            # Get free memory
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            gpu_vram_free_gb = free_mem / (1024**3)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error detecting GPU: {e}")

    return SystemResources(
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_vram_total_gb=gpu_vram_total_gb,
        gpu_vram_free_gb=gpu_vram_free_gb,
        ram_total_gb=ram_total_gb,
        ram_available_gb=ram_available_gb,
    )


def gb_to_vllm_fraction(gb: float) -> float:
    """Convert a GB memory limit to a vLLM gpu_memory_utilization fraction.

    Uses the total GPU VRAM to compute the fraction, clamped to [0.05, 0.95].
    Falls back to 0.9 if GPU detection fails.
    """
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / (1024**3)
            if total_gb > 0:
                return min(max(gb / total_gb, 0.05), 0.95)
    except Exception:
        pass
    return 0.9
