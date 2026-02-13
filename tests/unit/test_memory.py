"""Unit tests for memory.py helpers."""

from unittest.mock import MagicMock, patch

import pytest

from el_pipeline.memory import gb_to_vllm_fraction


class TestGbToVllmFraction:
    """Tests for gb_to_vllm_fraction()."""

    def _make_mock_torch(self, total_gb: float, cuda_available: bool = True):
        """Create a mock torch module with given VRAM."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = cuda_available
        props = MagicMock()
        props.total_memory = total_gb * (1024**3)
        mock_torch.cuda.get_device_properties.return_value = props
        return mock_torch

    def test_basic_conversion(self):
        """GB value converts to correct fraction of total VRAM."""
        mock_torch = self._make_mock_torch(24.0)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = gb_to_vllm_fraction(12.0)
            assert abs(result - 0.5) < 0.01

    def test_clamps_to_max(self):
        """Result is clamped to 0.95 maximum."""
        mock_torch = self._make_mock_torch(24.0)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = gb_to_vllm_fraction(100.0)
            assert result == 0.95

    def test_clamps_to_min(self):
        """Result is clamped to 0.05 minimum."""
        mock_torch = self._make_mock_torch(24.0)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = gb_to_vllm_fraction(0.001)
            assert result == 0.05

    def test_no_gpu_returns_default(self):
        """Falls back to 0.9 when CUDA is not available."""
        mock_torch = self._make_mock_torch(24.0, cuda_available=False)
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = gb_to_vllm_fraction(10.0)
            assert result == 0.9

    def test_returns_float(self):
        """Always returns a float."""
        result = gb_to_vllm_fraction(10.0)
        assert isinstance(result, float)
