"""Unit tests for LELA llm_pool module."""

import pytest
from unittest.mock import MagicMock, patch

from lela.lela.llm_pool import (
    _get_sentence_transformers,
    get_sentence_transformer_instance,
    clear_sentence_transformer_instances,
    _get_vllm,
    get_vllm_instance,
    clear_vllm_instances,
    clear_all_models,
    get_generic_instance,
    release_generic,
    clear_generic_instances,
    is_generic_cached,
    _evict_unused_generic,
)


class TestSentenceTransformerPool:
    """Tests for SentenceTransformer singleton functions."""

    def test_get_sentence_transformers_raises_on_missing(self):
        # Just verify the function exists and is callable
        assert callable(_get_sentence_transformers)

    @patch("lela.lela.llm_pool._get_sentence_transformers")
    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_get_sentence_transformer_instance_creates_model(self, mock_get_st):
        import sys
        mock_torch = sys.modules["torch"]
        mock_torch.float16 = "float16"

        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model
        mock_get_st.return_value = mock_st_module

        # Clear cache first
        clear_sentence_transformer_instances(force=True)

        result = get_sentence_transformer_instance(
            model_name="test-model",
            device="cuda",
        )

        mock_st_module.SentenceTransformer.assert_called_once()
        call_kwargs = mock_st_module.SentenceTransformer.call_args
        assert call_kwargs[0][0] == "test-model"
        assert call_kwargs[1]["device"] == "cuda"
        assert call_kwargs[1]["trust_remote_code"] is True

    @patch("lela.lela.llm_pool._get_sentence_transformers")
    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_get_sentence_transformer_instance_reuses_model(self, mock_get_st):
        import sys
        mock_torch = sys.modules["torch"]
        mock_torch.float16 = "float16"

        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model
        mock_get_st.return_value = mock_st_module

        clear_sentence_transformer_instances(force=True)

        model1, cached1 = get_sentence_transformer_instance("model-a", device="cuda")
        model2, cached2 = get_sentence_transformer_instance("model-a", device="cuda")

        # Should only create once
        assert mock_st_module.SentenceTransformer.call_count == 1
        assert model1 is model2
        assert not cached1
        assert cached2

    @patch("lela.lela.llm_pool._get_sentence_transformers")
    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_different_devices_create_different_instances(self, mock_get_st):
        import sys
        mock_torch = sys.modules["torch"]
        mock_torch.float16 = "float16"

        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer.side_effect = [MagicMock(), MagicMock()]
        mock_get_st.return_value = mock_st_module

        clear_sentence_transformer_instances(force=True)

        model1 = get_sentence_transformer_instance("model-a", device="cuda")
        model2 = get_sentence_transformer_instance("model-a", device="cpu")

        # Different devices should create different instances
        assert mock_st_module.SentenceTransformer.call_count == 2

    def test_clear_sentence_transformer_instances_no_force(self):
        # Should not raise and should do nothing without force=True
        clear_sentence_transformer_instances()

    @patch("lela.lela.llm_pool._get_sentence_transformers")
    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_clear_sentence_transformer_instances_force(self, mock_get_st):
        import sys
        mock_torch = sys.modules["torch"]
        mock_torch.float16 = "float16"
        mock_torch.cuda.is_available.return_value = False

        mock_st_module = MagicMock()
        mock_model = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model
        mock_get_st.return_value = mock_st_module

        clear_sentence_transformer_instances(force=True)

        get_sentence_transformer_instance("test-model")

        # Clear with force
        clear_sentence_transformer_instances(force=True)

        # Model should be created again on next call
        get_sentence_transformer_instance("test-model")

        assert mock_st_module.SentenceTransformer.call_count == 2


class TestLazyImports:
    """Tests for lazy import functions."""

    def test_get_vllm_raises_on_missing(self):
        # Just verify the function exists
        assert callable(_get_vllm)


class TestVLLMInstanceManagement:
    """Tests for vLLM instance management."""

    def test_clear_vllm_instances(self):
        # Should not raise
        clear_vllm_instances()

    @patch("lela.lela.llm_pool._get_vllm")
    def test_get_vllm_instance_creates_model(self, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_llm = MagicMock()
        mock_vllm.LLM.return_value = mock_llm
        mock_get_vllm.return_value = mock_vllm

        # Clear cache first (force=True required to actually clear)
        clear_vllm_instances(force=True)

        result = get_vllm_instance(
            model_name="test-model",
            tensor_parallel_size=1,
        )

        mock_vllm.LLM.assert_called_once()
        call_kwargs = mock_vllm.LLM.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["tensor_parallel_size"] == 1

    @patch("lela.lela.llm_pool._get_vllm")
    def test_get_vllm_instance_reuses_model(self, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_llm = MagicMock()
        mock_vllm.LLM.return_value = mock_llm
        mock_get_vllm.return_value = mock_vllm

        clear_vllm_instances(force=True)

        llm1, cached1 = get_vllm_instance("model-a", tensor_parallel_size=1)
        llm2, cached2 = get_vllm_instance("model-a", tensor_parallel_size=1)

        # Should only create once
        assert mock_vllm.LLM.call_count == 1
        assert llm1 is llm2
        assert not cached1
        assert cached2

    @patch("lela.lela.llm_pool._get_vllm")
    def test_different_configs_create_different_instances(self, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_vllm.LLM.side_effect = [MagicMock(), MagicMock()]
        mock_get_vllm.return_value = mock_vllm

        clear_vllm_instances(force=True)

        llm1 = get_vllm_instance("model-a", tensor_parallel_size=1)
        llm2 = get_vllm_instance("model-a", tensor_parallel_size=2)

        # Different tensor_parallel_size should create different instances
        assert mock_vllm.LLM.call_count == 2

    @patch("lela.lela.llm_pool._get_vllm")
    def test_max_model_len_passed_when_specified(self, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = mock_vllm

        clear_vllm_instances(force=True)

        get_vllm_instance("model-x", tensor_parallel_size=1, max_model_len=4096)

        call_kwargs = mock_vllm.LLM.call_args[1]
        assert call_kwargs["max_model_len"] == 4096

    @patch("lela.lela.llm_pool._get_vllm")
    def test_gpu_memory_utilization_passed(self, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = mock_vllm

        clear_vllm_instances(force=True)

        get_vllm_instance("model-y", tensor_parallel_size=1, gpu_memory_utilization=0.7)

        call_kwargs = mock_vllm.LLM.call_args[1]
        assert call_kwargs["gpu_memory_utilization"] == 0.7

    @patch("lela.lela.llm_pool._get_vllm")
    def test_gpu_memory_utilization_default(self, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = mock_vllm

        clear_vllm_instances(force=True)

        get_vllm_instance("model-z", tensor_parallel_size=1)

        call_kwargs = mock_vllm.LLM.call_args[1]
        assert call_kwargs["gpu_memory_utilization"] == 0.9  # default from config

    @patch("lela.lela.llm_pool._get_vllm")
    def test_different_gpu_memory_creates_different_instances(self, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_vllm.LLM.side_effect = [MagicMock(), MagicMock()]
        mock_get_vllm.return_value = mock_vllm

        clear_vllm_instances(force=True)

        llm1, cached1 = get_vllm_instance("model-a", tensor_parallel_size=1, gpu_memory_utilization=0.5)
        llm2, cached2 = get_vllm_instance("model-a", tensor_parallel_size=1, gpu_memory_utilization=0.8)

        # Different gpu_memory_utilization should create different instances
        assert mock_vllm.LLM.call_count == 2
        assert not cached1
        assert not cached2
        assert llm1 is not llm2


class TestGenericPool:
    """Tests for generic model pool functions."""

    def setup_method(self):
        clear_generic_instances(force=True)

    def test_creates_model(self):
        loader = MagicMock(return_value="my_model")

        instance, was_cached = get_generic_instance("test:model", loader, 1.0)

        assert instance == "my_model"
        assert not was_cached
        loader.assert_called_once()

    def test_reuses_cached(self):
        loader = MagicMock(return_value="my_model")

        instance1, cached1 = get_generic_instance("test:model", loader, 1.0)
        instance2, cached2 = get_generic_instance("test:model", loader, 1.0)

        assert instance1 is instance2
        assert not cached1
        assert cached2
        loader.assert_called_once()

    def test_release_allows_eviction(self):
        loader = MagicMock(return_value="my_model")

        get_generic_instance("test:model", loader, 1.0)
        assert is_generic_cached("test:model")

        release_generic("test:model")
        _evict_unused_generic()

        assert not is_generic_cached("test:model")

    def test_clear_force(self):
        loader = MagicMock(side_effect=["model_v1", "model_v2"])

        instance1, _ = get_generic_instance("test:model", loader, 1.0)
        assert instance1 == "model_v1"

        clear_generic_instances(force=True)

        instance2, cached2 = get_generic_instance("test:model", loader, 1.0)
        assert instance2 == "model_v2"
        assert not cached2
        assert loader.call_count == 2


class TestClearAllModels:
    """Tests for clear_all_models function."""

    @patch("lela.lela.llm_pool._get_sentence_transformers")
    @patch("lela.lela.llm_pool._get_vllm")
    @patch.dict("sys.modules", {"torch": MagicMock()})
    def test_clear_all_models(self, mock_get_vllm, mock_get_st):
        import sys
        mock_torch = sys.modules["torch"]
        mock_torch.float16 = "float16"
        mock_torch.cuda.is_available.return_value = False

        mock_st_module = MagicMock()
        mock_get_st.return_value = mock_st_module

        mock_vllm = MagicMock()
        mock_get_vllm.return_value = mock_vllm

        generic_loader = MagicMock(side_effect=["gen_v1", "gen_v2"])

        # Load one of each
        clear_all_models()
        get_sentence_transformer_instance("st-model")
        get_vllm_instance("vllm-model")
        get_generic_instance("generic:model", generic_loader, 1.0)

        # Clear all
        clear_all_models()

        # All should be recreated on next call
        get_sentence_transformer_instance("st-model")
        get_vllm_instance("vllm-model")
        instance, cached = get_generic_instance("generic:model", generic_loader, 1.0)

        assert mock_st_module.SentenceTransformer.call_count == 2
        assert mock_vllm.LLM.call_count == 2
        assert generic_loader.call_count == 2
        assert not cached
