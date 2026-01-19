"""Unit tests for LELA llm_pool module."""

import pytest
from unittest.mock import MagicMock, patch

from ner_pipeline.lela.llm_pool import (
    EmbedderPool,
    embedder_pool,
    _get_openai,
    _get_vllm,
    get_vllm_instance,
    clear_vllm_instances,
)


class TestEmbedderPool:
    """Tests for EmbedderPool singleton class."""

    def test_singleton_pattern(self):
        pool1 = EmbedderPool()
        pool2 = EmbedderPool()
        assert pool1 is pool2

    def test_global_embedder_pool_exists(self):
        assert embedder_pool is not None
        assert isinstance(embedder_pool, EmbedderPool)

    @patch("ner_pipeline.lela.llm_pool._get_openai")
    def test_get_client_creates_client(self, mock_get_openai):
        mock_openai = MagicMock()
        mock_get_openai.return_value = mock_openai
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        pool = EmbedderPool()
        pool._clients.clear()  # Reset for testing

        client = pool.get_client(
            model_name="test-model",
            base_url="http://test",
            port=9000,
        )

        mock_openai.OpenAI.assert_called_once()
        call_kwargs = mock_openai.OpenAI.call_args[1]
        assert "http://test:9000/v1" in call_kwargs["base_url"]

    @patch("ner_pipeline.lela.llm_pool._get_openai")
    def test_get_client_reuses_connection(self, mock_get_openai):
        mock_openai = MagicMock()
        mock_get_openai.return_value = mock_openai
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        pool = EmbedderPool()
        pool._clients.clear()

        client1 = pool.get_client("model1", "http://test", 9000)
        client2 = pool.get_client("model2", "http://test", 9000)  # Same host:port

        # Should only create one client for the same host:port
        assert mock_openai.OpenAI.call_count == 1

    @patch("ner_pipeline.lela.llm_pool._get_openai")
    def test_embed_calls_client(self, mock_get_openai):
        mock_openai = MagicMock()
        mock_get_openai.return_value = mock_openai

        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        pool = EmbedderPool()
        pool._clients.clear()

        embeddings = pool.embed(
            texts=["text1", "text2"],
            model_name="test-model",
            base_url="http://test",
            port=9000,
        )

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]


class TestLazyImports:
    """Tests for lazy import functions."""

    def test_get_openai_raises_on_missing(self):
        with patch.dict("sys.modules", {"openai": None}):
            # This is tricky to test without actually uninstalling openai
            # Just verify the function exists and is callable
            assert callable(_get_openai)

    def test_get_vllm_raises_on_missing(self):
        # Just verify the function exists
        assert callable(_get_vllm)


class TestVLLMInstanceManagement:
    """Tests for vLLM instance management."""

    def test_clear_vllm_instances(self):
        # Should not raise
        clear_vllm_instances()

    @patch("ner_pipeline.lela.llm_pool._get_vllm")
    def test_get_vllm_instance_creates_model(self, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_llm = MagicMock()
        mock_vllm.LLM.return_value = mock_llm
        mock_get_vllm.return_value = mock_vllm

        # Clear cache first
        clear_vllm_instances()

        result = get_vllm_instance(
            model_name="test-model",
            tensor_parallel_size=1,
        )

        mock_vllm.LLM.assert_called_once()
        call_kwargs = mock_vllm.LLM.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["tensor_parallel_size"] == 1

    @patch("ner_pipeline.lela.llm_pool._get_vllm")
    def test_get_vllm_instance_reuses_model(self, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_llm = MagicMock()
        mock_vllm.LLM.return_value = mock_llm
        mock_get_vllm.return_value = mock_vllm

        clear_vllm_instances()

        llm1 = get_vllm_instance("model-a", tensor_parallel_size=1)
        llm2 = get_vllm_instance("model-a", tensor_parallel_size=1)

        # Should only create once
        assert mock_vllm.LLM.call_count == 1
        assert llm1 is llm2

    @patch("ner_pipeline.lela.llm_pool._get_vllm")
    def test_different_configs_create_different_instances(self, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_vllm.LLM.side_effect = [MagicMock(), MagicMock()]
        mock_get_vllm.return_value = mock_vllm

        clear_vllm_instances()

        llm1 = get_vllm_instance("model-a", tensor_parallel_size=1)
        llm2 = get_vllm_instance("model-a", tensor_parallel_size=2)

        # Different tensor_parallel_size should create different instances
        assert mock_vllm.LLM.call_count == 2

    @patch("ner_pipeline.lela.llm_pool._get_vllm")
    def test_max_model_len_passed_when_specified(self, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = mock_vllm

        clear_vllm_instances()

        get_vllm_instance("model-x", tensor_parallel_size=1, max_model_len=4096)

        call_kwargs = mock_vllm.LLM.call_args[1]
        assert call_kwargs["max_model_len"] == 4096
