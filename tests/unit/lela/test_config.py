"""Unit tests for LELA config module."""

import pytest

from lela.lela import config


class TestLELAConfig:
    """Tests for LELA configuration constants."""

    def test_ner_labels_is_list(self):
        assert isinstance(config.NER_LABELS, list)
        assert len(config.NER_LABELS) > 0

    def test_ner_labels_contains_expected_types(self):
        expected = {"person", "organization", "location"}
        actual = set(config.NER_LABELS)
        assert expected.issubset(actual)

    def test_default_models_are_strings(self):
        assert isinstance(config.DEFAULT_GLINER_MODEL, str)
        assert isinstance(config.DEFAULT_LLM_MODEL, str)
        assert isinstance(config.DEFAULT_EMBEDDER_MODEL, str)
        assert isinstance(config.DEFAULT_RERANKER_MODEL, str)

    def test_default_models_have_org_prefix(self):
        # Model IDs should have org/model format
        assert "/" in config.DEFAULT_GLINER_MODEL
        assert "/" in config.DEFAULT_LLM_MODEL
        assert "/" in config.DEFAULT_EMBEDDER_MODEL

    def test_top_k_values_are_positive_integers(self):
        assert isinstance(config.CANDIDATES_TOP_K, int)
        assert isinstance(config.RERANKER_TOP_K, int)
        assert config.CANDIDATES_TOP_K > 0
        assert config.RERANKER_TOP_K > 0

    def test_candidates_top_k_greater_than_reranker(self):
        # We retrieve more candidates than we keep after reranking
        assert config.CANDIDATES_TOP_K > config.RERANKER_TOP_K

    def test_span_markers_are_strings(self):
        assert isinstance(config.SPAN_OPEN, str)
        assert isinstance(config.SPAN_CLOSE, str)
        assert len(config.SPAN_OPEN) == 1
        assert len(config.SPAN_CLOSE) == 1

    def test_not_an_entity_value(self):
        assert config.NOT_AN_ENTITY == ""

    def test_task_descriptions_are_nonempty(self):
        assert len(config.RETRIEVER_TASK) > 0
        assert len(config.RERANKER_TASK) > 0

    def test_generation_config_has_max_tokens(self):
        assert "max_tokens" in config.DEFAULT_GENERATION_CONFIG
        assert config.DEFAULT_GENERATION_CONFIG["max_tokens"] > 0

    def test_tensor_parallel_size_default(self):
        assert config.DEFAULT_TENSOR_PARALLEL_SIZE >= 1

    def test_model_tuples_have_vram_gb(self):
        """All model tuples should have (model_id, display_name, vram_gb)."""
        for model_list in (
            config.AVAILABLE_LLM_MODELS,
            config.AVAILABLE_EMBEDDING_MODELS,
            config.AVAILABLE_CROSS_ENCODER_MODELS,
            config.AVAILABLE_VLLM_RERANKER_MODELS,
        ):
            for entry in model_list:
                assert len(entry) == 3, f"Expected 3-tuple, got {entry}"
                model_id, display_name, vram_gb = entry
                assert isinstance(model_id, str)
                assert isinstance(display_name, str)
                assert isinstance(vram_gb, (int, float))
                assert vram_gb > 0

    def test_get_model_vram_gb_known_models(self):
        """get_model_vram_gb should return correct values for known models."""
        for model_list in (
            config.AVAILABLE_LLM_MODELS,
            config.AVAILABLE_EMBEDDING_MODELS,
            config.AVAILABLE_CROSS_ENCODER_MODELS,
            config.AVAILABLE_VLLM_RERANKER_MODELS,
        ):
            for model_id, _, expected_vram in model_list:
                assert config.get_model_vram_gb(model_id) == expected_vram

    def test_get_model_vram_gb_unknown_model(self):
        """get_model_vram_gb should return 2.0 for unknown models."""
        assert config.get_model_vram_gb("unknown/model") == 2.0

    def test_default_gliner_vram_gb(self):
        assert isinstance(config.DEFAULT_GLINER_VRAM_GB, (int, float))
        assert config.DEFAULT_GLINER_VRAM_GB > 0
