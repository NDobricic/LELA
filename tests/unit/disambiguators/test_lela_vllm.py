"""Unit tests for LELAvLLMDisambiguator."""

from unittest.mock import MagicMock, patch
import json
import os
import tempfile

import pytest

from ner_pipeline.types import Candidate, Document, Entity, Mention
from ner_pipeline.knowledge_bases.lela_jsonl import LELAJSONLKnowledgeBase


class TestLELAvLLMDisambiguator:
    """Tests for LELAvLLMDisambiguator class."""

    @pytest.fixture
    def lela_kb_data(self) -> list[dict]:
        return [
            {"title": "Barack Obama", "description": "44th US President"},
            {"title": "Michelle Obama", "description": "Former First Lady"},
            {"title": "Joe Biden", "description": "46th US President"},
        ]

    @pytest.fixture
    def temp_kb_file(self, lela_kb_data: list[dict]) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in lela_kb_data:
                f.write(json.dumps(item) + "\n")
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def kb(self, temp_kb_file: str) -> LELAJSONLKnowledgeBase:
        return LELAJSONLKnowledgeBase(path=temp_kb_file)

    @pytest.fixture
    def sample_candidates(self) -> list[Candidate]:
        return [
            Candidate(entity_id="Barack Obama", score=0.9, description="44th US President"),
            Candidate(entity_id="Michelle Obama", score=0.8, description="Former First Lady"),
            Candidate(entity_id="Joe Biden", score=0.7, description="46th US President"),
        ]

    @pytest.fixture
    def sample_mention(self) -> Mention:
        return Mention(start=0, end=5, text="Obama")

    @pytest.fixture
    def sample_doc(self) -> Document:
        return Document(
            id="test-doc",
            text="Obama was the 44th president of the United States."
        )

    @patch("ner_pipeline.disambiguators.lela_vllm._get_vllm")
    @patch("ner_pipeline.disambiguators.lela_vllm.get_vllm_instance")
    def test_requires_knowledge_base(self, mock_get_instance, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())
        mock_get_instance.return_value = MagicMock()

        from ner_pipeline.disambiguators.lela_vllm import LELAvLLMDisambiguator
        with pytest.raises(ValueError, match="requires a knowledge base"):
            LELAvLLMDisambiguator(kb=None)

    @patch("ner_pipeline.disambiguators.lela_vllm._get_vllm")
    @patch("ner_pipeline.disambiguators.lela_vllm.get_vllm_instance")
    def test_disambiguate_returns_entity(
        self, mock_get_instance, mock_get_vllm, kb, sample_candidates, sample_mention, sample_doc
    ):
        mock_vllm = MagicMock()
        mock_sampling_params = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, mock_sampling_params)

        mock_llm = MagicMock()
        mock_get_instance.return_value = mock_llm

        # LLM returns "answer": 1 (first candidate)
        mock_output = MagicMock()
        mock_output.text = '"answer": 1'
        mock_response = MagicMock()
        mock_response.outputs = [mock_output]
        mock_llm.chat.return_value = [mock_response]

        from ner_pipeline.disambiguators.lela_vllm import LELAvLLMDisambiguator
        disambiguator = LELAvLLMDisambiguator(kb=kb)

        result = disambiguator.disambiguate(sample_mention, sample_candidates, sample_doc)

        assert result is not None
        assert isinstance(result, Entity)
        assert result.id == "Barack Obama"

    @patch("ner_pipeline.disambiguators.lela_vllm._get_vllm")
    @patch("ner_pipeline.disambiguators.lela_vllm.get_vllm_instance")
    def test_disambiguate_empty_candidates(
        self, mock_get_instance, mock_get_vllm, kb, sample_mention, sample_doc
    ):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())
        mock_get_instance.return_value = MagicMock()

        from ner_pipeline.disambiguators.lela_vllm import LELAvLLMDisambiguator
        disambiguator = LELAvLLMDisambiguator(kb=kb)

        result = disambiguator.disambiguate(sample_mention, [], sample_doc)

        assert result is None

    @patch("ner_pipeline.disambiguators.lela_vllm._get_vllm")
    @patch("ner_pipeline.disambiguators.lela_vllm.get_vllm_instance")
    def test_single_candidate_no_none_option(
        self, mock_get_instance, mock_get_vllm, kb, sample_mention, sample_doc
    ):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())
        mock_get_instance.return_value = MagicMock()

        from ner_pipeline.disambiguators.lela_vllm import LELAvLLMDisambiguator
        disambiguator = LELAvLLMDisambiguator(kb=kb, add_none_candidate=False)

        candidates = [
            Candidate(entity_id="Barack Obama", score=0.9, description="44th US President"),
        ]

        result = disambiguator.disambiguate(sample_mention, candidates, sample_doc)

        # Should return the single candidate directly without LLM call
        assert result is not None
        assert result.id == "Barack Obama"

    @patch("ner_pipeline.disambiguators.lela_vllm._get_vllm")
    @patch("ner_pipeline.disambiguators.lela_vllm.get_vllm_instance")
    def test_answer_zero_returns_none(
        self, mock_get_instance, mock_get_vllm, kb, sample_candidates, sample_mention, sample_doc
    ):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())

        mock_llm = MagicMock()
        mock_get_instance.return_value = mock_llm

        # LLM returns "answer": 0 (none option)
        mock_output = MagicMock()
        mock_output.text = '"answer": 0'
        mock_response = MagicMock()
        mock_response.outputs = [mock_output]
        mock_llm.chat.return_value = [mock_response]

        from ner_pipeline.disambiguators.lela_vllm import LELAvLLMDisambiguator
        disambiguator = LELAvLLMDisambiguator(kb=kb, add_none_candidate=True)

        result = disambiguator.disambiguate(sample_mention, sample_candidates, sample_doc)

        assert result is None

    @patch("ner_pipeline.disambiguators.lela_vllm._get_vllm")
    @patch("ner_pipeline.disambiguators.lela_vllm.get_vllm_instance")
    def test_selects_second_candidate(
        self, mock_get_instance, mock_get_vllm, kb, sample_candidates, sample_mention, sample_doc
    ):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())

        mock_llm = MagicMock()
        mock_get_instance.return_value = mock_llm

        # LLM returns "answer": 2 (second candidate)
        mock_output = MagicMock()
        mock_output.text = '"answer": 2'
        mock_response = MagicMock()
        mock_response.outputs = [mock_output]
        mock_llm.chat.return_value = [mock_response]

        from ner_pipeline.disambiguators.lela_vllm import LELAvLLMDisambiguator
        disambiguator = LELAvLLMDisambiguator(kb=kb)

        result = disambiguator.disambiguate(sample_mention, sample_candidates, sample_doc)

        assert result is not None
        assert result.id == "Michelle Obama"

    @patch("ner_pipeline.disambiguators.lela_vllm._get_vllm")
    @patch("ner_pipeline.disambiguators.lela_vllm.get_vllm_instance")
    def test_out_of_range_answer_returns_none(
        self, mock_get_instance, mock_get_vllm, kb, sample_candidates, sample_mention, sample_doc
    ):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())

        mock_llm = MagicMock()
        mock_get_instance.return_value = mock_llm

        # LLM returns answer out of range
        mock_output = MagicMock()
        mock_output.text = '"answer": 99'
        mock_response = MagicMock()
        mock_response.outputs = [mock_output]
        mock_llm.chat.return_value = [mock_response]

        from ner_pipeline.disambiguators.lela_vllm import LELAvLLMDisambiguator
        disambiguator = LELAvLLMDisambiguator(kb=kb)

        result = disambiguator.disambiguate(sample_mention, sample_candidates, sample_doc)

        assert result is None

    @patch("ner_pipeline.disambiguators.lela_vllm._get_vllm")
    @patch("ner_pipeline.disambiguators.lela_vllm.get_vllm_instance")
    def test_invalid_output_format_returns_none(
        self, mock_get_instance, mock_get_vllm, kb, sample_candidates, sample_mention, sample_doc
    ):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())

        mock_llm = MagicMock()
        mock_get_instance.return_value = mock_llm

        # LLM returns unparseable output
        mock_output = MagicMock()
        mock_output.text = 'I think the answer is Barack Obama'
        mock_response = MagicMock()
        mock_response.outputs = [mock_output]
        mock_llm.chat.return_value = [mock_response]

        from ner_pipeline.disambiguators.lela_vllm import LELAvLLMDisambiguator
        disambiguator = LELAvLLMDisambiguator(kb=kb)

        result = disambiguator.disambiguate(sample_mention, sample_candidates, sample_doc)

        # Should return None (answer 0) when parsing fails
        assert result is None

    @patch("ner_pipeline.disambiguators.lela_vllm._get_vllm")
    @patch("ner_pipeline.disambiguators.lela_vllm.get_vllm_instance")
    def test_llm_error_returns_none(
        self, mock_get_instance, mock_get_vllm, kb, sample_candidates, sample_mention, sample_doc
    ):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())

        mock_llm = MagicMock()
        mock_get_instance.return_value = mock_llm
        mock_llm.chat.side_effect = Exception("LLM error")

        from ner_pipeline.disambiguators.lela_vllm import LELAvLLMDisambiguator
        disambiguator = LELAvLLMDisambiguator(kb=kb)

        result = disambiguator.disambiguate(sample_mention, sample_candidates, sample_doc)

        assert result is None


class TestLELAvLLMDisambiguatorParsing:
    """Tests for output parsing logic."""

    def test_parse_output_standard_format(self):
        from ner_pipeline.disambiguators.lela_vllm import LELAvLLMDisambiguator
        assert LELAvLLMDisambiguator._parse_output('"answer": 1') == 1
        assert LELAvLLMDisambiguator._parse_output('"answer": 2') == 2
        assert LELAvLLMDisambiguator._parse_output('"answer": 0') == 0

    def test_parse_output_without_quotes(self):
        from ner_pipeline.disambiguators.lela_vllm import LELAvLLMDisambiguator
        assert LELAvLLMDisambiguator._parse_output('answer: 1') == 1
        assert LELAvLLMDisambiguator._parse_output('answer: 3') == 3

    def test_parse_output_with_surrounding_text(self):
        from ner_pipeline.disambiguators.lela_vllm import LELAvLLMDisambiguator
        assert LELAvLLMDisambiguator._parse_output('Based on context, "answer": 2') == 2
        assert LELAvLLMDisambiguator._parse_output('{"answer": 1}') == 1

    def test_parse_output_invalid_returns_zero(self):
        from ner_pipeline.disambiguators.lela_vllm import LELAvLLMDisambiguator
        assert LELAvLLMDisambiguator._parse_output('no answer here') == 0
        assert LELAvLLMDisambiguator._parse_output('') == 0
        assert LELAvLLMDisambiguator._parse_output('Barack Obama') == 0


class TestLELAvLLMDisambiguatorConfig:
    """Tests for disambiguator configuration."""

    @patch("ner_pipeline.disambiguators.lela_vllm._get_vllm")
    @patch("ner_pipeline.disambiguators.lela_vllm.get_vllm_instance")
    def test_custom_model_name(self, mock_get_instance, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())
        mock_get_instance.return_value = MagicMock()

        # Create minimal KB
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"title": "Test", "description": "Test"}\n')
            path = f.name

        try:
            kb = LELAJSONLKnowledgeBase(path=path)

            from ner_pipeline.disambiguators.lela_vllm import LELAvLLMDisambiguator
            LELAvLLMDisambiguator(kb=kb, model_name="custom/model")

            # Check that get_vllm_instance was called with correct model
            mock_get_instance.assert_called_once()
            call_kwargs = mock_get_instance.call_args[1]
            assert call_kwargs["model_name"] == "custom/model"
        finally:
            os.unlink(path)

    @patch("ner_pipeline.disambiguators.lela_vllm._get_vllm")
    @patch("ner_pipeline.disambiguators.lela_vllm.get_vllm_instance")
    def test_tensor_parallel_size(self, mock_get_instance, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())
        mock_get_instance.return_value = MagicMock()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"title": "Test", "description": "Test"}\n')
            path = f.name

        try:
            kb = LELAJSONLKnowledgeBase(path=path)

            from ner_pipeline.disambiguators.lela_vllm import LELAvLLMDisambiguator
            LELAvLLMDisambiguator(kb=kb, tensor_parallel_size=4)

            call_kwargs = mock_get_instance.call_args[1]
            assert call_kwargs["tensor_parallel_size"] == 4
        finally:
            os.unlink(path)

    @patch("ner_pipeline.disambiguators.lela_vllm._get_vllm")
    @patch("ner_pipeline.disambiguators.lela_vllm.get_vllm_instance")
    def test_custom_system_prompt(self, mock_get_instance, mock_get_vllm):
        mock_vllm = MagicMock()
        mock_get_vllm.return_value = (mock_vllm, MagicMock())
        mock_get_instance.return_value = MagicMock()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"title": "Test", "description": "Test"}\n')
            path = f.name

        try:
            kb = LELAJSONLKnowledgeBase(path=path)

            from ner_pipeline.disambiguators.lela_vllm import LELAvLLMDisambiguator
            custom_prompt = "Custom disambiguation prompt"
            disambiguator = LELAvLLMDisambiguator(kb=kb, system_prompt=custom_prompt)

            assert disambiguator.system_prompt == custom_prompt
        finally:
            os.unlink(path)
