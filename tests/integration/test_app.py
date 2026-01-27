"""Integration tests for the Gradio web application (app.py)."""

import pytest

from app import (
    compute_linking_stats,
    format_highlighted_text,
    get_available_components,
    run_pipeline,
)
from tests.conftest import MockGradioFile, MockGradioProgress


@pytest.mark.integration
class TestFormatHighlightedText:
    """Tests for format_highlighted_text function."""

    def test_format_empty_entities(self):
        """No entities returns single tuple with full text."""
        result = {"text": "Hello world", "entities": []}
        highlighted = format_highlighted_text(result)
        assert highlighted == [("Hello world", None)]

    def test_format_single_entity(self):
        """One entity is highlighted correctly."""
        result = {
            "text": "Barack Obama was president.",
            "entities": [
                {
                    "text": "Barack Obama",
                    "start": 0,
                    "end": 12,
                    "label": "PERSON",
                    "entity_title": "Barack Obama",
                }
            ],
        }
        highlighted = format_highlighted_text(result)
        assert len(highlighted) == 2
        assert highlighted[0] == ("Barack Obama", "PERSON: Barack Obama")
        assert highlighted[1] == (" was president.", None)

    def test_format_multiple_entities(self):
        """Multiple entities with gaps are formatted correctly."""
        result = {
            "text": "Albert Einstein was born in Germany.",
            "entities": [
                {
                    "text": "Albert Einstein",
                    "start": 0,
                    "end": 15,
                    "label": "PERSON",
                    "entity_title": "Albert Einstein",
                },
                {
                    "text": "Germany",
                    "start": 28,
                    "end": 35,
                    "label": "GPE",
                    "entity_title": "Germany",
                },
            ],
        }
        highlighted = format_highlighted_text(result)
        assert len(highlighted) == 4
        assert highlighted[0] == ("Albert Einstein", "PERSON: Albert Einstein")
        assert highlighted[1] == (" was born in ", None)
        assert highlighted[2] == ("Germany", "GPE: Germany")
        assert highlighted[3] == (".", None)

    def test_format_linked_entity(self):
        """Linked entity shows 'LABEL: Title' format."""
        result = {
            "text": "Obama",
            "entities": [
                {
                    "text": "Obama",
                    "start": 0,
                    "end": 5,
                    "label": "PERSON",
                    "entity_title": "Barack Obama",
                }
            ],
        }
        highlighted = format_highlighted_text(result)
        assert highlighted[0] == ("Obama", "PERSON: Barack Obama")

    def test_format_unlinked_entity(self):
        """Unlinked entity shows 'LABEL [NOT IN KB]' format."""
        result = {
            "text": "John Doe is here.",
            "entities": [
                {
                    "text": "John Doe",
                    "start": 0,
                    "end": 8,
                    "label": "PERSON",
                    "entity_title": None,
                }
            ],
        }
        highlighted = format_highlighted_text(result)
        assert highlighted[0] == ("John Doe", "PERSON [NOT IN KB]")

    def test_format_entity_at_boundaries(self):
        """Entity at start and end of text are handled correctly."""
        result = {
            "text": "Obama",
            "entities": [
                {
                    "text": "Obama",
                    "start": 0,
                    "end": 5,
                    "label": "PERSON",
                    "entity_title": "Barack Obama",
                }
            ],
        }
        highlighted = format_highlighted_text(result)
        assert len(highlighted) == 1
        assert highlighted[0] == ("Obama", "PERSON: Barack Obama")

    def test_format_missing_label_uses_ent(self):
        """Missing label defaults to 'ENT'."""
        result = {
            "text": "Entity here.",
            "entities": [
                {
                    "text": "Entity",
                    "start": 0,
                    "end": 6,
                    "entity_title": None,
                }
            ],
        }
        highlighted = format_highlighted_text(result)
        assert highlighted[0] == ("Entity", "ENT [NOT IN KB]")


@pytest.mark.integration
class TestComputeLinkingStats:
    """Tests for compute_linking_stats function."""

    def test_stats_no_entities(self):
        """No entities returns appropriate message."""
        result = {"entities": []}
        stats = compute_linking_stats(result)
        assert stats == "No entities found."

    def test_stats_all_linked(self):
        """All entities linked shows 100%."""
        result = {
            "entities": [
                {"entity_title": "Entity 1", "linking_confidence": 0.9},
                {"entity_title": "Entity 2", "linking_confidence": 0.8},
            ]
        }
        stats = compute_linking_stats(result)
        assert "Linked to KB: 2 (100.0%)" in stats
        assert "Not in KB: 0 (0.0%)" in stats

    def test_stats_none_linked(self):
        """No entities linked shows 0%."""
        result = {
            "entities": [
                {"entity_title": None},
                {"entity_title": None},
            ]
        }
        stats = compute_linking_stats(result)
        assert "Linked to KB: 0 (0.0%)" in stats
        assert "Not in KB: 2 (100.0%)" in stats

    def test_stats_partial_linked(self):
        """Mixed linking shows correct percentages."""
        result = {
            "entities": [
                {"entity_title": "Entity 1", "linking_confidence": 0.9},
                {"entity_title": None},
                {"entity_title": "Entity 3", "linking_confidence": 0.7},
                {"entity_title": None},
            ]
        }
        stats = compute_linking_stats(result)
        assert "Total entities: 4" in stats
        assert "Linked to KB: 2 (50.0%)" in stats
        assert "Not in KB: 2 (50.0%)" in stats

    def test_stats_with_confidence(self):
        """Average confidence is calculated correctly."""
        result = {
            "entities": [
                {"entity_title": "Entity 1", "linking_confidence": 0.9},
                {"entity_title": "Entity 2", "linking_confidence": 0.7},
            ]
        }
        stats = compute_linking_stats(result)
        assert "Avg. confidence (linked): 0.800" in stats

    def test_stats_empty_result_dict(self):
        """Empty result dict returns no entities message."""
        result = {}
        stats = compute_linking_stats(result)
        assert stats == "No entities found."


@pytest.mark.integration
class TestGetAvailableComponents:
    """Tests for get_available_components function."""

    def test_returns_all_categories(self):
        """Returns all expected component categories."""
        components = get_available_components()
        expected_keys = {"loaders", "ner", "candidates", "rerankers", "disambiguators", "knowledge_bases"}
        assert set(components.keys()) == expected_keys

    def test_disambiguators_includes_vllm(self):
        """Disambiguators always includes lela_vllm."""
        components = get_available_components()
        assert "lela_vllm" in components["disambiguators"]

    def test_ner_includes_expected_types(self):
        """NER includes expected types."""
        components = get_available_components()
        assert "simple" in components["ner"]
        assert "spacy" in components["ner"]
        assert "gliner" in components["ner"]

    def test_candidates_includes_expected_types(self):
        """Candidate generators include expected types."""
        components = get_available_components()
        assert "fuzzy" in components["candidates"]
        assert "bm25" in components["candidates"]


@pytest.mark.integration
class TestRunPipeline:
    """Tests for run_pipeline function."""

    def test_run_pipeline_text_input(self, mock_kb_file: MockGradioFile, mock_progress: MockGradioProgress):
        """Pipeline processes text input correctly."""
        text_input = "Barack Obama was president."
        highlighted, stats, result = run_pipeline(
            text_input=text_input,
            file_input=None,
            kb_file=mock_kb_file,
            loader_type="text",
            ner_type="simple",
            spacy_model="en_core_web_sm",
            gliner_model="urchade/gliner_large",
            gliner_labels="",
            gliner_threshold=0.5,
            simple_min_len=3,
            cand_type="fuzzy",
            cand_top_k=10,
            cand_use_context=True,
            reranker_type="none",
            reranker_top_k=10,
            disambig_type="first",
            tournament_batch_size=4,
            tournament_shuffle=False,
            tournament_thinking=False,
            kb_type="custom",
            progress=mock_progress,
        )
        assert isinstance(highlighted, list)
        assert isinstance(stats, str)
        assert isinstance(result, dict)

    def test_run_pipeline_returns_tuple(self, mock_kb_file: MockGradioFile, mock_progress: MockGradioProgress):
        """Pipeline returns tuple of (highlighted, stats, result)."""
        text_input = "Test text."
        output = run_pipeline(
            text_input=text_input,
            file_input=None,
            kb_file=mock_kb_file,
            loader_type="text",
            ner_type="simple",
            spacy_model="en_core_web_sm",
            gliner_model="urchade/gliner_large",
            gliner_labels="",
            gliner_threshold=0.5,
            simple_min_len=3,
            cand_type="fuzzy",
            cand_top_k=10,
            cand_use_context=True,
            reranker_type="none",
            reranker_top_k=10,
            disambig_type="first",
            tournament_batch_size=4,
            tournament_shuffle=False,
            tournament_thinking=False,
            kb_type="custom",
            progress=mock_progress,
        )
        assert len(output) == 3

    def test_run_pipeline_result_structure(self, mock_kb_file: MockGradioFile, mock_progress: MockGradioProgress):
        """Result has text and entities keys."""
        text_input = "Barack Obama was president."
        _, _, result = run_pipeline(
            text_input=text_input,
            file_input=None,
            kb_file=mock_kb_file,
            loader_type="text",
            ner_type="simple",
            spacy_model="en_core_web_sm",
            gliner_model="urchade/gliner_large",
            gliner_labels="",
            gliner_threshold=0.5,
            simple_min_len=3,
            cand_type="fuzzy",
            cand_top_k=10,
            cand_use_context=True,
            reranker_type="none",
            reranker_top_k=10,
            disambig_type="first",
            tournament_batch_size=4,
            tournament_shuffle=False,
            tournament_thinking=False,
            kb_type="custom",
            progress=mock_progress,
        )
        assert "text" in result
        assert "entities" in result

    def test_run_pipeline_no_kb_error(self, mock_progress: MockGradioProgress):
        """Returns error without KB file."""
        highlighted, stats, result = run_pipeline(
            text_input="Some text",
            file_input=None,
            kb_file=None,
            loader_type="text",
            ner_type="simple",
            spacy_model="en_core_web_sm",
            gliner_model="urchade/gliner_large",
            gliner_labels="",
            gliner_threshold=0.5,
            simple_min_len=3,
            cand_type="fuzzy",
            cand_top_k=10,
            cand_use_context=True,
            reranker_type="none",
            reranker_top_k=10,
            disambig_type="first",
            tournament_batch_size=4,
            tournament_shuffle=False,
            tournament_thinking=False,
            kb_type="custom",
            progress=mock_progress,
        )
        assert "error" in result
        assert "Knowledge Base" in result["error"]

    def test_run_pipeline_no_input_error(self, mock_kb_file: MockGradioFile, mock_progress: MockGradioProgress):
        """Returns error without text or file input."""
        highlighted, stats, result = run_pipeline(
            text_input="",
            file_input=None,
            kb_file=mock_kb_file,
            loader_type="text",
            ner_type="simple",
            spacy_model="en_core_web_sm",
            gliner_model="urchade/gliner_large",
            gliner_labels="",
            gliner_threshold=0.5,
            simple_min_len=3,
            cand_type="fuzzy",
            cand_top_k=10,
            cand_use_context=True,
            reranker_type="none",
            reranker_top_k=10,
            disambig_type="first",
            tournament_batch_size=4,
            tournament_shuffle=False,
            tournament_thinking=False,
            kb_type="custom",
            progress=mock_progress,
        )
        assert "error" in result
        assert "Input" in result["error"]
