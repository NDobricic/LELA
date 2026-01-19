"""Unit tests for LELAGLiNERNER."""

from unittest.mock import MagicMock, patch

import pytest

from ner_pipeline.types import Mention


class TestLELAGLiNERNER:
    """Tests for LELAGLiNERNER class with mocked GLiNER."""

    @patch("ner_pipeline.ner.lela_gliner._get_gliner")
    def test_initialization_loads_model(self, mock_get_gliner):
        mock_gliner_class = MagicMock()
        mock_model = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model
        mock_get_gliner.return_value = mock_gliner_class

        from ner_pipeline.ner.lela_gliner import LELAGLiNERNER
        ner = LELAGLiNERNER()

        mock_gliner_class.from_pretrained.assert_called_once()

    @patch("ner_pipeline.ner.lela_gliner._get_gliner")
    def test_initialization_with_custom_model(self, mock_get_gliner):
        mock_gliner_class = MagicMock()
        mock_get_gliner.return_value = mock_gliner_class

        from ner_pipeline.ner.lela_gliner import LELAGLiNERNER
        ner = LELAGLiNERNER(model_name="custom/model")

        mock_gliner_class.from_pretrained.assert_called_once_with("custom/model")

    @patch("ner_pipeline.ner.lela_gliner._get_gliner")
    def test_initialization_with_custom_labels(self, mock_get_gliner):
        mock_gliner_class = MagicMock()
        mock_get_gliner.return_value = mock_gliner_class

        from ner_pipeline.ner.lela_gliner import LELAGLiNERNER
        custom_labels = ["person", "company"]
        ner = LELAGLiNERNER(labels=custom_labels)

        assert ner.labels == custom_labels

    @patch("ner_pipeline.ner.lela_gliner._get_gliner")
    def test_initialization_with_custom_threshold(self, mock_get_gliner):
        mock_gliner_class = MagicMock()
        mock_get_gliner.return_value = mock_gliner_class

        from ner_pipeline.ner.lela_gliner import LELAGLiNERNER
        ner = LELAGLiNERNER(threshold=0.7)

        assert ner.threshold == 0.7

    @patch("ner_pipeline.ner.lela_gliner._get_gliner")
    def test_extract_returns_mentions(self, mock_get_gliner):
        mock_gliner_class = MagicMock()
        mock_model = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model
        mock_get_gliner.return_value = mock_gliner_class

        # Mock predictions
        mock_model.predict_entities.return_value = [
            {"start": 0, "end": 12, "text": "Barack Obama", "label": "person"},
        ]

        from ner_pipeline.ner.lela_gliner import LELAGLiNERNER
        ner = LELAGLiNERNER()

        mentions = ner.extract("Barack Obama was president.")

        assert len(mentions) == 1
        assert all(isinstance(m, Mention) for m in mentions)

    @patch("ner_pipeline.ner.lela_gliner._get_gliner")
    def test_mention_has_correct_fields(self, mock_get_gliner):
        mock_gliner_class = MagicMock()
        mock_model = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model
        mock_get_gliner.return_value = mock_gliner_class

        mock_model.predict_entities.return_value = [
            {"start": 0, "end": 12, "text": "Barack Obama", "label": "person"},
        ]

        from ner_pipeline.ner.lela_gliner import LELAGLiNERNER
        ner = LELAGLiNERNER()

        mentions = ner.extract("Barack Obama was president.")

        mention = mentions[0]
        assert mention.start == 0
        assert mention.end == 12
        assert mention.text == "Barack Obama"
        assert mention.label == "person"

    @patch("ner_pipeline.ner.lela_gliner._get_gliner")
    def test_mention_has_context(self, mock_get_gliner):
        mock_gliner_class = MagicMock()
        mock_model = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model
        mock_get_gliner.return_value = mock_gliner_class

        mock_model.predict_entities.return_value = [
            {"start": 0, "end": 12, "text": "Barack Obama", "label": "person"},
        ]

        from ner_pipeline.ner.lela_gliner import LELAGLiNERNER
        ner = LELAGLiNERNER()

        mentions = ner.extract("Barack Obama was president.")

        # Context should be extracted
        assert mentions[0].context is not None

    @patch("ner_pipeline.ner.lela_gliner._get_gliner")
    def test_extract_multiple_entities(self, mock_get_gliner):
        mock_gliner_class = MagicMock()
        mock_model = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model
        mock_get_gliner.return_value = mock_gliner_class

        mock_model.predict_entities.return_value = [
            {"start": 0, "end": 12, "text": "Barack Obama", "label": "person"},
            {"start": 30, "end": 43, "text": "United States", "label": "location"},
        ]

        from ner_pipeline.ner.lela_gliner import LELAGLiNERNER
        ner = LELAGLiNERNER()

        mentions = ner.extract("Barack Obama was president of United States.")

        assert len(mentions) == 2
        assert mentions[0].text == "Barack Obama"
        assert mentions[1].text == "United States"

    @patch("ner_pipeline.ner.lela_gliner._get_gliner")
    def test_extract_empty_text(self, mock_get_gliner):
        mock_gliner_class = MagicMock()
        mock_model = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model
        mock_get_gliner.return_value = mock_gliner_class

        from ner_pipeline.ner.lela_gliner import LELAGLiNERNER
        ner = LELAGLiNERNER()

        mentions = ner.extract("")

        assert mentions == []
        # Should not call predict on empty text
        mock_model.predict_entities.assert_not_called()

    @patch("ner_pipeline.ner.lela_gliner._get_gliner")
    def test_context_mode_parameter(self, mock_get_gliner):
        mock_gliner_class = MagicMock()
        mock_get_gliner.return_value = mock_gliner_class

        from ner_pipeline.ner.lela_gliner import LELAGLiNERNER
        ner = LELAGLiNERNER(context_mode="window")

        assert ner.context_mode == "window"

    @patch("ner_pipeline.ner.lela_gliner._get_gliner")
    def test_default_labels_from_config(self, mock_get_gliner):
        mock_gliner_class = MagicMock()
        mock_get_gliner.return_value = mock_gliner_class

        from ner_pipeline.ner.lela_gliner import LELAGLiNERNER
        from ner_pipeline.lela.config import NER_LABELS

        ner = LELAGLiNERNER()

        assert ner.labels == list(NER_LABELS)

    @patch("ner_pipeline.ner.lela_gliner._get_gliner")
    def test_threshold_passed_to_predict(self, mock_get_gliner):
        mock_gliner_class = MagicMock()
        mock_model = MagicMock()
        mock_gliner_class.from_pretrained.return_value = mock_model
        mock_get_gliner.return_value = mock_gliner_class

        mock_model.predict_entities.return_value = []

        from ner_pipeline.ner.lela_gliner import LELAGLiNERNER
        ner = LELAGLiNERNER(threshold=0.7, labels=["person"])

        ner.extract("Test text")

        mock_model.predict_entities.assert_called_once_with(
            "Test text",
            labels=["person"],
            threshold=0.7,
        )
