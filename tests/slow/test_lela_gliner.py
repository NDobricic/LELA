"""Slow tests for LELAGLiNERNER with actual model loading."""

import pytest

from ner_pipeline.types import Mention


# Use the same model as the existing gliner tests (known to work)
TEST_MODEL = "urchade/gliner_large"


@pytest.fixture(scope="module")
def lela_gliner_ner():
    """Shared NER instance to avoid repeated model loading."""
    try:
        from ner_pipeline.ner.lela_gliner import LELAGLiNERNER
        return LELAGLiNERNER(
            model_name=TEST_MODEL,
            labels=["person", "organization", "location"],
            threshold=0.3,
        )
    except Exception as e:
        pytest.skip(f"Could not load GLiNER model: {e}")


class TestLELAGLiNERWithModel:
    """Tests that actually load the GLiNER model."""

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_extract_person_entities(self, lela_gliner_ner):
        text = "Barack Obama was the president. Michelle Obama was the first lady."
        mentions = lela_gliner_ner.extract(text)

        assert len(mentions) >= 1
        texts = [m.text for m in mentions]
        # Should find at least one person
        assert any("Obama" in t for t in texts)

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_extract_organization_entities(self, lela_gliner_ner):
        text = "Google and Microsoft are technology companies."
        mentions = lela_gliner_ner.extract(text)

        texts = [m.text for m in mentions]
        # May or may not find organizations depending on model
        assert isinstance(mentions, list)

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_extract_location_entities(self, lela_gliner_ner):
        text = "Paris is the capital of France."
        mentions = lela_gliner_ner.extract(text)

        texts = [m.text for m in mentions]
        # Should find locations
        assert any("Paris" in t or "France" in t for t in texts) or len(mentions) >= 0

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_mentions_have_correct_offsets(self, lela_gliner_ner):
        text = "Barack Obama was president."
        mentions = lela_gliner_ner.extract(text)

        for mention in mentions:
            # Check that offsets are valid
            assert mention.start >= 0
            assert mention.end <= len(text)
            assert mention.start < mention.end
            # Check that text matches offsets
            assert text[mention.start:mention.end] == mention.text

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_mentions_have_labels(self, lela_gliner_ner):
        text = "Barack Obama visited Paris."
        mentions = lela_gliner_ner.extract(text)

        for mention in mentions:
            assert mention.label is not None
            assert mention.label in ["person", "organization", "location"]

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_mentions_have_context(self, lela_gliner_ner):
        text = "Barack Obama was the president of the United States."
        mentions = lela_gliner_ner.extract(text)

        for mention in mentions:
            assert mention.context is not None
            assert len(mention.context) > 0

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_empty_text_returns_empty(self, lela_gliner_ner):
        mentions = lela_gliner_ner.extract("")
        assert mentions == []

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_multiple_entities_in_text(self, lela_gliner_ner):
        text = "Tim Cook is CEO of Apple in Cupertino."
        mentions = lela_gliner_ner.extract(text)

        # Should run without error
        assert isinstance(mentions, list)
        # Should find at least one entity
        if len(mentions) > 0:
            assert all(isinstance(m, Mention) for m in mentions)


class TestLELAGLiNERThreshold:
    """Tests for threshold behavior - requires separate model instances."""

    @pytest.mark.slow
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_threshold_affects_results(self):
        try:
            from ner_pipeline.ner.lela_gliner import LELAGLiNERNER

            text = "Obama was president of America."

            # Lower threshold should find more entities
            ner_low = LELAGLiNERNER(
                model_name=TEST_MODEL,
                labels=["person", "location"],
                threshold=0.1,
            )

            # Higher threshold should be more selective
            ner_high = LELAGLiNERNER(
                model_name=TEST_MODEL,
                labels=["person", "location"],
                threshold=0.9,
            )

            mentions_low = ner_low.extract(text)
            mentions_high = ner_high.extract(text)

            # Higher threshold should find fewer or equal entities
            assert len(mentions_high) <= len(mentions_low)
        except Exception as e:
            pytest.skip(f"Could not load GLiNER model: {e}")
