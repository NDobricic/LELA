"""NER models."""

from .spacy import SpacyNER  # noqa: F401
from .gliner import GLiNERNER  # noqa: F401
from .transformers import TransformersNER  # noqa: F401
from .simple import SimpleRegexNER  # noqa: F401
from .lela_gliner import LELAGLiNERNER  # noqa: F401

