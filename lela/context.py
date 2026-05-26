"""
Context extraction utilities for entity linking.

Provides functions to extract surrounding context for mentions,
which improves disambiguation accuracy.
"""
import re
from typing import Optional

from spacy.tokens import Doc, Span

from lela.lela.config import SPAN_OPEN, SPAN_CLOSE


def build_marked_text(doc: Doc, ent: Span, context_window: int = 0) -> str:
    """Return the mention marked with SPAN_OPEN/SPAN_CLOSE, optionally cropped
    to ``context_window`` tokens (half on each side). 0 / negative = full doc.
    """
    if not context_window or context_window <= 0:
        text = doc.text
        return f"{text[:ent.start_char]}{SPAN_OPEN}{ent.text}{SPAN_CLOSE}{text[ent.end_char:]}"

    half = context_window // 2
    left = max(0, ent.start - half)
    right = min(len(doc), ent.end + half)
    left_text = doc[left : ent.start].text
    right_text = doc[ent.end : right].text
    return f"{left_text} {SPAN_OPEN}{ent.text}{SPAN_CLOSE} {right_text}".strip()


def extract_sentence_context(
    text: str,
    start: int,
    end: int,
    max_sentences: int = 1,
) -> str:
    """
    Extract the sentence(s) containing the mention.
    
    Args:
        text: Full document text
        start: Mention start offset
        end: Mention end offset
        max_sentences: Number of sentences to include before/after (default: 1)
    
    Returns:
        The sentence(s) containing the mention
    """
    sentence_pattern = re.compile(r'[.!?]+\s+|\n\n+')
    
    sentences = []
    current_start = 0
    
    for match in sentence_pattern.finditer(text):
        sentences.append((current_start, match.end()))
        current_start = match.end()
    
    if current_start < len(text):
        sentences.append((current_start, len(text)))
    
    if not sentences:
        return text
    
    mention_sentence_idx = None
    for i, (sent_start, sent_end) in enumerate(sentences):
        if sent_start <= start < sent_end:
            mention_sentence_idx = i
            break
    
    if mention_sentence_idx is None:
        return text[max(0, start - 100):min(len(text), end + 100)]
    
    context_start_idx = max(0, mention_sentence_idx - max_sentences)
    context_end_idx = min(len(sentences), mention_sentence_idx + max_sentences + 1)
    
    context_start = sentences[context_start_idx][0]
    context_end = sentences[context_end_idx - 1][1]
    
    return text[context_start:context_end].strip()


def extract_window_context(
    text: str,
    start: int,
    end: int,
    window_chars: int = 150,
) -> str:
    """
    Extract a fixed character window around the mention.
    
    Args:
        text: Full document text
        start: Mention start offset
        end: Mention end offset
        window_chars: Number of characters before/after mention
    
    Returns:
        Text window around the mention
    """
    context_start = max(0, start - window_chars)
    context_end = min(len(text), end + window_chars)
    
    if context_start > 0:
        space_idx = text.find(' ', context_start)
        if space_idx != -1 and space_idx < start:
            context_start = space_idx + 1
    
    if context_end < len(text):
        space_idx = text.rfind(' ', end, context_end)
        if space_idx != -1:
            context_end = space_idx
    
    return text[context_start:context_end].strip()


def extract_context(
    text: str,
    start: int,
    end: int,
    mode: str = "sentence",
    **kwargs,
) -> str:
    """
    Extract context around a mention using the specified mode.
    
    Args:
        text: Full document text
        start: Mention start offset
        end: Mention end offset
        mode: "sentence" or "window"
        **kwargs: Additional arguments for the specific mode
    
    Returns:
        Context string
    """
    if mode == "sentence":
        return extract_sentence_context(text, start, end, **kwargs)
    elif mode == "window":
        return extract_window_context(text, start, end, **kwargs)
    else:
        raise ValueError(f"Unknown context mode: {mode}")

