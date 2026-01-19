"""Prompt templates for LELA disambiguation."""

from typing import List, Optional, Tuple

DEFAULT_SYSTEM_PROMPT = """You are an expert designed to disambiguate entities in text, taking into account the overall context and a list of entity candidates. You are provided with an input text that includes a full contextual narrative, a marked mention enclosed in square brackets, and a list of candidates, each preceded by an index number.
Your task is to determine the most appropriate entity from the candidates based on the context and candidate entity descriptions.
Please show your choice in the answer field with only the choice index number, e.g., "answer": 3."""


def create_disambiguation_messages(
    marked_text: str,
    candidates: List[Tuple[str, str]],
    system_prompt: Optional[str] = None,
    query_prompt: Optional[str] = None,
    add_none_candidate: bool = True,
    add_descriptions: bool = True,
) -> List[dict]:
    """
    Create message list for LLM disambiguation.

    Args:
        marked_text: Text with mention marked using [brackets]
        candidates: List of (entity_id, description) tuples
        system_prompt: Optional custom system prompt
        query_prompt: Optional additional query context
        add_none_candidate: Whether to include "None" option
        add_descriptions: Whether to include entity descriptions

    Returns:
        List of message dicts for chat API
    """
    messages = []

    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    messages.append({"role": "system", "content": system_prompt})

    if query_prompt:
        messages.append({"role": "user", "content": query_prompt})

    # Build candidate list string
    none_option = "0. None of the listed candidates\n" if add_none_candidate else ""

    candidate_lines = []
    for i, (entity_id, description) in enumerate(candidates):
        if add_descriptions and description:
            candidate_lines.append(f"{i + 1}. {entity_id} - {description}")
        else:
            candidate_lines.append(f"{i + 1}. {entity_id}")

    candidate_str = none_option + "\n".join(candidate_lines)

    user_message = f"Input text: {marked_text}\nList of candidate entities:\n{candidate_str}"
    messages.append({"role": "user", "content": user_message})

    return messages


def mark_mention_in_text(
    text: str,
    start: int,
    end: int,
    open_marker: str = "[",
    close_marker: str = "]",
) -> str:
    """
    Mark a mention in text with brackets.

    Args:
        text: Full text
        start: Mention start offset
        end: Mention end offset
        open_marker: Opening bracket character
        close_marker: Closing bracket character

    Returns:
        Text with mention marked
    """
    return f"{text[:start]}{open_marker}{text[start:end]}{close_marker}{text[end:]}"
