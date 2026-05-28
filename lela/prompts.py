"""Prompt templates for LELA disambiguation."""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from lela.knowledge_bases.base import KnowledgeBase
    from lela._types import Candidate

DEFAULT_SYSTEM_PROMPT = f"""You are an expert designed to disambiguate entities in text, taking into account the overall context and a list of entity candidates.
You are provided with an input text that includes a full contextual narrative, the mention enclosed between the '[' and ']' characters, and a list of candidates, each preceded by an index number.
Your task is to determine the most appropriate entity from the candidates based on the context and candidate entity descriptions.
Please show your choice in the answer field with only the choice index number, e.g., "answer": 3."""


def create_disambiguation_messages(
    marked_text: str,
    candidates: List["Candidate"],
    kb: Optional["KnowledgeBase"] = None,
    system_prompt: Optional[str] = None,
    query_prompt: Optional[str] = None,
    add_none_candidate: bool = True,
    add_descriptions: bool = True,
) -> List[dict]:
    """
    Create message list for LLM disambiguation.

    Thinking-mode control is handled at the chat-template level via
    ``chat_template_kwargs={"enable_thinking": ...}`` by the caller, not here.
    """
    messages = []

    final_system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
    messages.append({"role": "system", "content": final_system_prompt})

    if query_prompt:
        messages.append({"role": "user", "content": query_prompt})

    none_option = "0. None of the listed candidates\n" if add_none_candidate else ""

    candidate_lines = []
    for i, candidate in enumerate(candidates):
        if kb:
            entity = kb.get_entity(candidate.entity_id)
            title = entity.title if entity else candidate.entity_id
        else:
            title = candidate.entity_id

        if add_descriptions and candidate.description:
            candidate_lines.append(f"{i + 1}. {title} - {candidate.description}")
        else:
            candidate_lines.append(f"{i + 1}. {title}")

    candidate_str = none_option + "\n".join(candidate_lines)

    user_message = (
        f"Input text: {marked_text}\nList of candidate entities:\n{candidate_str}"
    )

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
