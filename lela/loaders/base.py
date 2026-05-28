from typing import Iterator, Protocol

from lela._types import Document


class DocumentLoader(Protocol):
    """Loads documents from a path."""

    def load(self, path: str) -> Iterator[Document]:
        ...

